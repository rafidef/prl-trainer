"""
Tiled / streaming NoisyGEMM PoW scan for full-scale mining (M=N=131072).

The dense correctness core (noisy_gemm.build_scan) materializes the whole
candidate grid — impossible at production scale (~134M tiles/scan). This module
streams instead, exactly as the CUDA kernel does:

  * The scan is split into row-batches of `rbatch` 64-row blocks. Each 64-row
    block holds 32 candidate tile_m (top half i=0..31, rows {tile_m, tile_m+32}).
  * Within a row-batch we stream the K dimension with a `lax.scan` over the G
    rank-sized k-blocks, carrying the running int32 accumulators for the top and
    bottom rows against *all* column-blocks at once (one big MXU matmul per step).
  * Each k-block contributes one XOR-folded uint32 per (candidate, col-block);
    these are folded into the 16-word transcripts (slot = g % 16).
  * After the k-scan we keyed-BLAKE3 every transcript and compare to the target.

The host driver loops over row-batches and stops at the first one containing a
share (first-hit early-exit). Only running accumulators (T x N int32) and the
per-k combined values (G x T x NC uint32) live in HBM — never the full grid.

Bit-identical to the NumPy reference (validated in tests/test_tiled_scan.py).

Optimizations over the original:
  * Transcript accumulation fused into lax.scan — eliminates the (G, T, NC)
    intermediate tensor from HBM.
  * Optional N-tiling (ncbatch) to reduce accumulator HBM traffic.
  * Auto-scaled rbatch for larger HBM (v4: 32 GB).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import blake3_jax as bj
from .noisy_gemm import _lex_argmin, _rotl32

_U32 = jnp.uint32
_BLOCK = 64


def _xor_reduce_last64(cacc: jax.Array, nc: int) -> jax.Array:
    """XOR-reduce each 64-wide column block of an (T, N) int32 accumulator.

    Returns (T, NC) uint32 — one folded word per (row, col-block).
    """
    T = cacc.shape[0]
    cacc_u = jax.lax.bitcast_convert_type(cacc, _U32).reshape(T, nc, 64)
    return jax.lax.reduce(cacc_u, _U32(0), jax.lax.bitwise_xor, dimensions=(2,))


@dataclass
class TiledScan:
    M: int
    N: int
    K: int
    rank: int
    rbatch: int          # number of 64-row blocks per batch
    ncbatch: int         # column-blocks per sub-batch (0 = all at once)
    rows_pattern: tuple
    cols_pattern: tuple
    scan_batch: object   # jitted f(An, Bn, key_words, target_words, rb0)
    debug_batch: object  # jitted f(An, Bn, rb0) -> transcripts (T, NC, 16)

    @property
    def num_batches(self) -> int:
        return self.M // (self.rbatch * _BLOCK)


def build_tiled_scan(M, N, K, rank, rows_pattern, cols_pattern,
                     rbatch=8, ncbatch=None) -> TiledScan:
    rows_pattern = tuple(int(x) for x in rows_pattern)
    cols_pattern = tuple(int(x) for x in cols_pattern)
    assert rows_pattern == (0, 32), "tiled scan specialized for rows_pattern [0,32]"
    assert list(cols_pattern) == list(range(64)), "tiled scan specialized for cols [0..63]"
    assert M % (rbatch * _BLOCK) == 0, "rbatch*64 must divide M"
    assert N % _BLOCK == 0 and K % rank == 0

    G = K // rank
    NC = N // _BLOCK            # number of 64-col blocks (= tile_n candidates)
    T = rbatch * 32            # candidate rows per batch (32 per 64-row block)
    span = rbatch * _BLOCK     # An rows pulled per batch

    # N-tiling: split NC columns into sub-batches of ncbatch each.
    # ncbatch=None or 0 means process all columns at once (original behavior).
    if ncbatch is None or ncbatch <= 0 or ncbatch >= NC:
        ncbatch_actual = NC  # no N-tiling
    else:
        # Round up to make NC divisible.
        while NC % ncbatch != 0 and ncbatch < NC:
            ncbatch += 1
        ncbatch_actual = ncbatch

    # Static map: flat candidate row c -> tile_m offset within the batch.
    c2off = np.array([(c // 32) * _BLOCK + (c % 32) for c in range(T)], dtype=np.int32)
    c2off_j = jnp.asarray(c2off)
    cb2n = jnp.asarray(np.arange(NC, dtype=np.int32) * _BLOCK)

    # Precompute slot indices for transcript folding.
    slot_indices = jnp.asarray(np.array([g % 16 for g in range(G)], dtype=np.int32))

    def _transcripts_fused(An, Bn, rb0):
        """Row-batch transcripts with transcript accumulation fused into lax.scan.

        The transcript (T, NC, 16) is carried directly in the scan state, so the
        (G, T, NC) combined intermediate never materializes in HBM.
        """
        rows = jax.lax.dynamic_slice(An, (rb0, 0), (span, K))     # (span, K) int8
        rows = rows.reshape(rbatch, _BLOCK, K)
        top = rows[:, :32, :].reshape(T, K)                       # (T, K)
        bot = rows[:, 32:, :].reshape(T, K)                       # (T, K)

        A2 = jnp.concatenate([top, bot], axis=0)                  # (2T, K)
        A2_g = jnp.transpose(A2.reshape(2 * T, G, rank), (1, 0, 2))   # (G, 2T, rank)
        Bn_g = Bn.reshape(G, rank, N)                                 # (G, rank, N)

        def step(carry, xs):
            acc, transcript = carry             # acc: (2T, N), transcript: (T, NC, 16)
            a_g, b_g, slot = xs                 # (2T,rank), (rank,N), scalar int32
            c = jax.lax.dot_general(
                a_g.astype(jnp.int8), b_g.astype(jnp.int8),
                (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
            acc = acc + c                                            # (2T, N)
            xr = _xor_reduce_last64(acc, NC)                         # (2T, NC)
            combined = xr[:T] ^ xr[T:]                               # (T, NC)
            # Fuse transcript fold: tr[:, :, slot] = rotl(tr[:, :, slot], 13) ^ combined
            old_slot = transcript[:, :, slot]
            new_slot = _rotl32(old_slot, 13) ^ combined
            transcript = transcript.at[:, :, slot].set(new_slot)
            return (acc, transcript), None       # No stacked output!

        acc0 = jnp.zeros((2 * T, N), jnp.int32)
        tr0 = jnp.zeros((T, NC, 16), _U32)
        (_, transcript), _ = jax.lax.scan(step, (acc0, tr0), (A2_g, Bn_g, slot_indices))

        return transcript                                             # (T, NC, 16)

    def _transcripts_ntiled(An, Bn, rb0):
        """N-tiled variant: process columns in sub-batches to reduce HBM traffic.

        For each column sub-batch, runs the full k-scan with a smaller accumulator.
        This trades more MXU invocations for less HBM bandwidth per invocation.
        """
        rows = jax.lax.dynamic_slice(An, (rb0, 0), (span, K))
        rows = rows.reshape(rbatch, _BLOCK, K)
        top = rows[:, :32, :].reshape(T, K)
        bot = rows[:, 32:, :].reshape(T, K)

        A2 = jnp.concatenate([top, bot], axis=0)
        A2_g = jnp.transpose(A2.reshape(2 * T, G, rank), (1, 0, 2))  # (G, 2T, rank)

        n_sub = ncbatch_actual * _BLOCK  # columns per sub-batch

        # Sub-batch scan: full k-dimension for a slice of columns.
        slot_idx = slot_indices

        def sub_batch_scan(Bn_sub):
            """Run full k-scan for one column sub-batch. Bn_sub: (K, n_sub)."""
            Bn_sub_g = Bn_sub.reshape(G, rank, n_sub)

            def step(carry, xs):
                acc, transcript = carry
                a_g, b_g, slot = xs
                c = jax.lax.dot_general(
                    a_g.astype(jnp.int8), b_g.astype(jnp.int8),
                    (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
                acc = acc + c
                xr = _xor_reduce_last64(acc, ncbatch_actual)
                combined = xr[:T] ^ xr[T:]
                old_slot = transcript[:, :, slot]
                new_slot = _rotl32(old_slot, 13) ^ combined
                transcript = transcript.at[:, :, slot].set(new_slot)
                return (acc, transcript), None

            acc0 = jnp.zeros((2 * T, n_sub), jnp.int32)
            tr0 = jnp.zeros((T, ncbatch_actual, 16), _U32)
            (_, transcript), _ = jax.lax.scan(step, (acc0, tr0), (A2_g, Bn_sub_g, slot_idx))
            return transcript  # (T, ncbatch_actual, 16)

        # Process each column sub-batch and concatenate transcripts.
        n_sub_batches = NC // ncbatch_actual
        transcripts = []
        for sb in range(n_sub_batches):
            col_start = sb * n_sub
            Bn_sub = jax.lax.dynamic_slice(Bn, (0, col_start), (K, n_sub))
            transcripts.append(sub_batch_scan(Bn_sub))

        return jnp.concatenate(transcripts, axis=1)  # (T, NC, 16)

    # Select the appropriate transcript function.
    use_ntiling = ncbatch_actual < NC

    @jax.jit
    def scan_batch(An, Bn, key_words, target_words, rb0):
        if use_ntiling:
            tr = _transcripts_ntiled(An, Bn, rb0)
        else:
            tr = _transcripts_fused(An, Bn, rb0)
        tr_flat = tr.reshape(T * NC, 16)
        digest = bj.compress_keyed_root(tr_flat, key_words)          # (T*NC, 8)
        idx = _lex_argmin(digest)
        meets = bj.below_or_equal_target(digest[idx][None, :], target_words)[0]
        c = idx // NC
        cb = idx % NC
        tile_m = rb0 + c2off_j[c]
        tile_n = cb2n[cb]
        return meets, tile_m.astype(jnp.int32), tile_n.astype(jnp.int32), tr_flat[idx]

    @jax.jit
    def debug_batch(An, Bn, rb0):
        if use_ntiling:
            return _transcripts_ntiled(An, Bn, rb0)
        return _transcripts_fused(An, Bn, rb0)

    return TiledScan(M, N, K, rank, rbatch, ncbatch_actual, rows_pattern, cols_pattern,
                     scan_batch, debug_batch)


def find_share_device(scan: TiledScan, An_j, Bn_j, key_words, target_words):
    """Host driver over already-on-device arrays: scan row-batches in order and
    return the first share found, with first-hit early-exit.

    Returns (tile_m, tile_n, transcript_bytes) or None.
    """
    span = scan.rbatch * _BLOCK
    for b in range(scan.num_batches):
        rb0 = b * span
        meets, tm, tn, tr = scan.scan_batch(An_j, Bn_j, key_words, target_words,
                                            jnp.int32(rb0))
        if bool(meets):
            tr_np = np.asarray(tr, dtype=np.uint32)
            return int(tm), int(tn), struct.pack("<16I", *[int(w) for w in tr_np])
    return None


def find_first_share(scan: TiledScan, An, Bn, key: bytes, target_le: bytes):
    """Convenience wrapper: accepts NumPy An/Bn + raw key/target bytes."""
    kw = bj.key_words_from_bytes(key)
    tw = jnp.asarray(np.frombuffer(target_le, dtype="<u4").astype(np.uint32))
    An_j = jnp.asarray(np.ascontiguousarray(An, dtype=np.int8))
    Bn_j = jnp.asarray(np.ascontiguousarray(Bn, dtype=np.int8))
    return find_share_device(scan, An_j, Bn_j, kw, tw)


def pick_rbatch(M: int, max_rbatch: int = 16) -> int:
    """Largest power-of-two rbatch (<= max_rbatch) such that rbatch*64 divides M."""
    nblocks = M // _BLOCK
    rb = 1
    while rb * 2 <= max_rbatch and nblocks % (rb * 2) == 0:
        rb *= 2
    return rb
