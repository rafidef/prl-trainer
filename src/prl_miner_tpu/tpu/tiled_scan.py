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
    rows_pattern: tuple
    cols_pattern: tuple
    scan_batch: object   # jitted f(An, Bn, key_words, target_words, rb0)
    debug_batch: object  # jitted f(An, Bn, rb0) -> transcripts (T, NC, 16)

    @property
    def num_batches(self) -> int:
        return self.M // (self.rbatch * _BLOCK)


def build_tiled_scan(M, N, K, rank, rows_pattern, cols_pattern, rbatch=8) -> TiledScan:
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

    # Static map: flat candidate row c -> tile_m offset within the batch.
    c2off = np.array([(c // 32) * _BLOCK + (c % 32) for c in range(T)], dtype=np.int32)
    c2off_j = jnp.asarray(c2off)
    cb2n = jnp.asarray(np.arange(NC, dtype=np.int32) * _BLOCK)

    def _transcripts(An, Bn, rb0):
        # Pull this batch's rows and split into top (i=0..31) / bottom (i=32..63).
        rows = jax.lax.dynamic_slice(An, (rb0, 0), (span, K))     # (span, K) int8
        rows = rows.reshape(rbatch, _BLOCK, K)
        top = rows[:, :32, :].reshape(T, K)                       # (T, K)
        bot = rows[:, 32:, :].reshape(T, K)                       # (T, K)

        # Stack top+bottom into ONE matmul of 2T rows per k-block: bigger MXU op,
        # half the launches vs two separate (T×rank)@(rank×N) matmuls.
        A2 = jnp.concatenate([top, bot], axis=0)                  # (2T, K)
        A2_g = jnp.transpose(A2.reshape(2 * T, G, rank), (1, 0, 2))   # (G, 2T, rank)
        Bn_g = Bn.reshape(G, rank, N)                                 # (G, rank, N)

        def step(acc, xs):
            a_g, b_g = xs                                            # (2T,rank),(rank,N)
            c = jax.lax.dot_general(
                a_g.astype(jnp.int8), b_g.astype(jnp.int8),
                (((1,), (0,)), ((), ())), preferred_element_type=jnp.int32)
            acc = acc + c                                            # (2T, N)
            xr = _xor_reduce_last64(acc, NC)                         # (2T, NC)
            combined = xr[:T] ^ xr[T:]                               # (T, NC)
            return acc, combined

        acc0 = jnp.zeros((2 * T, N), jnp.int32)
        _, combined_all = jax.lax.scan(step, acc0, (A2_g, Bn_g))  # (G, T, NC)

        # Fold each k-block's combined into the 16-word transcript (slot = g%16).
        slots = [jnp.zeros((T, NC), _U32) for _ in range(16)]
        for g in range(G):
            s = g % 16
            slots[s] = _rotl32(slots[s], 13) ^ combined_all[g]
        return jnp.stack(slots, axis=-1)                             # (T, NC, 16)

    @jax.jit
    def scan_batch(An, Bn, key_words, target_words, rb0):
        tr = _transcripts(An, Bn, rb0)                               # (T, NC, 16)
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
        return _transcripts(An, Bn, rb0)

    return TiledScan(M, N, K, rank, rbatch, rows_pattern, cols_pattern,
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
