"""
JAX NoisyGEMM PoW search — the TPU replacement for the CUDA mining kernel.

Given the *noised* matrices An (m x k int8) and Bn (k x n int8), this scans every
candidate hash tile, builds its 64-byte PoW transcript exactly as the reference
does, keyed-BLAKE3-hashes it, and reports the lowest-hash tile (and whether it
meets the share target).

The hot path is entirely integer:
  * int8 x int8 -> int32 GEMM per rank-sized k-block (the TPU MXU op),
  * cumulative int32 accumulation across k-blocks,
  * XOR-fold of the running accumulator into a 16-word transcript (VPU),
  * single-block keyed BLAKE3 + 256-bit target compare (VPU).

No floating point anywhere, so results are bit-identical to the CUDA kernel /
NumPy reference. This module favors a clear, faithful formulation; the
TPU-throughput tiling/pipelining lives in a later optimization pass.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import blake3_jax as bj

_U32 = jnp.uint32


def _rotl32(x: jax.Array, n: int) -> jax.Array:
    return (x << _U32(n)) | (x >> _U32(32 - n))


def candidate_starts(M: int, N: int, rows_pattern, cols_pattern):
    """Static lists of candidate (tile_m) row starts and (tile_n) col starts.

    Matches reference.candidate_tiles: 64-row blocks, len(cols_pattern)-wide col
    steps, every patterned row/col must stay in bounds.
    """
    block = 64
    max_row_off = max(rows_pattern)
    max_col_off = max(cols_pattern)
    ncols = len(cols_pattern)
    # Only the top (block - max_row_off) rows of each 64-block are tile_m starts,
    # so both patterned rows (tile_m, tile_m+max_row_off) stay in the same block.
    tms = [rb + i for rb in range(0, M, block) for i in range(block - max_row_off)
           if rb + i + max_row_off < M]
    tns = [cb for cb in range(0, N, ncols) if cb + max_col_off < N]
    return tms, tns


def _lex_argmin(digest_flat: jax.Array) -> jax.Array:
    """Index of the minimum 256-bit little-endian value in a (T, 8) uint32 array.

    Compares most-significant word (index 7) first; ties broken toward the lowest
    flat index. Pure-device, O(8) passes.
    """
    T = digest_flat.shape[0]
    alive = jnp.ones((T,), dtype=bool)
    BIG = _U32(0xFFFFFFFF)
    for i in range(7, -1, -1):
        w = digest_flat[:, i]
        masked = jnp.where(alive, w, BIG)
        mn = jnp.min(masked)
        alive = alive & (w == mn)
    # Lowest flat index among the survivors.
    idx = jnp.argmax(alive.astype(jnp.int32) * (jnp.arange(T, 0, -1)))
    return idx


def build_scan(M: int, N: int, K: int, rank: int, rows_pattern, cols_pattern):
    """Build a jitted scan closure specialized to the given (static) shape/profile.

    Returns f(An, Bn, key_words, target_words) ->
        (found: bool, tile_m: int32, tile_n: int32, transcript: uint32[16]).
    """
    rows_pattern = tuple(int(x) for x in rows_pattern)
    cols_pattern = tuple(int(x) for x in cols_pattern)
    tms, tns = candidate_starts(M, N, rows_pattern, cols_pattern)
    G = K // rank
    P = len(rows_pattern)
    C = len(cols_pattern)

    tms_arr = jnp.asarray(np.array(tms, dtype=np.int32))
    tns_arr = jnp.asarray(np.array(tns, dtype=np.int32))
    # Absolute row indices [T_m, P] and col indices [T_n, C].
    row_idx = (np.array(tms, dtype=np.int32)[:, None]
               + np.array(rows_pattern, dtype=np.int32)[None, :])      # (T_m, P)
    col_idx = (np.array(tns, dtype=np.int32)[:, None]
               + np.array(cols_pattern, dtype=np.int32)[None, :])      # (T_n, C)
    row_idx_j = jnp.asarray(row_idx)
    col_idx_j = jnp.asarray(col_idx)
    T_m = len(tms)
    T_n = len(tns)

    @jax.jit
    def f(An, Bn, key_words, target_words):
        An = An.astype(jnp.int8)
        Bn = Bn.astype(jnp.int8)

        # Gather patterned rows of A and cols of B.
        Arows = An[row_idx_j, :]                         # (T_m, P, K) int8
        Bcols = Bn[:, col_idx_j]                         # (K, T_n, C) int8
        Bcols = jnp.transpose(Bcols, (1, 0, 2))          # (T_n, K, C)

        # Split K into G rank-sized blocks.
        Arows = Arows.reshape(T_m, P, G, rank)           # (T_m, P, G, rank)
        Bcols = Bcols.reshape(T_n, G, rank, C)           # (T_n, G, rank, C)

        # Per-(tm,tn,kblock) int32 block product: sum_r Arows * Bcols.
        # einsum 'mpgr,ngrc->mngpc'. int8 inputs, int32 accumulation (MXU op on TPU).
        block_prod = jnp.einsum(
            "mpgr,ngrc->mngpc",
            Arows.astype(jnp.int32), Bcols.astype(jnp.int32),
            preferred_element_type=jnp.int32,
        )                                                # (T_m, T_n, G, P, C) int32

        # Cumulative accumulation across k-blocks.
        Ccum = jnp.cumsum(block_prod, axis=2)            # (T_m, T_n, G, P, C) int32

        # XOR-fold each cumulative block over (P, C) -> combined[ ., ., G] uint32.
        Ccum_u = jax.lax.bitcast_convert_type(Ccum, jnp.uint32)
        combined = jax.lax.reduce(
            Ccum_u, _U32(0), jax.lax.bitwise_xor, dimensions=(3, 4)
        )                                                # (T_m, T_n, G) uint32

        # Fold into the 16-word transcript with rotate-left-13 mixing.
        tr = jnp.zeros((T_m, T_n, 16), dtype=jnp.uint32)
        for g in range(G):
            slot = g % 16
            mixed = _rotl32(tr[:, :, slot], 13) ^ combined[:, :, g]
            tr = tr.at[:, :, slot].set(mixed)

        # Keyed BLAKE3 + target compare.
        tr_flat = tr.reshape(T_m * T_n, 16)
        digest = bj.compress_keyed_root(tr_flat, key_words)   # (T, 8)
        idx = _lex_argmin(digest)
        meets = bj.below_or_equal_target(digest[idx][None, :], target_words)[0]

        tm = tms_arr[idx // T_n]
        tn = tns_arr[idx % T_n]
        winner_tr = tr_flat[idx]
        return meets, tm.astype(jnp.int32), tn.astype(jnp.int32), winner_tr

    f._meta = dict(T_m=T_m, T_n=T_n, G=G, P=P, C=C, tms=tms, tns=tns)
    return f


def transcript_for_tile(An, Bn, tm, tn, K, rank, rows_pattern, cols_pattern):
    """JAX transcript for a single tile (for direct bit-exact comparison)."""
    rows_pattern = tuple(int(x) for x in rows_pattern)
    cols_pattern = tuple(int(x) for x in cols_pattern)
    G = K // rank
    rows = np.array([tm + o for o in rows_pattern], dtype=np.int32)
    cols = np.array([tn + o for o in cols_pattern], dtype=np.int32)
    Arows = jnp.asarray(An)[jnp.asarray(rows), :].reshape(len(rows), G, rank)
    Bcols = jnp.asarray(Bn)[:, jnp.asarray(cols)].reshape(G, rank, len(cols))
    bp = jnp.einsum("pgr,grc->gpc", Arows.astype(jnp.int32), Bcols.astype(jnp.int32),
                    preferred_element_type=jnp.int32)
    Ccum = jnp.cumsum(bp, axis=0)
    Ccum_u = jax.lax.bitcast_convert_type(Ccum, jnp.uint32)
    combined = jax.lax.reduce(Ccum_u, _U32(0), jax.lax.bitwise_xor, dimensions=(1, 2))
    tr = jnp.zeros((16,), dtype=jnp.uint32)
    for g in range(G):
        slot = g % 16
        tr = tr.at[slot].set(_rotl32(tr[slot], 13) ^ combined[g])
    return np.asarray(tr)
