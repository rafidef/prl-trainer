"""
NumPy reference oracle for Pearl NoisyGEMM mining.

This is the *authoritative* bit-exact specification of the work the miner does,
extracted from the validated `prl_miner.selftest` reference (which is itself
checked against the live AlphaPool-accepted CUDA kernel). The JAX/TPU core is
validated against this module, so it is deliberately simple and literal — clarity
over speed.

Profile: the live AlphaPool profile is rows_pattern = [0, 32],
cols_pattern = [0..63], rank = 128, but the functions below take the patterns as
arguments so they stay general.
"""
from __future__ import annotations

import struct

import numpy as np
import blake3 as _blake3


def rotl32(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def clamp_i8(a: np.ndarray) -> np.ndarray:
    return np.clip(a, -128, 127).astype(np.int8)


def apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b):
    """Apply Pearl low-rank ±1 noise to A and B (returns int8 An, Bn).

    A_noised[i,j] = clamp(A[i,j] + E_AL[i, r0a[j]] - E_AL[i, r1a[j]])
    B_noised[k,j] = clamp(B[k,j] + E_BR[j, r0b[k]] - E_BR[j, r1b[k]])
    """
    A = A.astype(np.int32)
    An = clamp_i8(A + E_AL[:, r0a].astype(np.int32) - E_AL[:, r1a].astype(np.int32))
    B = B.astype(np.int32)
    EBR0 = E_BR[:, r0b].astype(np.int32).T  # (k, n)
    EBR1 = E_BR[:, r1b].astype(np.int32).T
    Bn = clamp_i8(B + EBR0 - EBR1)
    return An, Bn


def ref_transcript(An, Bn, tm, tn, K, rank, rows_pattern, cols_pattern):
    """64-byte PoW transcript (16 uint32 words) for the hash tile at (tm, tn)."""
    rows = [tm + off for off in rows_pattern]
    cols = [tn + off for off in cols_pattern]
    nrows, ncols = len(rows), len(cols)
    tr = [0] * 16
    Cacc = np.zeros((nrows, ncols), dtype=np.int64)
    for rc, kb in enumerate(range(0, K, rank)):
        Cacc += (An[np.ix_(rows, range(kb, kb + rank))].astype(np.int64)
                 @ Bn[np.ix_(range(kb, kb + rank), cols)].astype(np.int64))
        combined = 0
        for v in Cacc.astype(np.uint32).flatten():
            combined ^= int(v)
        tr[rc % 16] = rotl32(tr[rc % 16], 13) ^ combined
    return tr


def khash(tr, key: bytes) -> int:
    """Keyed BLAKE3 of the transcript -> 256-bit little-endian integer."""
    tb = b"".join(struct.pack("<I", w & 0xFFFFFFFF) for w in tr)
    return int.from_bytes(_blake3.blake3(tb, key=key).digest(), "little")


def candidate_tiles(M, N, rows_pattern, cols_pattern):
    """Enumerate (tile_m, tile_n) candidates exactly as the kernel scans them.

    A 64-row block holds the candidates whose two patterned rows (tile_m+0 and
    tile_m+max_row_off) both land inside that same block, i.e. i in
    range(64 - max_row_off) — for the live [0,32] pattern that is i in 0..31, the
    top half (matching selftest's `range(32)` and its `tm % 64 < 32` assertion).
    Columns step by len(cols_pattern); every patterned row/col must stay in bounds.
    """
    block = 64
    max_row_off = max(rows_pattern)
    max_col_off = max(cols_pattern)
    ncols = len(cols_pattern)
    for rb in range(0, M, block):
        for i in range(block - max_row_off):
            tm = rb + i
            if tm + max_row_off >= M:
                continue
            for cb in range(0, N, ncols):
                if cb + max_col_off >= N:
                    continue
                yield tm, cb


def full_scan(An, Bn, K, rank, key, target_int, rows_pattern, cols_pattern):
    """Reference full scan. Returns (tile_m, tile_n, transcript, hash_int) of the
    lowest-hash candidate, plus whether it meets the target."""
    M, _ = An.shape
    _, N = Bn.shape
    best = None
    for tm, tn in candidate_tiles(M, N, rows_pattern, cols_pattern):
        tr = ref_transcript(An, Bn, tm, tn, K, rank, rows_pattern, cols_pattern)
        h = khash(tr, key)
        if best is None or h < best[3]:
            best = (tm, tn, tr, h)
    found = best is not None and best[3] <= target_int
    return best, found
