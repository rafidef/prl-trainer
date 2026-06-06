"""
Pearl NoisyGEMM noise generator — pure Python/NumPy.

Ported verbatim (same semantics) from prl-miner-turing/src/prl_miner/noise.py so
the TPU package is self-contained. Produces, per job:
  E_AL : (m, rank) int8  dense noise for A   (key = commitment_A, seed "A_tensor")
  E_BR : (n, rank) int8  dense noise for B   (key = commitment_B, seed "B_tensor")
  r0a, r1a : (k,) int32   sparse +1/-1 column indices for A
  r0b, r1b : (k,) int32   sparse +1/-1 column indices for B

Applied as:
  A_noised[i,j] = clamp(A[i,j] + E_AL[i, r0a[j]] - E_AL[i, r1a[j]])
  B_noised[k,j] = clamp(B[k,j] + E_BR[j, r0b[k]] - E_BR[j, r1b[k]])
"""
from __future__ import annotations

import struct

import numpy as np
import blake3 as _blake3

SEED_A = b"A_tensor" + b"\x00" * 24
SEED_B = b"B_tensor" + b"\x00" * 24

VALS_PER_HASH = 32
NOISE_RANGE = 64
ZERO_POINT = 32


def _get_random_hash(chunk_idx: int, seed_fixed: bytes, key: bytes, is_sparse: bool) -> bytes:
    data = bytearray(64)
    if not is_sparse:
        struct.pack_into("<I", data, 0, chunk_idx + 1)
    else:
        struct.pack_into("<I", data, 4, chunk_idx + 1)
    data[32:64] = seed_fixed
    return _blake3.blake3(bytes(data), key=key).digest()


def _mulhi32(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) >> 32


def generate_dense(key: bytes, seed_fixed: bytes, rows: int, rank: int) -> np.ndarray:
    """Dense noise (rows, rank) int8, values in [-32, 32)."""
    total = rows * rank
    num_chunks = (total + VALS_PER_HASH - 1) // VALS_PER_HASH
    raw = bytearray(num_chunks * VALS_PER_HASH)
    for i in range(num_chunks):
        raw[i * VALS_PER_HASH:(i + 1) * VALS_PER_HASH] = _get_random_hash(i, seed_fixed, key, False)
    arr = np.frombuffer(bytes(raw[:total]), dtype=np.uint8)
    values = ((arr.astype(np.int16) % NOISE_RANGE) - ZERO_POINT).astype(np.int8)
    return np.ascontiguousarray(values.reshape(rows, rank))


def generate_sparse_indices(key: bytes, seed_fixed: bytes, k_len: int, rank: int):
    """Return (r0, r1) int32 arrays of length k_len: the +1/-1 column indices."""
    HASHES_PER_CHUNK = 8
    num_chunks = (k_len + HASHES_PER_CHUNK - 1) // HASHES_PER_CHUNK
    r0 = np.zeros(k_len, dtype=np.int32)
    r1 = np.zeros(k_len, dtype=np.int32)
    rank_mask = rank - 1
    for chunk_idx in range(num_chunks):
        h = _get_random_hash(chunk_idx, seed_fixed, key, True)
        uints = struct.unpack_from("<8I", h)
        for sub in range(8):
            k_pos = chunk_idx * HASHES_PER_CHUNK + sub
            if k_pos >= k_len:
                break
            u = uints[sub]
            a = u & rank_mask
            b = (a ^ (1 + _mulhi32(rank - 1, u))) & 0xFF
            r0[k_pos] = a
            r1[k_pos] = b
    return r0, r1


class NoiseGenerator:
    def __init__(self, rank: int = 128):
        assert rank > 0 and (rank & (rank - 1)) == 0, "rank must be power of 2"
        self.rank = rank

    def generate(self, key_A: bytes, key_B: bytes, m: int, k: int, n: int):
        r = self.rank
        E_AL = generate_dense(key_A, SEED_A, m, r)
        r0a, r1a = generate_sparse_indices(key_A, SEED_A, k, r)
        E_BR = generate_dense(key_B, SEED_B, n, r)
        r0b, r1b = generate_sparse_indices(key_B, SEED_B, k, r)
        return E_AL, r0a, r1a, E_BR, r0b, r1b

    def generate_sparse(self, key_A: bytes, key_B: bytes, k: int):
        r = self.rank
        r0a, r1a = generate_sparse_indices(key_A, SEED_A, k, r)
        r0b, r1b = generate_sparse_indices(key_B, SEED_B, k, r)
        return r0a, r1a, r0b, r1b
