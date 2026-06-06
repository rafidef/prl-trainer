"""
Dense Pearl noise generation in JAX (runs on the TPU).

Each 32-byte block of dense noise is a single keyed BLAKE3 of a 64-byte message
whose first word is the chunk index (+1) and whose last 8 words are the fixed
seed ("A_tensor"/"B_tensor"). That is exactly the single-block keyed hash in
blake3_jax, so we generate all chunks as one batched compression — no host
BLAKE3 loop (the GIL-heavy path the CUDA miner moved on-device).

Bit-identical to prl_miner_tpu.noise.generate_dense (validated in tests).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import blake3_jax as bj

_U32 = jnp.uint32
VALS_PER_HASH = 32
NOISE_RANGE = 64
ZERO_POINT = 32


def generate_dense_jax(key: bytes, seed_fixed: bytes, rows: int, rank: int) -> jax.Array:
    """Dense noise (rows, rank) int8 in [-32, 32), generated on-device."""
    if len(seed_fixed) != 32:
        raise ValueError("seed_fixed must be 32 bytes")
    total = rows * rank
    num_chunks = (total + VALS_PER_HASH - 1) // VALS_PER_HASH

    key_words = bj.key_words_from_bytes(key)
    seed_words = jnp.asarray(np.frombuffer(seed_fixed, dtype="<u4").astype("uint32"))  # (8,)

    # Build (num_chunks, 16) messages: word0 = chunk_idx+1, words 8..15 = seed.
    chunk_idx = jnp.arange(1, num_chunks + 1, dtype=_U32)              # (num_chunks,)
    msgs = jnp.zeros((num_chunks, 16), dtype=_U32)
    msgs = msgs.at[:, 0].set(chunk_idx)
    msgs = msgs.at[:, 8:16].set(jnp.broadcast_to(seed_words, (num_chunks, 8)))

    digest = bj.compress_keyed_root(msgs, key_words)                  # (num_chunks, 8) uint32

    # Expand each LE word into its 4 little-endian bytes.
    b0 = digest & _U32(0xFF)
    b1 = (digest >> _U32(8)) & _U32(0xFF)
    b2 = (digest >> _U32(16)) & _U32(0xFF)
    b3 = (digest >> _U32(24)) & _U32(0xFF)
    byts = jnp.stack([b0, b1, b2, b3], axis=-1)                       # (num_chunks, 8, 4)
    flat = byts.reshape(-1)[:total]                                   # (total,) uint32 in 0..255

    vals = (flat % _U32(NOISE_RANGE)).astype(jnp.int32) - ZERO_POINT
    return vals.astype(jnp.int8).reshape(rows, rank)
