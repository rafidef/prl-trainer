"""
Single-block keyed BLAKE3 in pure JAX.

Pearl's PoW check (`_khash` in the reference miner) hashes the 64-byte PoW
transcript with a *keyed* BLAKE3 and compares the 256-bit little-endian digest
against the share target. The transcript is exactly one BLAKE3 block (16 x
uint32 = 64 bytes), and it is simultaneously the first block, last block, and
root of a single one-chunk message. So the entire hash is a *single*
compression call — no chunk chaining, no tree. That makes it trivially
vectorizable: one compression over a whole batch of candidate tiles on the TPU
VPU, exactly mirroring the GPU kernel's `hash_keyed_block`.

This module implements only that single-block keyed/root compression, batched
over a leading dimension. It is validated bit-for-bit against the reference
`blake3` library in tests/test_blake3_jax.py.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

# BLAKE3 IV — identical to the SHA-256 / BLAKE2s initialization vector.
_IV = (
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
)

# Domain-separation flags (BLAKE3 spec).
CHUNK_START = 1 << 0
CHUNK_END = 1 << 1
ROOT = 1 << 3
KEYED_HASH = 1 << 4

# Flags for a one-block, one-chunk, keyed, root hash (Pearl's transcript hash).
_FLAGS_KEYED_ROOT_BLOCK = CHUNK_START | CHUNK_END | ROOT | KEYED_HASH  # = 27

# Message word permutation applied between rounds.
_MSG_PERMUTATION = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)

_U32 = jnp.uint32


def _rotr(x: jax.Array, n: int) -> jax.Array:
    """Rotate-right a uint32 array by n bits (n in 1..31)."""
    return (x >> _U32(n)) | (x << _U32(32 - n))


def _g(state: list[jax.Array], a: int, b: int, c: int, d: int,
       mx: jax.Array, my: jax.Array) -> None:
    """The BLAKE3 quarter-round G, mutating `state` in place (batched)."""
    state[a] = state[a] + state[b] + mx
    state[d] = _rotr(state[d] ^ state[a], 16)
    state[c] = state[c] + state[d]
    state[b] = _rotr(state[b] ^ state[c], 12)
    state[a] = state[a] + state[b] + my
    state[d] = _rotr(state[d] ^ state[a], 8)
    state[c] = state[c] + state[d]
    state[b] = _rotr(state[b] ^ state[c], 7)


def _round(state: list[jax.Array], m: list[jax.Array]) -> None:
    # Columns.
    _g(state, 0, 4, 8, 12, m[0], m[1])
    _g(state, 1, 5, 9, 13, m[2], m[3])
    _g(state, 2, 6, 10, 14, m[4], m[5])
    _g(state, 3, 7, 11, 15, m[6], m[7])
    # Diagonals.
    _g(state, 0, 5, 10, 15, m[8], m[9])
    _g(state, 1, 6, 11, 12, m[10], m[11])
    _g(state, 2, 7, 8, 13, m[12], m[13])
    _g(state, 3, 4, 9, 14, m[14], m[15])


def key_words_from_bytes(key: bytes) -> jax.Array:
    """Convert a 32-byte key into 8 little-endian uint32 words (shape (8,))."""
    if len(key) != 32:
        raise ValueError("BLAKE3 key must be exactly 32 bytes")
    import numpy as np
    return jnp.asarray(np.frombuffer(key, dtype="<u4").astype("uint32"))


def compress_block(messages: jax.Array, cv_words: jax.Array,
                   block_len: int, flags: int) -> jax.Array:
    """Single-block BLAKE3 compression over a batch of 16-word messages.

    Generic core: the chaining value `cv_words` is the key (keyed hash) or the IV
    (plain hash); `flags`/`block_len` are chosen by the caller. Counter t = 0
    (single chunk). Returns (..., 8) uint32 — the 32-byte digest as 8 LE words.
    """
    messages = messages.astype(_U32)
    batch = messages.shape[:-1]
    cv = cv_words.astype(_U32)

    state = [
        jnp.broadcast_to(cv[0], batch), jnp.broadcast_to(cv[1], batch),
        jnp.broadcast_to(cv[2], batch), jnp.broadcast_to(cv[3], batch),
        jnp.broadcast_to(cv[4], batch), jnp.broadcast_to(cv[5], batch),
        jnp.broadcast_to(cv[6], batch), jnp.broadcast_to(cv[7], batch),
        jnp.broadcast_to(_U32(_IV[0]), batch),
        jnp.broadcast_to(_U32(_IV[1]), batch),
        jnp.broadcast_to(_U32(_IV[2]), batch),
        jnp.broadcast_to(_U32(_IV[3]), batch),
        jnp.broadcast_to(_U32(0), batch),               # t_lo
        jnp.broadcast_to(_U32(0), batch),               # t_hi
        jnp.broadcast_to(_U32(block_len), batch),       # block_len
        jnp.broadcast_to(_U32(flags), batch),
    ]

    m = [messages[..., i] for i in range(16)]
    for r in range(7):
        _round(state, m)
        if r < 6:
            m = [m[_MSG_PERMUTATION[i]] for i in range(16)]

    # 32-byte output: out[i] = v[i] ^ v[i+8], for i in 0..7.
    out = [state[i] ^ state[i + 8] for i in range(8)]
    return jnp.stack(out, axis=-1)


_IV_WORDS = jnp.asarray([_U32(x) for x in _IV])
_FLAGS_PLAIN_ROOT_BLOCK = CHUNK_START | CHUNK_END | ROOT  # non-keyed single block


def compress_keyed_root(messages: jax.Array, key_words: jax.Array) -> jax.Array:
    """Keyed/root single-block BLAKE3 (Pearl's PoW transcript hash).

    Args:
        messages:  uint32 (..., 16) — each row one 64-byte block (16 LE words).
        key_words: uint32 (8,) — the key as 8 LE words.
    Returns:
        uint32 (..., 8) — digest as 8 LE words (word 0 least-significant).
    """
    return compress_block(messages, key_words, 64, _FLAGS_KEYED_ROOT_BLOCK)


def below_or_equal_target(digest_words: jax.Array, target_words: jax.Array) -> jax.Array:
    """Compare 256-bit little-endian values: digest <= target, elementwise.

    Args:
        digest_words: uint32 (..., 8), LE word order (word 0 = least significant).
        target_words: uint32 (8,), LE word order.

    Returns:
        bool array of shape (...,) — True where digest <= target.
    """
    digest_words = digest_words.astype(_U32)
    target_words = target_words.astype(_U32)
    # Compare from the most-significant word (index 7) down to the least.
    less = jnp.zeros(digest_words.shape[:-1], dtype=bool)
    equal = jnp.ones(digest_words.shape[:-1], dtype=bool)
    for i in range(7, -1, -1):
        d = digest_words[..., i]
        t = target_words[i]
        less = less | (equal & (d < t))
        equal = equal & (d == t)
    return less | equal
