"""
AlphaPool `pearl.challenge` solver on the TPU.

The pool opens each connection with a proof-of-work challenge: find a uint64
nonce such that BLAKE3(seed_32 || nonce_le8) has >= `difficulty` leading zero
bits (difficulty is 32 in practice). That is ~4 billion hashes — far too slow in
pure-Python BLAKE3, so the CUDA miner solves it on the GPU. Here we do the same
on the TPU: vectorize the (single-block, non-keyed) BLAKE3 over a batch of nonces
and scan batches until one clears the bar.

The hashed message is seed(32B = 8 words) || nonce(8B = 2 words) || zero-pad, a
40-byte single block, hashed with the plain (IV) chaining value.
"""
from __future__ import annotations

import time
import logging

import jax
import jax.numpy as jnp
import numpy as np

from . import blake3_jax as bj

log = logging.getLogger(__name__)
_U32 = jnp.uint32


def _meets_difficulty(digest_words: jax.Array, d: int) -> jax.Array:
    """Big-endian leading-zero-bits(digest) >= d, elementwise.

    Digest byte order is word0's 4 LE bytes first (digest[0] = word0 & 0xff),
    and `_count_leading_zero_bits` reads digest[0] as most-significant. So the
    first whole zero *byte* is (word[p//4] >> 8*(p%4)) & 0xff for p = 0,1,...
    """
    nbytes = d // 8
    rem = d % 8
    ok = jnp.ones(digest_words.shape[:-1], dtype=bool)
    for p in range(nbytes):
        byte = (digest_words[..., p // 4] >> _U32(8 * (p % 4))) & _U32(0xFF)
        ok = ok & (byte == _U32(0))
    if rem:
        byte = (digest_words[..., nbytes // 4] >> _U32(8 * (nbytes % 4))) & _U32(0xFF)
        ok = ok & ((byte >> _U32(8 - rem)) == _U32(0))
    return ok


def _build_kernel(difficulty: int):
    @jax.jit
    def kernel(seed_words, nonce_lo, nonce_hi):
        # Full uint64 nonce split into two uint32 halves so the TPU stays in
        # 32-bit math. Message = seed(8 words) || nonce_lo || nonce_hi || zeros,
        # i.e. seed_32 || nonce.to_bytes(8, "little"), a 40-byte single block.
        B = nonce_lo.shape[0]
        msg = jnp.zeros((B, 16), dtype=_U32)
        msg = msg.at[:, 0:8].set(jnp.broadcast_to(seed_words, (B, 8)))
        msg = msg.at[:, 8].set(nonce_lo)          # nonce bits 0..31
        msg = msg.at[:, 9].set(nonce_hi)          # nonce bits 32..63
        digest = bj.compress_block(msg, bj._IV_WORDS, 40, bj._FLAGS_PLAIN_ROOT_BLOCK)
        hits = _meets_difficulty(digest, difficulty)
        any_hit = jnp.any(hits)
        # Lowest-index (lowest nonce) among hits.
        idx = jnp.argmax(hits.astype(jnp.int32) * jnp.arange(B, 0, -1))
        return any_hit, nonce_lo[idx], nonce_hi[idx]
    return kernel


def solve_challenge_tpu(seed: bytes, difficulty: int,
                        batch: int = 1 << 23, max_nonce: int = 1 << 36):
    """Return the winning uint64 nonce (int) or None if none found below max_nonce.

    Searches the full 64-bit nonce space (the pool's nonce is uint64); ~37% of
    difficulty-32 seeds have no solution below 2^32, so a 32-bit-only search would
    spuriously fail on them.
    """
    assert len(seed) == 32, "challenge seed must be 32 bytes"
    seed_words = jnp.asarray(np.frombuffer(seed, dtype="<u4").astype(np.uint32))
    kernel = _build_kernel(difficulty)
    t0 = time.monotonic()
    base = 0
    while base < max_nonce:
        n = min(batch, max_nonce - base)
        nonces = np.arange(base, base + n, dtype=np.uint64)
        nonce_lo = jnp.asarray((nonces & np.uint64(0xFFFFFFFF)).astype(np.uint32))
        nonce_hi = jnp.asarray((nonces >> np.uint64(32)).astype(np.uint32))
        any_hit, w_lo, w_hi = kernel(seed_words, nonce_lo, nonce_hi)
        if bool(any_hit):
            won = (int(w_hi) << 32) | int(w_lo)
            dt = time.monotonic() - t0
            log.info("Challenge solved: difficulty=%d nonce=%d in %.2fs (~%.0f Mh/s)",
                     difficulty, won, dt, (base + n) / dt / 1e6 if dt > 0 else 0)
            return won
        base += n
    return None
