"""
AlphaPool `pearl.challenge` solver on the TPU.

The pool opens each connection with a proof-of-work challenge: find a uint64
nonce such that BLAKE3(seed_32 || nonce_le8) has >= `difficulty` leading zero
bits (difficulty is 32 in practice). That is ~4 billion hashes — far too slow in
pure-Python BLAKE3, so the CUDA miner solves it on the GPU. Here we do the same
on the TPU: vectorize the (single-block, non-keyed) BLAKE3 over a batch of nonces
and scan batches until one clears the bar.

Multi-chip: when multiple local chips are available, the nonce space is split
across them for ~N× faster solving (N = number of local chips).

The hashed message is seed(32B = 8 words) || nonce(8B = 2 words) || zero-pad, a
40-byte single block, hashed with the plain (IV) chaining value.
"""
from __future__ import annotations

import time
import logging
import concurrent.futures

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


def _solve_on_device(device, seed_words_np, difficulty, batch, start, end):
    """Solve challenge on a single device starting from nonce `start` up to `end`."""
    seed_words = jax.device_put(jnp.asarray(seed_words_np), device)
    kernel = _build_kernel(difficulty)
    base = start
    while base < end:
        n = min(batch, end - base)
        nonces = np.arange(base, base + n, dtype=np.uint64)
        nonce_lo = jax.device_put(
            jnp.asarray((nonces & np.uint64(0xFFFFFFFF)).astype(np.uint32)), device)
        nonce_hi = jax.device_put(
            jnp.asarray((nonces >> np.uint64(32)).astype(np.uint32)), device)
        any_hit, w_lo, w_hi = kernel(seed_words, nonce_lo, nonce_hi)
        if bool(any_hit):
            return (int(w_hi) << 32) | int(w_lo), base + n
        base += n
    return None, end


def solve_challenge_tpu(seed: bytes, difficulty: int,
                        batch: int = 1 << 23, max_nonce: int = 1 << 36):
    """Return the winning uint64 nonce (int) or None if none found below max_nonce.

    Searches the full 64-bit nonce space (the pool's nonce is uint64); ~37% of
    difficulty-32 seeds have no solution below 2^32, so a 32-bit-only search would
    spuriously fail on them.

    Multi-chip: splits the nonce space across all local TPU chips for parallel
    solving. Each chip searches a disjoint region.
    """
    assert len(seed) == 32, "challenge seed must be 32 bytes"
    seed_words_np = np.frombuffer(seed, dtype="<u4").astype(np.uint32)

    local_devs = jax.local_devices()
    num_devices = len(local_devs)
    t0 = time.monotonic()

    if num_devices <= 1:
        # Single device: original serial path.
        won, searched = _solve_on_device(
            local_devs[0], seed_words_np, difficulty, batch, 0, max_nonce)
        if won is not None:
            dt = time.monotonic() - t0
            log.info("Challenge solved: difficulty=%d nonce=%d in %.2fs (~%.0f Mh/s)",
                     difficulty, won, dt, searched / dt / 1e6 if dt > 0 else 0)
        return won

    # Multi-chip: split nonce space evenly across devices.
    chunk = max_nonce // num_devices
    log.info("Challenge: splitting %d nonces across %d chips (%d each)",
             max_nonce, num_devices, chunk)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_devices) as pool:
        futures = {}
        for i, dev in enumerate(local_devs):
            start = i * chunk
            end = max_nonce if i == num_devices - 1 else (i + 1) * chunk
            futures[pool.submit(_solve_on_device, dev, seed_words_np,
                                difficulty, batch, start, end)] = i

        for fut in concurrent.futures.as_completed(futures):
            won, searched = fut.result()
            if won is not None:
                dt = time.monotonic() - t0
                log.info("Challenge solved: difficulty=%d nonce=%d on chip %d "
                         "in %.2fs (~%.0f Mh/s effective)",
                         difficulty, won, futures[fut], dt,
                         searched / dt / 1e6 if dt > 0 else 0)
                # Cancel remaining futures (best-effort).
                for f in futures:
                    f.cancel()
                return won

    return None
