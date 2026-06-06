"""Validate the JAX single-block keyed BLAKE3 against the reference blake3 lib."""
import os
import struct
import sys

import numpy as np
import blake3 as _blake3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prl_miner_tpu.tpu import blake3_jax as bj  # noqa: E402


def _ref_khash_words(transcript_words, key: bytes):
    """Reference: exactly what selftest._khash does, returning the 8 LE words."""
    tb = b"".join(struct.pack("<I", int(w) & 0xFFFFFFFF) for w in transcript_words)
    digest = _blake3.blake3(tb, key=key).digest()
    return np.frombuffer(digest, dtype="<u4").astype(np.uint32), int.from_bytes(digest, "little")


def test_keyed_single_block_matches_reference():
    rng = np.random.default_rng(0)
    n = 257  # batch, deliberately not a round number
    keys = [rng.integers(0, 256, 32, dtype=np.uint8).tobytes() for _ in range(3)]
    transcripts = rng.integers(0, 2**32, size=(n, 16), dtype=np.uint64).astype(np.uint32)

    for key in keys:
        kw = bj.key_words_from_bytes(key)
        digest_words = np.asarray(bj.compress_keyed_root(transcripts, kw))
        assert digest_words.shape == (n, 16 // 2)  # (n, 8)
        for i in range(n):
            ref_words, _ = _ref_khash_words(transcripts[i], key)
            assert np.array_equal(digest_words[i], ref_words), (
                f"mismatch row {i}: jax={digest_words[i]} ref={ref_words}")


def test_below_target_matches_integer_compare():
    rng = np.random.default_rng(1)
    n = 64
    key = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    kw = bj.key_words_from_bytes(key)
    transcripts = rng.integers(0, 2**32, size=(n, 16), dtype=np.uint64).astype(np.uint32)
    digest_words = bj.compress_keyed_root(transcripts, kw)

    ints = []
    for i in range(n):
        _, hi = _ref_khash_words(transcripts[i], key)
        ints.append(hi)

    # Pick a target near the median so both branches are exercised.
    target_int = int(np.median(ints))
    target_words = np.frombuffer(target_int.to_bytes(32, "little"), dtype="<u4").astype(np.uint32)
    got = np.asarray(bj.below_or_equal_target(digest_words, target_words))
    want = np.array([hi <= target_int for hi in ints])
    assert np.array_equal(got, want)


if __name__ == "__main__":
    test_keyed_single_block_matches_reference()
    test_below_target_matches_integer_compare()
    print("blake3_jax: ALL OK")
