"""Validate the TPU pearl.challenge solver against the blake3 library + CPU scan."""
import os
import sys

import numpy as np
import blake3 as _blake3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prl_miner_tpu.tpu import challenge as ch    # noqa: E402


def _clz_bits(data: bytes) -> int:
    count = 0
    for b in data:
        if b == 0:
            count += 8
        else:
            mask = 0x80
            while mask and not (b & mask):
                count += 1
                mask >>= 1
            break
    return count


def _cpu_first_nonce(seed: bytes, difficulty: int, limit: int) -> int | None:
    for nonce in range(limit):
        h = _blake3.blake3(seed + nonce.to_bytes(8, "little")).digest()
        if _clz_bits(h) >= difficulty:
            return nonce
    return None


def test_solver_finds_valid_lowest_nonce():
    difficulty = 16
    for s in range(3):
        seed = _blake3.blake3(b"seed" + bytes([s])).digest()
        nonce = ch.solve_challenge_tpu(seed, difficulty, batch=1 << 20, max_nonce=1 << 22)
        assert nonce is not None, f"seed {s}: solver found no nonce"
        # The found nonce really clears the bar.
        h = _blake3.blake3(seed + nonce.to_bytes(8, "little")).digest()
        assert _clz_bits(h) >= difficulty, f"seed {s}: nonce {nonce} does not meet difficulty"
        # And it is the lowest such nonce.
        cpu = _cpu_first_nonce(seed, difficulty, nonce + 1)
        assert cpu == nonce, f"seed {s}: solver nonce {nonce} != cpu-first {cpu}"


def test_meets_difficulty_d32_is_word0_zero():
    import jax.numpy as jnp
    # Construct digests with word0 == 0 (should pass d=32) and != 0 (should fail).
    digests = jnp.asarray(np.array([[0, 1, 2, 3, 4, 5, 6, 7],
                                    [1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint32))
    got = np.asarray(ch._meets_difficulty(digests, 32))
    assert got.tolist() == [True, False]


if __name__ == "__main__":
    test_meets_difficulty_d32_is_word0_zero()
    print("meets_difficulty d=32: OK")
    test_solver_finds_valid_lowest_nonce()
    print("TPU challenge solver finds valid lowest nonce: OK")
