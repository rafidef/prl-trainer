"""JAX dense-noise generator must be bit-identical to the NumPy reference."""
import os
import sys

import numpy as np
import blake3 as _blake3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prl_miner_tpu import noise as npnoise              # noqa: E402
from prl_miner_tpu.tpu import noise_jax as jnoise       # noqa: E402


def test_generate_dense_matches_numpy():
    for seed_fixed in (npnoise.SEED_A, npnoise.SEED_B):
        for rows in (128, 257, 1024):
            for rank in (128,):
                for s in range(3):
                    key = _blake3.blake3(bytes([s]) + seed_fixed[:4]).digest()
                    cpu = npnoise.generate_dense(key, seed_fixed, rows, rank)
                    gpu = np.asarray(jnoise.generate_dense_jax(key, seed_fixed, rows, rank))
                    assert gpu.shape == cpu.shape == (rows, rank)
                    assert gpu.dtype == np.int8
                    assert np.array_equal(gpu, cpu), \
                        f"mismatch seed_fixed={seed_fixed[:4]!r} rows={rows} s={s}"


if __name__ == "__main__":
    test_generate_dense_matches_numpy()
    print("noise_jax: dense noise bit-identical to NumPy reference — OK")
