"""
Multi-device pinning tests.

Run with simulated multiple devices on CPU:
    XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    JAX_PLATFORMS=cpu python -m pytest -q tests/test_multi_device.py

Validates:
  1. TpuMiner(device_id=N) places arrays on the correct device.
  2. Independent scans on different devices produce correct (bit-exact) results.
  3. Device pinning does not break the tiled scan.
"""
import os
import struct

import numpy as np
import pytest
import blake3 as _blake3

# Set XLA flags BEFORE importing JAX if not already set.
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prl_miner_tpu import reference as ref
from prl_miner_tpu import noise as npnoise
from prl_miner_tpu.tpu.miner import TpuMiner

ROWS = (0, 32)
COLS = tuple(range(64))


@pytest.fixture
def num_devices():
    n = len(jax.local_devices())
    if n < 2:
        pytest.skip("Need at least 2 simulated devices; "
                    "run with XLA_FLAGS='--xla_force_host_platform_device_count=4'")
    return n


def _reference_winner(An, Bn, K, rank, key):
    """Find the winning tile via the NumPy reference."""
    best, _ = ref.full_scan(An, Bn, K, rank, key, target_int=-1,
                            rows_pattern=ROWS, cols_pattern=COLS)
    tm, tn, tr, hmin = best
    return tm, tn, tr, hmin.to_bytes(32, "little")


def test_device_pinning_basic(num_devices):
    """TpuMiner instances pin matrices to different devices."""
    miners = [TpuMiner(i) for i in range(min(num_devices, 4))]
    rng = np.random.default_rng(42)
    A = np.ascontiguousarray(rng.integers(-64, 64, (128, 256), dtype=np.int8))
    B = np.ascontiguousarray(rng.integers(-64, 64, (256, 128), dtype=np.int8))

    for m in miners:
        m.set_matrices(A, B)
        # Verify the array is on the expected device.
        assert m._A.devices() == {m._device}, \
            f"A on {m._A.devices()} but expected {m._device}"
        assert m._B.devices() == {m._device}, \
            f"B on {m._B.devices()} but expected {m._device}"


def test_independent_scans_different_devices(num_devices):
    """Different TpuMiner instances on different devices produce identical results."""
    M, N, K, rank = 128, 128, 256, 128
    rng = np.random.default_rng(99)
    A = np.ascontiguousarray(rng.integers(-64, 64, (M, K), dtype=np.int8))
    B = np.ascontiguousarray(rng.integers(-64, 64, (K, N), dtype=np.int8))

    a_seed = _blake3.blake3(b"test-a-multi").digest()
    b_seed = _blake3.blake3(b"test-b-multi").digest()
    E_AL = npnoise.generate_dense(a_seed, npnoise.SEED_A, M, rank)
    E_BR = npnoise.generate_dense(b_seed, npnoise.SEED_B, N, rank)
    r0a, r1a = npnoise.generate_sparse_indices(a_seed, npnoise.SEED_A, K, rank)
    r0b, r1b = npnoise.generate_sparse_indices(b_seed, npnoise.SEED_B, K, rank)

    # Get reference result.
    An, Bn = ref.apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b)
    tm_ref, tn_ref, tr_ref, tgt = _reference_winner(An, Bn, K, rank, a_seed)

    # Run mine_seeded on two different devices and compare.
    for dev_id in range(min(num_devices, 2)):
        miner = TpuMiner(dev_id)
        miner.set_matrices(A, B)
        raw = miner.mine_seeded(a_seed, b_seed, r0a, r1a, r0b, r1b, rank, tgt, None)
        assert raw is not None, f"device {dev_id}: no winner found"
        tm, tn, tb = raw
        assert (tm, tn) == (tm_ref, tn_ref), \
            f"device {dev_id}: tile {(tm,tn)} != ref {(tm_ref,tn_ref)}"
        assert list(struct.unpack_from("<16I", tb)) == list(tr_ref), \
            f"device {dev_id}: transcript mismatch"


def test_tiled_scan_different_device(num_devices):
    """Tiled scan works correctly on a non-default device."""
    M, N, K, rank = 1152, 1152, 256, 128
    rng = np.random.default_rng(77)
    A = np.ascontiguousarray(rng.integers(-64, 64, (M, K), dtype=np.int8))
    B = np.ascontiguousarray(rng.integers(-64, 64, (K, N), dtype=np.int8))

    a_seed = _blake3.blake3(b"test-a-tiled-multi").digest()
    b_seed = _blake3.blake3(b"test-b-tiled-multi").digest()
    E_AL = npnoise.generate_dense(a_seed, npnoise.SEED_A, M, rank)
    E_BR = npnoise.generate_dense(b_seed, npnoise.SEED_B, N, rank)
    r0a, r1a = npnoise.generate_sparse_indices(a_seed, npnoise.SEED_A, K, rank)
    r0b, r1b = npnoise.generate_sparse_indices(b_seed, npnoise.SEED_B, K, rank)

    An, Bn = ref.apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b)
    tm_ref, tn_ref, tr_ref, tgt = _reference_winner(An, Bn, K, rank, a_seed)

    # Use device 1 (non-default) for the tiled scan.
    dev_id = min(1, num_devices - 1)
    miner = TpuMiner(dev_id)
    miner.set_matrices(A, B)
    raw = miner.mine_seeded(a_seed, b_seed, r0a, r1a, r0b, r1b, rank, tgt, None)
    assert raw is not None, "tiled scan on non-default device: no winner found"
    tm, tn, tb = raw
    assert (tm, tn) == (tm_ref, tn_ref), f"tile {(tm,tn)} != ref {(tm_ref,tn_ref)}"
    assert list(struct.unpack_from("<16I", tb)) == list(tr_ref), "transcript mismatch"
    # Confirm tiled scan was used.
    assert miner._tiled is not None, "tiled scan path was not exercised"


def test_cuda_device_count_uses_local():
    """cuda_device_count() should report local devices, not global."""
    from prl_miner_tpu.tpu.miner import cuda_device_count
    count = cuda_device_count()
    assert count == len(jax.local_devices())
