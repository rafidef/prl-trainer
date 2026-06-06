"""
On-device end-to-end selftest — run on the TPU VM before mining:

    JAX_PLATFORMS=tpu python -m prl_miner_tpu  (with PRL_MODE=selftest)
    or: python -m prl_miner_tpu.selftest

Checks, against an independent NumPy reference:
  1. on-device dense noise == CPU reference,
  2. TpuMiner.mine_seeded returns the correct winning tile + transcript (both the
     dense single-shot core and the tiled/streaming scan at >1M grid),
  3. the TPU pearl.challenge solver finds a valid difficulty-16 nonce.
Prints the JAX backend so you can confirm you are actually on the TPU.
"""
import struct

import numpy as np
import blake3 as _blake3
import jax

from . import reference as ref
from . import noise as npnoise
from .tpu.miner import TpuMiner
from .tpu import noise_jax
from .tpu.challenge import solve_challenge_tpu

ROWS = (0, 32)
COLS = tuple(range(64))


def _winner_target(An, Bn, K, rank, key):
    best, _ = ref.full_scan(An, Bn, K, rank, key, target_int=-1,
                            rows_pattern=ROWS, cols_pattern=COLS)
    tm, tn, tr, hmin = best
    return tm, tn, tr, hmin.to_bytes(32, "little")


def _check_mine_seeded(miner, M, N, K, rank, label):
    rng = np.random.default_rng(hash((M, N, K)) & 0xFFFF)
    A = np.ascontiguousarray(rng.integers(-64, 64, (M, K), dtype=np.int8))
    B = np.ascontiguousarray(rng.integers(-64, 64, (K, N), dtype=np.int8))
    a_seed = _blake3.blake3(b"a" + label.encode()).digest()
    b_seed = _blake3.blake3(b"b" + label.encode()).digest()
    E_AL = npnoise.generate_dense(a_seed, npnoise.SEED_A, M, rank)
    E_BR = npnoise.generate_dense(b_seed, npnoise.SEED_B, N, rank)
    r0a, r1a = npnoise.generate_sparse_indices(a_seed, npnoise.SEED_A, K, rank)
    r0b, r1b = npnoise.generate_sparse_indices(b_seed, npnoise.SEED_B, K, rank)
    An, Bn = ref.apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b)
    tm_r, tn_r, tr_r, tgt = _winner_target(An, Bn, K, rank, a_seed)

    miner.set_matrices(A, B)
    raw = miner.mine_seeded(a_seed, b_seed, r0a, r1a, r0b, r1b, rank, tgt, None)
    assert raw is not None, f"{label}: no winner found"
    tm, tn, tb = raw
    assert (tm, tn) == (tm_r, tn_r), f"{label}: tile {(tm,tn)} != {(tm_r,tn_r)}"
    assert list(struct.unpack_from("<16I", tb)) == list(tr_r), f"{label}: transcript mismatch"
    print(f"  mine_seeded {label}: OK  tile=({tm},{tn})")


def main() -> None:
    print("JAX backend:", jax.default_backend(), "| devices:", jax.devices())
    miner = TpuMiner(0)

    # 1. Dense noise on-device == CPU.
    for which, seed_fixed in ((0, npnoise.SEED_A), (1, npnoise.SEED_B)):
        for rows in (128, 257):
            key = _blake3.blake3(bytes([which]) + seed_fixed[:4]).digest()
            gpu = np.asarray(miner.debug_dense_noise(key, which, rows, 128))
            cpu = npnoise.generate_dense(key, seed_fixed, rows, 128)
            assert np.array_equal(gpu, cpu), f"dense noise mismatch which={which} rows={rows}"
    print("  dense noise (on-device == CPU): OK")

    # 2. mine_seeded — dense core and tiled core.
    _check_mine_seeded(miner, 128, 128, 256, 128, "dense-core")
    _check_mine_seeded(miner, 1152, 1152, 256, 128, "tiled-core")
    assert miner._tiled is not None, "tiled scan path was not exercised"

    # 3. Challenge solver.
    seed = _blake3.blake3(b"challenge").digest()
    nonce = solve_challenge_tpu(seed, 16, max_nonce=1 << 22)
    h = _blake3.blake3(seed + int(nonce).to_bytes(8, "little")).digest()
    lead = (int.from_bytes(h, "big").bit_length())
    assert 256 - lead >= 16, "challenge nonce does not meet difficulty"
    print(f"  challenge solver (difficulty 16): OK  nonce={nonce}")

    print("\nALL SELFTESTS PASSED — TPU miner is correct on this device.")


if __name__ == "__main__":
    main()
