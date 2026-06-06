"""
End-to-end TpuMiner validation — mirrors prl_miner.selftest's mine()/mine_seeded()
winner+transcript checks, but drives the TPU backend against the NumPy oracle.
"""
import os
import struct
import sys

import numpy as np
import blake3 as _blake3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prl_miner_tpu import reference as ref                       # noqa: E402
from prl_miner_tpu import noise as npnoise                       # noqa: E402
from prl_miner_tpu.tpu.miner import TpuMiner                     # noqa: E402

ROWS = (0, 32)
COLS = tuple(range(64))
M = N = 128
K, rank = 256, 128


def _winner_target(An, Bn, key):
    best, _ = ref.full_scan(An, Bn, K, rank, key, target_int=-1,
                            rows_pattern=ROWS, cols_pattern=COLS)
    tm, tn, tr, hmin = best
    return tm, tn, tr, hmin.to_bytes(32, "little")


def test_mine_explicit_noise_matches_reference():
    miner = TpuMiner(0)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        A = np.ascontiguousarray(rng.integers(-64, 64, (M, K), dtype=np.int8))
        B = np.ascontiguousarray(rng.integers(-64, 64, (K, N), dtype=np.int8))
        E_AL = np.ascontiguousarray(rng.integers(-32, 32, (M, rank), dtype=np.int8))
        E_BR = np.ascontiguousarray(rng.integers(-32, 32, (N, rank), dtype=np.int8))
        r0a = rng.integers(0, rank, K, dtype=np.int32)
        r1a = rng.integers(0, rank, K, dtype=np.int32)
        r0b = rng.integers(0, rank, K, dtype=np.int32)
        r1b = rng.integers(0, rank, K, dtype=np.int32)
        key = bytes(rng.integers(0, 256, 32, dtype=np.uint8).tolist())

        An, Bn = ref.apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b)
        tm_r, tn_r, tr_r, tgt = _winner_target(An, Bn, key)

        miner.set_matrices(A, B)
        raw = miner.mine(E_AL, r0a, r1a, E_BR, r0b, r1b, rank, key, tgt, ncb=6)
        assert raw is not None, f"seed {seed}: expected a winner at its own target"
        tm, tn, tb = raw
        assert (tm, tn) == (tm_r, tn_r), f"seed {seed}: tile {(tm,tn)} != {(tm_r,tn_r)}"
        assert list(struct.unpack_from("<16I", tb)) == list(tr_r), \
            f"seed {seed}: transcript mismatch"


def test_mine_seeded_onchip_noise_matches_reference():
    """The live path: dense noise generated on-device from commitment seeds."""
    miner = TpuMiner(0)
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        A = np.ascontiguousarray(rng.integers(-64, 64, (M, K), dtype=np.int8))
        B = np.ascontiguousarray(rng.integers(-64, 64, (K, N), dtype=np.int8))
        a_seed = _blake3.blake3(b"a" + bytes([seed])).digest()
        b_seed = _blake3.blake3(b"b" + bytes([seed])).digest()

        # Reference noise from the same seeds (CPU), then winner under tight target.
        E_AL = npnoise.generate_dense(a_seed, npnoise.SEED_A, M, rank)
        E_BR = npnoise.generate_dense(b_seed, npnoise.SEED_B, N, rank)
        r0a, r1a = npnoise.generate_sparse_indices(a_seed, npnoise.SEED_A, K, rank)
        r0b, r1b = npnoise.generate_sparse_indices(b_seed, npnoise.SEED_B, K, rank)
        An, Bn = ref.apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b)
        tm_r, tn_r, tr_r, tgt = _winner_target(An, Bn, a_seed)  # pow_key == a_seed

        miner.set_matrices(A, B)
        raw = miner.mine_seeded(a_seed, b_seed, r0a, r1a, r0b, r1b, rank, tgt, ncb=6)
        assert raw is not None, f"mine_seeded seed {seed}: expected a winner"
        tm, tn, tb = raw
        assert (tm, tn) == (tm_r, tn_r), \
            f"mine_seeded seed {seed}: tile {(tm,tn)} != {(tm_r,tn_r)}"
        assert list(struct.unpack_from("<16I", tb)) == list(tr_r), \
            f"mine_seeded seed {seed}: transcript mismatch"


def test_mine_seeded_uses_tiled_path_at_scale():
    """At a grid above the tiled threshold, mine_seeded must auto-use the streaming
    scan and still return the reference winner."""
    from prl_miner_tpu.tpu import miner as miner_mod
    Mb = Nb = 1152          # 1152*1152 > 1024*1024 threshold; 1152/64=18 -> rbatch picks 2
    Kb, rankb = 256, 128
    miner = TpuMiner(0)
    rng = np.random.default_rng(2024)
    A = np.ascontiguousarray(rng.integers(-64, 64, (Mb, Kb), dtype=np.int8))
    B = np.ascontiguousarray(rng.integers(-64, 64, (Kb, Nb), dtype=np.int8))
    a_seed = _blake3.blake3(b"scale-a").digest()
    b_seed = _blake3.blake3(b"scale-b").digest()
    E_AL = npnoise.generate_dense(a_seed, npnoise.SEED_A, Mb, rankb)
    E_BR = npnoise.generate_dense(b_seed, npnoise.SEED_B, Nb, rankb)
    r0a, r1a = npnoise.generate_sparse_indices(a_seed, npnoise.SEED_A, Kb, rankb)
    r0b, r1b = npnoise.generate_sparse_indices(b_seed, npnoise.SEED_B, Kb, rankb)
    An, Bn = ref.apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b)
    best, _ = ref.full_scan(An, Bn, Kb, rankb, a_seed, target_int=-1,
                            rows_pattern=ROWS, cols_pattern=COLS)
    tm_r, tn_r, tr_r, hmin = best

    miner.set_matrices(A, B)
    raw = miner.mine_seeded(a_seed, b_seed, r0a, r1a, r0b, r1b, rankb,
                            hmin.to_bytes(32, "little"), ncb=6)
    assert miner._tiled is not None, "expected the tiled scan path to be used"
    assert raw is not None
    tm, tn, tb = raw
    assert (tm, tn) == (tm_r, tn_r), f"scale: tile {(tm,tn)} != {(tm_r,tn_r)}"
    assert list(struct.unpack_from("<16I", tb)) == list(tr_r)


def test_update_a_changes_result_path():
    """update_A refreshes A on-device (smoke test of the per-iteration path)."""
    miner = TpuMiner(0)
    rng = np.random.default_rng(7)
    A = np.ascontiguousarray(rng.integers(-64, 64, (M, K), dtype=np.int8))
    B = np.ascontiguousarray(rng.integers(-64, 64, (K, N), dtype=np.int8))
    miner.set_matrices(A, B)
    A2 = A.copy()
    A2[0, 0] = -64 if A2[0, 0] == 63 else A2[0, 0] + 1
    miner.update_A(A2)
    assert int(np.asarray(miner._A)[0, 0]) == int(A2[0, 0])


if __name__ == "__main__":
    test_mine_explicit_noise_matches_reference()
    print("mine() explicit-noise winner matches reference: OK")
    test_mine_seeded_onchip_noise_matches_reference()
    print("mine_seeded() on-chip-noise winner matches reference: OK")
    test_update_a_changes_result_path()
    print("update_A path: OK")
    print("\nMilestone 2 ACCEPTANCE: TpuMiner is bit-exact vs reference (mine + mine_seeded).")
