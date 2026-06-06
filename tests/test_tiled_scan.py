"""
Tiled/streaming scan must be bit-identical to the NumPy reference, and its
host driver must return the reference winner. Run at a small multi-batch,
multi-col-block size so the tiling logic is actually exercised.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prl_miner_tpu import reference as ref                  # noqa: E402
from prl_miner_tpu.tpu import tiled_scan as ts              # noqa: E402

ROWS = (0, 32)
COLS = tuple(range(64))
M = N = 256
K, rank = 384, 128        # G = 3 (not a multiple of 16 -> exercises the fold loop)
RBATCH = 2                # span = 128 -> 2 row-batches


def _rand_noised(seed):
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
    return An, Bn, key


def test_batch_transcripts_bit_identical():
    scan = ts.build_tiled_scan(M, N, K, rank, ROWS, COLS, rbatch=RBATCH)
    NC = N // 64
    span = RBATCH * 64
    import jax.numpy as jnp
    for seed in range(4):
        An, Bn, _ = _rand_noised(seed)
        An_j, Bn_j = jnp.asarray(An), jnp.asarray(Bn)
        for b in range(scan.num_batches):
            rb0 = b * span
            tr_batch = np.asarray(scan.debug_batch(An_j, Bn_j, jnp.int32(rb0)))  # (T,NC,16)
            T = tr_batch.shape[0]
            for c in range(T):
                tile_m = rb0 + (c // 32) * 64 + (c % 32)
                for cb in range(NC):
                    tile_n = cb * 64
                    tr_ref = ref.ref_transcript(An, Bn, tile_m, tile_n, K, rank, ROWS, COLS)
                    assert np.array_equal(tr_batch[c, cb], np.array(tr_ref, dtype=np.uint32)), (
                        f"seed {seed} batch {b} tile ({tile_m},{tile_n}) mismatch")


def test_find_first_share_matches_reference_winner():
    scan = ts.build_tiled_scan(M, N, K, rank, ROWS, COLS, rbatch=RBATCH)
    for seed in range(6):
        An, Bn, key = _rand_noised(seed)
        best, _ = ref.full_scan(An, Bn, K, rank, key, target_int=-1,
                                rows_pattern=ROWS, cols_pattern=COLS)
        tm_r, tn_r, tr_r, hmin = best
        target_le = hmin.to_bytes(32, "little")
        out = ts.find_first_share(scan, An, Bn, key, target_le)
        assert out is not None, f"seed {seed}: winner should meet its own target"
        tm, tn, tb = out
        assert tm % 64 < 32 and tn % 64 == 0
        assert (tm, tn) == (tm_r, tn_r), f"seed {seed}: {(tm,tn)} != {(tm_r,tn_r)}"
        assert list(struct.unpack_from("<16I", tb)) == list(tr_r), \
            f"seed {seed}: transcript mismatch"


def test_find_first_share_target_bounds():
    scan = ts.build_tiled_scan(M, N, K, rank, ROWS, COLS, rbatch=RBATCH)
    An, Bn, key = _rand_noised(99)
    assert ts.find_first_share(scan, An, Bn, key, (0).to_bytes(32, "little")) is None
    assert ts.find_first_share(scan, An, Bn, key, (2**256 - 1).to_bytes(32, "little")) is not None


if __name__ == "__main__":
    test_batch_transcripts_bit_identical()
    print("tiled batch transcripts bit-identical: OK")
    test_find_first_share_matches_reference_winner()
    print("tiled find_first_share matches reference winner: OK")
    test_find_first_share_target_bounds()
    print("tiled target bounds: OK")
    print("\nMilestone 3a ACCEPTANCE: tiled/streaming scan is bit-exact and ready for TPU.")
