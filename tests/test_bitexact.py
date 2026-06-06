"""
Milestone-1 acceptance gate: the JAX/TPU NoisyGEMM core must be bit-identical to
the NumPy reference oracle (which mirrors the live AlphaPool-accepted CUDA kernel).

Mirrors prl_miner.selftest: same noise model, same live profile
(rows=[0,32], cols=[0..63], rank=128), random A/B/E in the same ranges.
"""
import os
import sys

import numpy as np
import blake3 as _blake3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prl_miner_tpu import reference as ref            # noqa: E402
from prl_miner_tpu.tpu import noisy_gemm as ng        # noqa: E402
from prl_miner_tpu.tpu import blake3_jax as bj        # noqa: E402

ROWS = (0, 32)
COLS = tuple(range(64))


def _rand_case(seed, M, N, K, rank):
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


def test_transcripts_bit_identical():
    """Every candidate tile's 64-byte transcript must match the reference exactly."""
    M = N = 128
    K, rank = 256, 128
    for seed in range(6):
        An, Bn, _ = _rand_case(seed, M, N, K, rank)
        for tm, tn in ref.candidate_tiles(M, N, ROWS, COLS):
            tr_ref = ref.ref_transcript(An, Bn, tm, tn, K, rank, ROWS, COLS)
            tr_jax = ng.transcript_for_tile(An, Bn, tm, tn, K, rank, ROWS, COLS)
            assert np.array_equal(np.array(tr_ref, dtype=np.uint32), tr_jax), (
                f"seed {seed} tile ({tm},{tn}) transcript mismatch:\n"
                f" ref={tr_ref}\n jax={list(tr_jax)}")


def test_winner_matches_reference():
    """Under the tightest target (= global min hash) the JAX scan returns the
    same winning tile + transcript as the reference."""
    M = N = 128
    K, rank = 256, 128
    scan = ng.build_scan(M, N, K, rank, ROWS, COLS)
    for seed in range(8):
        An, Bn, key = _rand_case(seed, M, N, K, rank)
        best, _ = ref.full_scan(An, Bn, K, rank, key, target_int=-1,
                                rows_pattern=ROWS, cols_pattern=COLS)
        tm_ref, tn_ref, tr_ref, hmin = best
        tw = np.frombuffer(hmin.to_bytes(32, "little"), dtype="<u4").astype(np.uint32)
        kw = bj.key_words_from_bytes(key)
        meets, tm, tn, tr = scan(An, Bn, kw, tw)
        tm, tn = int(tm), int(tn)
        assert bool(meets), f"seed {seed}: winner should meet its own hash target"
        # Kernel invariant: tile_m is in the top half of its 64-block (rows_pattern[0,32]).
        assert tm % 64 < 32, f"seed {seed}: tile_m {tm} not in top half (tm%64={tm % 64})"
        assert tn % 64 == 0, f"seed {seed}: tile_n {tn} not 64-aligned"
        assert (tm, tn) == (tm_ref, tn_ref), (
            f"seed {seed}: winner tile {(tm, tn)} != ref {(tm_ref, tn_ref)}")
        assert np.array_equal(np.asarray(tr, dtype=np.uint32),
                              np.array(tr_ref, dtype=np.uint32)), \
            f"seed {seed}: winner transcript mismatch"


def test_found_flag_respects_target():
    """An impossibly-hard target (0) yields no share; an easy target (max) does."""
    M = N = 128
    K, rank = 256, 128
    scan = ng.build_scan(M, N, K, rank, ROWS, COLS)
    An, Bn, key = _rand_case(123, M, N, K, rank)
    kw = bj.key_words_from_bytes(key)
    hard = np.zeros(8, dtype=np.uint32)
    easy = np.full(8, 0xFFFFFFFF, dtype=np.uint32)
    meets_hard, _, _, _ = scan(An, Bn, kw, hard)
    meets_easy, _, _, _ = scan(An, Bn, kw, easy)
    assert not bool(meets_hard)
    assert bool(meets_easy)


if __name__ == "__main__":
    test_transcripts_bit_identical()
    print("transcripts bit-identical: OK")
    test_winner_matches_reference()
    print("winner matches reference: OK")
    test_found_flag_respects_target()
    print("found-flag respects target: OK")
    print("\nMilestone 1 ACCEPTANCE: JAX core is bit-exact vs reference.")
