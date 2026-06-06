"""
Throughput benchmark — run with PRL_MODE=bench (no pool, no early-exit noise):

    JAX_PLATFORMS=tpu PRL_MODE=bench python -m prl_miner_tpu
    # knobs: PRL_BENCH_M / _N / _K / _ITERS, PRL_RBATCH

Times full NoisyGEMM scans (target = 0 so nothing is ever "found", forcing the
scan to cover every row-batch) and reports clean TMAC/s. This is the honest
measure the pool's early-exit per-scan numbers can't give.
"""
from __future__ import annotations

import os
import time

import blake3 as _blake3
import jax

from .tpu.miner import TpuMiner
from . import noise as npnoise
from .worker import _fast_int8_matrix
import numpy as np


def main() -> None:
    M = int(os.environ.get("PRL_BENCH_M", "131072"))
    N = int(os.environ.get("PRL_BENCH_N", str(M)))
    K = int(os.environ.get("PRL_BENCH_K", "4096"))
    rank = 128
    iters = int(os.environ.get("PRL_BENCH_ITERS", "5"))

    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
    print(f"bench: M={M} N={N} K={K} rank={rank} iters={iters} "
          f"rbatch={os.environ.get('PRL_RBATCH', 'auto')}")

    rng = np.random.default_rng(0)
    t = time.time()
    A = _fast_int8_matrix(rng, M, K)
    B = _fast_int8_matrix(rng, K, N)
    print(f"matrix gen: {time.time() - t:.1f}s")

    miner = TpuMiner(0)
    miner.set_matrices(A, B)

    a_seed = _blake3.blake3(b"bench-a").digest()
    b_seed = _blake3.blake3(b"bench-b").digest()
    r0a, r1a = npnoise.generate_sparse_indices(a_seed, npnoise.SEED_A, K, rank)
    r0b, r1b = npnoise.generate_sparse_indices(b_seed, npnoise.SEED_B, K, rank)
    hard = (0).to_bytes(32, "little")   # impossible target -> full scan, no early-exit

    tiles = (M // 64 * 32) * (N // 64)
    macs = tiles * (2 * 64 * K)         # == M*N*K

    def one():
        return miner.mine_seeded(a_seed, b_seed, r0a, r1a, r0b, r1b, rank, hard, None)

    t = time.time()
    one()
    print(f"warmup (incl. XLA compile): {time.time() - t:.1f}s")

    best = 0.0
    for i in range(iters):
        t = time.time()
        one()
        dt = time.time() - t
        tmac = macs / dt / 1e12
        best = max(best, tmac)
        print(f"iter {i}: {dt:.2f}s   {tmac:.2f} TMAC/s")

    print(f"\nbest: {best:.2f} TMAC/s   ({tiles:,} tiles/scan, {macs/1e12:.1f} TMAC/scan)")


if __name__ == "__main__":
    main()
