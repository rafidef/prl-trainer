"""
Throughput benchmark — run with PRL_MODE=bench (no pool, no early-exit noise):

    JAX_PLATFORMS=tpu PRL_MODE=bench python -m prl_miner_tpu
    # knobs: PRL_BENCH_M / _N / _K / _ITERS, PRL_RBATCH, PRL_NCBATCH

Times full NoisyGEMM scans (target = 0 so nothing is ever "found", forcing the
scan to cover every row-batch) and reports clean TMAC/s. This is the honest
measure the pool's early-exit per-scan numbers can't give.

Multi-chip: benchmarks each local chip independently and reports aggregate.
"""
from __future__ import annotations

import os
import time
import threading

import blake3 as _blake3
import jax

from .tpu.miner import TpuMiner
from . import noise as npnoise
from .worker import _fast_int8_matrix
import numpy as np


def _bench_one_chip(device_id: int, M: int, N: int, K: int, rank: int,
                    iters: int, results: dict):
    """Run benchmark on a single chip. Stores results in results[device_id]."""
    miner = TpuMiner(device_id)
    rng = np.random.default_rng(device_id)
    A = _fast_int8_matrix(rng, M, K)
    B = _fast_int8_matrix(rng, K, N)
    miner.set_matrices(A, B)

    a_seed = _blake3.blake3(b"bench-a" + bytes([device_id])).digest()
    b_seed = _blake3.blake3(b"bench-b" + bytes([device_id])).digest()
    r0a, r1a = npnoise.generate_sparse_indices(a_seed, npnoise.SEED_A, K, rank)
    r0b, r1b = npnoise.generate_sparse_indices(b_seed, npnoise.SEED_B, K, rank)
    hard = (0).to_bytes(32, "little")   # impossible target -> full scan, no early-exit

    tiles = (M // 64 * 32) * (N // 64)
    macs = tiles * (2 * 64 * K)         # == M*N*K

    def one():
        return miner.mine_seeded(a_seed, b_seed, r0a, r1a, r0b, r1b, rank, hard, None)

    t = time.time()
    one()
    warmup_s = time.time() - t
    print(f"  chip {device_id}: warmup (incl. XLA compile): {warmup_s:.1f}s")

    best = 0.0
    for i in range(iters):
        t = time.time()
        one()
        dt = time.time() - t
        tmac = macs / dt / 1e12
        best = max(best, tmac)
        print(f"  chip {device_id}: iter {i}: {dt:.2f}s   {tmac:.2f} TMAC/s")

    results[device_id] = {
        'best_tmac_s': best,
        'tiles': tiles,
        'macs': macs,
        'warmup_s': warmup_s,
    }


def main() -> None:
    # Initialize distributed runtime (required on multi-host pods).
    try:
        jax.distributed.initialize()
    except Exception:
        pass

    M = int(os.environ.get("PRL_BENCH_M", "131072"))
    N = int(os.environ.get("PRL_BENCH_N", str(M)))
    K = int(os.environ.get("PRL_BENCH_K", "4096"))
    rank = 128
    iters = int(os.environ.get("PRL_BENCH_ITERS", "5"))
    multi = os.environ.get("PRL_BENCH_MULTI", "1").lower() in ("1", "true", "yes", "all")

    local_devs = jax.local_devices()
    process_idx = jax.process_index()
    num_devices = len(local_devs)

    print(f"JAX backend: {jax.default_backend()} | process: {process_idx} | "
          f"local devices: {num_devices}: {local_devs}")
    print(f"bench: M={M} N={N} K={K} rank={rank} iters={iters} "
          f"rbatch={os.environ.get('PRL_RBATCH', 'auto')} "
          f"ncbatch={os.environ.get('PRL_NCBATCH', 'auto')} "
          f"multi={multi}")

    if not multi or num_devices <= 1:
        # Single-chip benchmark (original behavior).
        results = {}
        _bench_one_chip(0, M, N, K, rank, iters, results)
        r = results[0]
        print(f"\nbest: {r['best_tmac_s']:.2f} TMAC/s   "
              f"({r['tiles']:,} tiles/scan, {r['macs']/1e12:.1f} TMAC/scan)")
        # Estimate % of peak (275 TOPS for v4, ~197 for v5e).
        print(f"% of 275 TOPS (v4): {r['best_tmac_s'] / 275 * 100:.1f}%")
        print(f"% of 197 TOPS (v5e): {r['best_tmac_s'] / 197 * 100:.1f}%")
        return

    # Multi-chip: benchmark all local chips in parallel.
    print(f"\n=== Benchmarking {num_devices} chips in parallel ===\n")
    t_alloc = time.time()
    # Pre-generate matrices once (shared by all chips — they each copy to device).
    print(f"Generating {M}×{K} + {K}×{N} matrices...")
    print(f"matrix gen overhead shared across chips")

    results = {}
    threads = []
    for dev_id in range(num_devices):
        th = threading.Thread(target=_bench_one_chip,
                              args=(dev_id, M, N, K, rank, iters, results))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()

    # Aggregate results.
    total_best = sum(r['best_tmac_s'] for r in results.values())
    print(f"\n=== Results (process {process_idx}) ===")
    for dev_id in sorted(results):
        r = results[dev_id]
        print(f"  chip {dev_id}: {r['best_tmac_s']:.2f} TMAC/s "
              f"({r['tiles']:,} tiles/scan)")
    print(f"\n  AGGREGATE: {total_best:.2f} TMAC/s across {num_devices} chips")
    print(f"  % of {num_devices}×275 TOPS (v4): "
          f"{total_best / (275 * num_devices) * 100:.1f}%")
    print(f"  % of {num_devices}×197 TOPS (v5e): "
          f"{total_best / (197 * num_devices) * 100:.1f}%")


if __name__ == "__main__":
    main()
