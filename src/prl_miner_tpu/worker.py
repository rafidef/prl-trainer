"""
TPU mining worker — one instance per TPU device. Adapted from the validated
prl-miner-turing GPU worker; the only backend change is CUDA `_C.GpuMiner` ->
`TpuMiner` (same set_matrices/update_A/mine_seeded surface). The host pipeline
(noise sparse gen, Merkle, PlainProof, double-buffered prepare, async proof
submission) is unchanged.

Flow per session:
  1. Generate random A (m×k int8) and B (k×n int8) once from matrix_seed. B is
     uploaded once; A is refreshed per iteration (only A mutates).
  2. Per search space: derive job_key + commitment seeds, compute the tiny sparse
     ±1 index arrays on CPU, and let TpuMiner.mine_seeded() generate the dense
     noise on-device, apply it, and run the NoisyGEMM PoW scan.
  3. On a hit, build the PlainProof (async, cached B^T) and submit.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import random
import threading
import time
from typing import Callable, Optional

import numpy as np
import blake3 as _blake3

from .stratum import Job, MiningParams
from .noise import NoiseGenerator
from .merkle import make_merkle_tree
from .plain_proof import (
    build_plain_proof,
    _serialize_mining_config,
    _offset_is_valid,
    pad_to_chunk_boundary,
    derive_job_key,
    derive_commitment_seeds,
)
from .tpu.miner import TpuMiner

log = logging.getLogger(__name__)


def _fast_int8_matrix(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    """Uniform int8 matrix in [-64, 63] via a fast PCG byte-fill.

    The miner's A/B are arbitrary (we mine, not run real inference), so only the
    Pearl data range matters, not the exact distribution. rng.bytes fills at
    ~GB/s — far faster than rng.integers over a range — cutting startup
    allocation of the ~0.5 GB matrices from tens of seconds to a couple.
    """
    buf = np.frombuffer(rng.bytes(rows * cols), dtype=np.uint8)
    return ((buf & np.uint8(0x7F)).astype(np.int8) - np.int8(64)).reshape(rows, cols)


def _format_hashrate(rate: float) -> str:
    units = ["H/s", "KH/s", "MH/s", "GH/s", "TH/s"]
    value = float(rate)
    for unit in units:
        if abs(value) < 1000.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1000.0
    return f"{value:.2f} H/s"


class TpuWorker:
    """Manages mining on a single TPU device. submit_callback: async coroutine(job, proof_b64)."""

    def __init__(
        self,
        device_id: int,
        submit_callback: Callable,
        matrix_seed: Optional[int] = None,
        status_interval: int = 30,
    ):
        self.device_id = device_id
        self.submit_callback = submit_callback
        self.matrix_seed = matrix_seed or random.getrandbits(64)
        self.status_interval = status_interval

        self._job: Optional[Job] = None
        self._params: Optional[MiningParams] = None
        self._A: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._Bt_flat: Optional[bytes] = None
        self._miner: Optional[TpuMiner] = None
        self._stop_flag = threading.Event()
        self._job_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._hashes_total = 0
        self._scan_seconds_total = 0.0
        self._shares_found = 0
        self._last_status = time.monotonic()
        self._last_status_hashes = 0
        self._last_status_scan_seconds = 0.0
        self._last_status_shares = 0

        self._current_k = 4096
        self._current_share_target = 0
        self._current_difficulty_factor = 1

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._proof_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._proof_sem = threading.BoundedSemaphore(2)
        self._next_space_future = None
        self._current_job_id = None
        self._matrices_uploaded = False
        self._bt_root_cache = (None, None)
        self._bt_tree_cache = (None, None)
        self._last_status_wall = time.monotonic()

        log.info("TpuWorker device=%d seed=0x%016x", device_id, self.matrix_seed)

    def set_job(self, job: Job, params: MiningParams) -> None:
        with self._job_lock:
            self._job = job
            self._params = params
        self._stop_flag.set()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"miner-tpu{self.device_id}"
        )
        self._thread.start()

    def _record_hashrate(self, job_id: str, hashes: int, elapsed: float) -> None:
        if hashes <= 0 or elapsed <= 0:
            return
        self._hashes_total += hashes
        self._scan_seconds_total += elapsed

        macs_per_hash = 2 * 64 * self._current_k  # 1 tile = 2 rows × 64 cols × k MACs
        current_tmac_s = (hashes / elapsed * macs_per_hash) / 1e12
        log.info("TPU %d job=%s scan %.2fs, %.2f TMAC/s",
                 self.device_id, job_id, elapsed, current_tmac_s)

        now = time.monotonic()
        if self.status_interval > 0 and now - self._last_status >= self.status_interval:
            window_hashes = self._hashes_total - self._last_status_hashes
            window_seconds = self._scan_seconds_total - self._last_status_scan_seconds
            window_shares = self._shares_found - self._last_status_shares
            whps = window_hashes / window_seconds if window_seconds > 0 else hashes / elapsed
            tmac_s = (whps * macs_per_hash) / 1e12
            wall_window = now - self._last_status_wall
            effective_tmac_s = ((window_hashes * macs_per_hash) / wall_window / 1e12
                                if wall_window > 0 else 0.0)
            if self._current_share_target > 0:
                adjusted_target = self._current_share_target * self._current_difficulty_factor
                probability = adjusted_target / (2 ** 256)
                expected = window_hashes * probability
                equiv_tmac_s = tmac_s * (window_shares / expected) if expected > 0 else 0.0
            else:
                equiv_tmac_s = tmac_s
            log.info("TPU %d status attempts=%d hits=%d peak_tmac_s=%.2f "
                     "effective_tmac_s=%.2f (duty %.0f%%) equiv_tmac_s=%.2f",
                     self.device_id, window_hashes, window_shares, tmac_s, effective_tmac_s,
                     (100.0 * effective_tmac_s / tmac_s) if tmac_s > 0 else 0.0, equiv_tmac_s)
            self._last_status = now
            self._last_status_wall = now
            self._last_status_hashes = self._hashes_total
            self._last_status_scan_seconds = self._scan_seconds_total
            self._last_status_shares = self._shares_found

    def _bt_root(self, job_key: bytes) -> bytes:
        ck, cv = self._bt_root_cache
        if ck == job_key:
            return cv
        root = _blake3.blake3(self._Bt_flat, key=job_key, max_threads=_blake3.blake3.AUTO).digest()
        self._bt_root_cache = (job_key, root)
        return root

    def _prepare_space(self, job: Job, params: MiningParams, mutate: bool,
                       A: Optional[np.ndarray]) -> dict:
        m, n, k = params.m, params.n, params.k
        rank = params.rank

        if A is None:
            log.info("Allocating A(%d×%d) + B(%d×%d)…", m, k, k, n)
            t_alloc = time.monotonic()
            rng = np.random.default_rng(self.matrix_seed)
            A = _fast_int8_matrix(rng, m, k)
            self._B = _fast_int8_matrix(rng, k, n)
            self._Bt_flat = pad_to_chunk_boundary(np.ascontiguousarray(self._B.T).tobytes())
            log.info("Allocated + transposed in %.1fs", time.monotonic() - t_alloc)

        if mutate:
            A = A.copy()
            val = A[0, 0]
            A[0, 0] = -64 if val == 63 else val + 1

        A_flat = pad_to_chunk_boundary(A.tobytes())
        mining_config_bytes = _serialize_mining_config(k, rank, params.rows_pattern, params.cols_pattern)
        job_key = derive_job_key(job.header_bytes, mining_config_bytes)
        A_root = _blake3.blake3(A_flat, key=job_key, max_threads=_blake3.blake3.AUTO).digest()
        Bt_root = self._bt_root(job_key)
        b_seed, a_seed = derive_commitment_seeds(job_key, A_root, Bt_root)

        r0a, r1a, r0b, r1b = NoiseGenerator(rank=rank).generate_sparse(a_seed, b_seed, k)
        return {
            'A': A, 'pow_key': a_seed, 'b_seed': b_seed, 'job_key': job_key,
            'r0a': r0a, 'r1a': r1a, 'r0b': r0b, 'r1b': r1b,
        }

    def _run(self) -> None:
        self._miner = TpuMiner(self.device_id)
        log.info("TpuWorker thread started (device %d, backend=%s)",
                 self.device_id, self._miner.platform)
        while True:
            if self._job is None:
                self._stop_flag.wait()
            self._stop_flag.clear()

            with self._job_lock:
                job = self._job
                params = self._params
            if job is None or params is None:
                continue

            try:
                if job.job_id != self._current_job_id:
                    self._current_job_id = job.job_id
                    self._next_space_future = None
                    log.info("Mining NEW job=%s", job.job_id)
                    space = self._prepare_space(job, params, mutate=False, A=self._A)
                    if self._A is None:
                        self._A = space['A']
                else:
                    if self._next_space_future is not None:
                        space = self._next_space_future.result()
                    else:
                        space = self._prepare_space(job, params, mutate=True, A=self._A)

                self._A = space['A']
                self._next_space_future = self._executor.submit(
                    self._prepare_space, job, params, True, self._A
                )
                self._mine_space(job, params, space)
            except Exception as e:
                log.exception("Mining error: %s", e)
                time.sleep(1.0)

    def _mine_space(self, job: Job, params: MiningParams, space: dict) -> None:
        t0 = time.monotonic()
        m, n, k = params.m, params.n, params.k
        rank = params.rank

        fast_ok = (
            list(params.rows_pattern) == [0, 32]
            and list(params.cols_pattern) == list(range(64))
            and rank == 128
            and m % 64 == 0 and n % 64 == 0 and k % 128 == 0
        )
        if not fast_ok:
            log.error("Unsupported mining profile (rows=%s cols=%d.. rank=%d m=%d n=%d k=%d). "
                      "Skipping job.", params.rows_pattern, params.cols_pattern[0] if params.cols_pattern else -1,
                      rank, m, n, k)
            return

        if not self._matrices_uploaded:
            self._miner.set_matrices(space['A'], self._B)
            self._matrices_uploaded = True
        else:
            self._miner.update_A(space['A'])

        difficulty_factor = len(params.rows_pattern) * len(params.cols_pattern) * k
        adjusted_target = job.share_target * difficulty_factor
        self._current_k = k
        self._current_share_target = job.share_target
        self._current_difficulty_factor = difficulty_factor
        target_le = min(adjusted_target, 2 ** 256 - 1).to_bytes(32, "little")

        scan_hashes = (m // 64 * 32) * (n // 64)
        scan_t0 = time.monotonic()
        try:
            raw = self._miner.mine_seeded(
                space['pow_key'], space['b_seed'],
                space['r0a'], space['r1a'], space['r0b'], space['r1b'],
                rank, target_le, None,
            )
        except Exception as e:
            log.error("TPU mine failed: %s", e, exc_info=True)
            return
        self._record_hashrate(job.job_id, scan_hashes, time.monotonic() - scan_t0)

        if raw is None:
            return
        tm, tn, tb = raw
        if not (_offset_is_valid(tm, params.rows_pattern) and _offset_is_valid(tn, params.cols_pattern)):
            return

        self._shares_found += 1
        log.info("Share found: job=%s tile_m=%d tile_n=%d (%.1fs)",
                 job.job_id, tm, tn, time.monotonic() - t0)

        a_row_indices = [tm + off for off in params.rows_pattern]
        bt_row_indices = [tn + off for off in params.cols_pattern]

        if self._proof_sem.acquire(blocking=False):
            self._proof_executor.submit(
                self._build_and_submit_proof,
                job, space, a_row_indices, bt_row_indices, m, n, k, rank, self._B
            )
        else:
            log.debug("Proof queue full; skipping share for job=%s", job.job_id)

    def _bt_tree(self, job_key: bytes):
        ck, cv = self._bt_tree_cache
        if ck == job_key:
            return cv
        tree = make_merkle_tree(self._Bt_flat, job_key)
        self._bt_tree_cache = (job_key, tree)
        return tree

    def _build_and_submit_proof(self, job: Job, space: dict, a_row_indices: list,
                                bt_row_indices: list, m: int, n: int, k: int,
                                rank: int, B: np.ndarray) -> None:
        log.info("Building PlainProof for job %s…", job.job_id)
        proof_t0 = time.monotonic()
        try:
            proof_b64 = build_plain_proof(
                A=space['A'], B=B, job_key=space['job_key'],
                a_row_indices=a_row_indices, bt_row_indices=bt_row_indices,
                m=m, n=n, k=k, noise_rank=rank,
                bt_bytes=self._Bt_flat, tree_bt=self._bt_tree(space['job_key']),
            )
        except Exception as e:
            log.error("PlainProof build failed: %s", e, exc_info=True)
            return
        finally:
            self._proof_sem.release()
        log.info("PlainProof built for job %s in %.2fs", job.job_id, time.monotonic() - proof_t0)

        if self._loop is None or self._loop.is_closed():
            return
        fut = asyncio.run_coroutine_threadsafe(self.submit_callback(job, proof_b64), self._loop)
        fut.add_done_callback(lambda f: f.exception() and log.error("Share submit failed: %s", f.exception()))
