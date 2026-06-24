"""
TpuMiner — TPU backend with the same surface the worker uses on `_C.GpuMiner`.

Drop-in for the CUDA miner: it holds A/B on-device, applies the low-rank ±1 noise
in JAX, and runs the NoisyGEMM PoW scan (build_scan) on the TPU. Returns the same
`(tile_m, tile_n, transcript_bytes)` tuple (or None) the CUDA kernel returns.

Multi-chip: each TpuMiner instance pins its data and computation to a specific
TPU chip via jax.device_put so that multiple TpuMiner instances (one per chip)
truly run in parallel.

API parity with prl_miner._C.GpuMiner:
    set_matrices(A, B) / update_A(A)
    mine(E_AL, r0a, r1a, E_BR, r0b, r1b, rank, key, target_le, ncb=None) -> raw|None
    mine_seeded(pow_key, b_seed, r0a, r1a, r0b, r1b, rank, target_le, ncb=None) -> raw|None
    debug_dense_noise(key, which, rows, rank) -> np.ndarray
The `ncb` (CUDA kernel-variant) argument is accepted and ignored on TPU.
"""
from __future__ import annotations

import os
import struct

import jax
import jax.numpy as jnp
import numpy as np

from . import noisy_gemm as ng
from . import tiled_scan as ts
from . import noise_jax
from . import blake3_jax as bj
from ..noise import SEED_A, SEED_B

# Live AlphaPool profile. Scans are specialized per (shape, profile); we cache one
# compiled scan per distinct (M, N, K, rank) so re-jit only happens on change.
_ROWS = (0, 32)
_COLS = tuple(range(64))

# Above this candidate-grid size, materializing the dense scan is wasteful/OOM, so
# switch to the streaming tiled scan. (The live profile is far above this.)
_TILED_THRESHOLD = 1024 * 1024


def _target_words_from_le(target_le: bytes) -> np.ndarray:
    if len(target_le) != 32:
        raise ValueError("target must be 32 bytes (little-endian uint256)")
    return np.frombuffer(target_le, dtype="<u4").astype(np.uint32)


def _apply_noise(A, B, E_AL, r0a, r1a, E_BR, r0b, r1b):
    """An[i,j]=clamp(A+E_AL[:,r0a]-E_AL[:,r1a]); Bn[k,j]=clamp(B+E_BR[:,r0b].T-E_BR[:,r1b].T).

    Not globally @jax.jit'd — called via the per-device jitted wrapper in TpuMiner.
    """
    A = A.astype(jnp.int32)
    An = jnp.clip(A + E_AL[:, r0a].astype(jnp.int32) - E_AL[:, r1a].astype(jnp.int32), -128, 127)
    B = B.astype(jnp.int32)
    EBR0 = E_BR[:, r0b].astype(jnp.int32).T   # (k, n)
    EBR1 = E_BR[:, r1b].astype(jnp.int32).T
    Bn = jnp.clip(B + EBR0 - EBR1, -128, 127)
    return An.astype(jnp.int8), Bn.astype(jnp.int8)


def _detect_max_rbatch() -> int:
    """Pick a sensible max_rbatch based on HBM size.

    v4 = 32 GB HBM → can handle larger rbatch.
    v5e = 16 GB → stick to 16.
    """
    try:
        dev = jax.local_devices()[0]
        # jax.devices() memory attributes vary by version; fallback to 16.
        stats = getattr(dev, 'memory_stats', lambda: None)()
        if stats and 'bytes_limit' in stats:
            hbm_gb = stats['bytes_limit'] / (1024 ** 3)
            if hbm_gb >= 28:  # v4 has 32 GB
                return 32
    except Exception:
        pass
    return 16


class TpuMiner:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        # Pin to a specific local chip. On single-host this is jax.devices()[id].
        local_devs = jax.local_devices()
        if device_id >= len(local_devs):
            raise ValueError(f"device_id={device_id} but only {len(local_devs)} local devices")
        self._device = local_devs[device_id]
        self.platform = jax.default_backend()
        self.sm = 0  # not meaningful on TPU; kept for API parity
        self._A = None          # device int8 (m, k)
        self._B = None          # device int8 (k, n)
        self._scan = None
        self._scan_key = None   # (M, N, K, rank)
        self._tiled = None
        self._tiled_key = None  # (M, N, K, rank)
        # Optional override of the auto-picked rbatch (e.g. via env on the TPU box).
        self.rbatch = int(os.environ.get("PRL_RBATCH", "0")) or None
        # Optional column-batch tiling override.
        self.ncbatch = int(os.environ.get("PRL_NCBATCH", "0")) or None
        # Auto-detect max rbatch from HBM size.
        self._max_rbatch = _detect_max_rbatch()
        # Per-device jitted noise function (compiles for self._device).
        self._jit_apply_noise = jax.jit(_apply_noise, device=self._device)

    # ---- matrix upload (pinned to self._device) ----------------------------
    def set_matrices(self, A: np.ndarray, B: np.ndarray) -> None:
        self._A = jax.device_put(
            jnp.asarray(np.ascontiguousarray(A, dtype=np.int8)), self._device)
        self._B = jax.device_put(
            jnp.asarray(np.ascontiguousarray(B, dtype=np.int8)), self._device)

    def update_A(self, A: np.ndarray) -> None:
        self._A = jax.device_put(
            jnp.asarray(np.ascontiguousarray(A, dtype=np.int8)), self._device)

    # ---- scan compilation cache -------------------------------------------
    def _get_scan(self, M, N, K, rank):
        key = (M, N, K, rank)
        if self._scan_key != key:
            self._scan = ng.build_scan(M, N, K, rank, _ROWS, _COLS)
            self._scan_key = key
        return self._scan

    def _get_tiled(self, M, N, K, rank):
        key = (M, N, K, rank)
        if self._tiled_key != key:
            rb = self.rbatch or ts.pick_rbatch(M, max_rbatch=self._max_rbatch)
            self._tiled = ts.build_tiled_scan(M, N, K, rank, _ROWS, _COLS, rbatch=rb,
                                              ncbatch=self.ncbatch)
            self._tiled_key = key
        return self._tiled

    # ---- core: scan over already-noised matrices --------------------------
    def _run(self, An, Bn, K, rank, key: bytes, target_le: bytes):
        M = int(An.shape[0])
        N = int(Bn.shape[1])
        kw = bj.key_words_from_bytes(key)
        tw = jax.device_put(jnp.asarray(_target_words_from_le(target_le)), self._device)
        kw = jax.device_put(kw, self._device)
        if M * N >= _TILED_THRESHOLD:
            scan = self._get_tiled(M, N, K, rank)
            return ts.find_share_device(scan, An, Bn, kw, tw)
        # Small problems: the dense single-shot core is simpler and plenty fast.
        meets, tm, tn, tr = self._get_scan(M, N, K, rank)(An, Bn, kw, tw)
        if not bool(meets):
            return None
        tr_np = np.asarray(tr, dtype=np.uint32)
        return int(tm), int(tn), struct.pack("<16I", *[int(w) for w in tr_np])

    # ---- public mining entrypoints ----------------------------------------
    def mine(self, E_AL, r0a, r1a, E_BR, r0b, r1b, rank, key, target_le, ncb=None):
        """Explicit dense noise supplied (matches GpuMiner.mine)."""
        assert self._A is not None and self._B is not None, "call set_matrices first"
        K = int(self._A.shape[1])
        An, Bn = self._jit_apply_noise(
            self._A, self._B,
            jax.device_put(jnp.asarray(E_AL, dtype=jnp.int8), self._device),
            jax.device_put(jnp.asarray(r0a), self._device),
            jax.device_put(jnp.asarray(r1a), self._device),
            jax.device_put(jnp.asarray(E_BR, dtype=jnp.int8), self._device),
            jax.device_put(jnp.asarray(r0b), self._device),
            jax.device_put(jnp.asarray(r1b), self._device),
        )
        return self._run(An, Bn, K, rank, key, target_le)

    def mine_seeded(self, pow_key, b_seed, r0a, r1a, r0b, r1b, rank, target_le, ncb=None):
        """Dense noise generated on-device from the commitment seeds (live path)."""
        assert self._A is not None and self._B is not None, "call set_matrices first"
        M = int(self._A.shape[0])
        N = int(self._B.shape[1])
        K = int(self._A.shape[1])
        E_AL = noise_jax.generate_dense_jax(pow_key, SEED_A, M, rank)
        E_BR = noise_jax.generate_dense_jax(b_seed, SEED_B, N, rank)
        # Pin noise to this device before applying.
        E_AL = jax.device_put(E_AL, self._device)
        E_BR = jax.device_put(E_BR, self._device)
        An, Bn = self._jit_apply_noise(
            self._A, self._B,
            E_AL,
            jax.device_put(jnp.asarray(r0a), self._device),
            jax.device_put(jnp.asarray(r1a), self._device),
            E_BR,
            jax.device_put(jnp.asarray(r0b), self._device),
            jax.device_put(jnp.asarray(r1b), self._device),
        )
        # pow_key (commitment_A) keys the PoW BLAKE3.
        return self._run(An, Bn, K, rank, pow_key, target_le)

    # ---- debug -------------------------------------------------------------
    def debug_dense_noise(self, key, which, rows, rank):
        seed = SEED_A if which == 0 else SEED_B
        return np.asarray(noise_jax.generate_dense_jax(key, seed, rows, rank))


def cuda_device_count() -> int:  # name kept for worker parity
    return len(jax.local_devices())


def cuda_device_name(i: int = 0) -> str:
    devs = jax.local_devices()
    return str(devs[i]) if i < len(devs) else "tpu?"


def get_device_sm(device: int = 0) -> int:
    return 0
