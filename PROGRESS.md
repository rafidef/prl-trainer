# PRL TPU Miner — Progress

A Google **Cloud TPU (v5e / v6e)** miner for Pearl (PRL), based on the validated
`../prl-miner-turing/` codebase. The host pipeline (AlphaPool Stratum, noise
generation, Merkle trees, PlainProof serialization — all already "Share ACCEPTED"
on AlphaPool) is reused unchanged. The **only** new component is the search
kernel: the Turing/CUDA `_C.GpuMiner` is replaced by a JAX/XLA implementation
that runs on the TPU MXU + VPU.

Why this is a good fit: Pearl's NoisyGEMM work is **100% integer**
(`int7×int7→int32` GEMM + integer XOR-fold + single-block keyed BLAKE3). No
floating point → bit-identical to the CUDA kernel, and the int8 MXU is exactly
what v5e/v6e are built for.

## Milestone 1 — DONE ✅  (bit-exact JAX core, CPU-validated)

The JAX search core is bit-identical to the NumPy reference oracle (which mirrors
the live AlphaPool-accepted CUDA kernel). Validated on CPU (`JAX_PLATFORMS=cpu`).

Files:
- `src/prl_miner_tpu/tpu/blake3_jax.py` — single-block **keyed/root BLAKE3** in
  JAX, batched over candidate tiles. Validated against the `blake3` library.
- `src/prl_miner_tpu/tpu/noisy_gemm.py` — `build_scan(M,N,K,rank,rows,cols)`
  returns a jitted `f(An, Bn, key_words, target_words) -> (found, tm, tn,
  transcript[16])`. Hot path: int8 GEMM per rank-block → cumulative int32 →
  XOR-fold → keyed BLAKE3 → 256-bit target compare → lexicographic argmin.
- `src/prl_miner_tpu/reference.py` — NumPy oracle (apply_noise / ref_transcript /
  khash / full_scan), extracted from `prl-miner-turing/selftest.py`.
- `tests/test_blake3_jax.py`, `tests/test_bitexact.py` — acceptance gates.

Run the gate:
```bash
cd prl-miner-tpu
pip install "jax[cpu]" blake3 numpy pytest
JAX_PLATFORMS=cpu python -m pytest -q tests/      # 5 passed
```

### Key protocol facts (verified against the pearl monorepo + turing miner)
- Live AlphaPool profile: `M=N=131072, K=4096, rank=128`, `rows_pattern=[0,32]`,
  `cols_pattern=[0..63]`, `mma_type=Int7xInt7ToInt32`.
- Noise: `An[i,j]=clamp_i8(A[i,j]+E_AL[i,r0a[j]]-E_AL[i,r1a[j]])`, symmetric for B.
- Transcript per hash tile: accumulate `Cacc += An_rows @ Bn_cols` per rank-block;
  `combined = XOR(uint32 of cumulative Cacc)`; `tr[g%16] = rotl(tr[g%16],13) ^ combined`.
- PoW: `blake3_keyed(transcript_64B, key=commitment_A)` as LE-uint256 `<= target`.
- Interface to reuse: the miner only needs the **noised** GEMM→transcript→hash
  path. Denoising + PlainProof (Merkle/bincode) are host-side and already work.

## Milestone 2 — DONE ✅  (TpuMiner backend, GpuMiner-compatible, CPU-validated)

`TpuMiner` exposes the same surface the worker calls on `_C.GpuMiner`
(`set_matrices`/`update_A`/`mine`/`mine_seeded`/`debug_dense_noise`), applies the
low-rank ±1 noise in JAX, and runs `build_scan` on-device. Dense E_AL/E_BR noise
is generated on-device in JAX (no host BLAKE3 loop). Both `mine()` (explicit
noise) and `mine_seeded()` (on-chip noise from commitment seeds) return the same
winning tile + 64-byte transcript as the NumPy oracle.

Files:
- `src/prl_miner_tpu/noise.py` — NumPy noise (ported from turing, self-contained).
- `src/prl_miner_tpu/tpu/noise_jax.py` — `generate_dense_jax`, bit-exact vs NumPy.
- `src/prl_miner_tpu/tpu/miner.py` — `TpuMiner` + module helpers
  (`cuda_device_count`/`cuda_device_name`/`get_device_sm` for worker parity).
- `tests/test_noise_jax.py`, `tests/test_tpu_miner.py`.

Full suite: `JAX_PLATFORMS=cpu python -m pytest -q tests/` → **9 passed**.

## Milestone 3a — DONE ✅  (tiled/streaming scan, full-scale ready, CPU-validated)

`tpu/tiled_scan.py` streams the scan so the live 131072² grid (~134M tiles) never
materializes:
- Split into row-batches of `rbatch` 64-row blocks. Each 64-row block holds 32
  tile_m candidates (top half; rows {tile_m, tile_m+32}).
- Per batch, a `lax.scan` over the G rank-blocks carries running int32
  accumulators for top & bottom rows vs **all** col-blocks at once (one big MXU
  matmul/step), emitting one XOR-folded uint32 per (candidate, col-block).
- Transcripts folded (slot = g%16), keyed-BLAKE3'd, compared to target.
- Host driver loops row-batches with **first-hit early-exit**.
Only running accumulators (T×N int32) + per-k combined (G×T×NC u32) live in HBM.

`TpuMiner` auto-selects tiled vs the dense core by grid size
(`_TILED_THRESHOLD`), and `pick_rbatch` chooses the largest pow2 rbatch dividing
M/64 (override with `PRL_RBATCH`). Validated bit-exact vs the reference incl. a
non-multiple-of-16 G and a >1M-grid end-to-end run. Full suite: **13 passed**.

Important correctness fix made here: candidate `tile_m` is restricted to the top
`64-max_row_off` rows of each 64-block (i in 0..31 for [0,32]); enforced
`tile_m % 64 < 32`, matching selftest. (The earlier core allowed every row.)

## Milestone 3b — DONE ✅  (runnable, cloneable repo)

The full host pipeline is wired around `TpuMiner` into an installable package
(`pip install -e ".[tpu]"`, console script `prl-miner-tpu`):
- Pure host modules reused from turing: `stratum.py` (AlphaPool), `plain_proof.py`,
  `merkle.py`, `merkle_fast.py`, `noise.py`.
- `worker.py` (TpuWorker) — turing worker with the backend swapped to `TpuMiner`;
  same double-buffered prepare + async PlainProof submission.
- `app.py` (orchestrator), `args.py` (CLI/env), `__main__.py`, `selftest.py`
  (on-device end-to-end check).
- `tpu/challenge.py` — TPU `pearl.challenge` (difficulty-32) solver, since the
  CPU fallback is too slow and there's no CUDA solver. Validated vs blake3 lib.
- Packaging: `pyproject.toml`, `scripts/setup_tpu_vm.sh`, `README.md`,
  `.env.example`, `.gitignore`.

Validated on CPU: `pip install -e .` + console script work; full suite **15
passed**; `python -m prl_miner_tpu.selftest` passes end-to-end (dense noise,
dense-core + tiled-core mine_seeded, challenge solver).

## Milestone 4 — DONE ✅  (LIVE on a real TPU)

Ran on a Colab TPU (single chip, v5e-class): on-device selftest passed bit-exact,
and got **multiple live "Share ACCEPTED" from AlphaPool**. Full pipeline works end
to end (challenge solved on TPU ~92 Mh/s → scan → PlainProof → accepted).
Sustained ~6.86 TMAC/s effective at the first cut (≈RTX-2080 class, ~3–7% of TPU
int8 peak). Fixed a live bug: the challenge solver only searched 32-bit nonces, so
~37% of difficulty-32 seeds fell back to the CPU solver — now full uint64.

## Optimization pass (CPU-validated, bit-exact, 16 tests pass)
- **Fused top+bottom matmul**: one `(2T×128)@(128×N)` MXU op per k-block instead
  of two `(T×128)@(128×N)` — bigger matmul, half the launches.
- **Fast matrix RNG** (`rng.bytes` byte-fill) — ~0.5 GB A/B generation drops from
  tens of seconds to ~couple; cuts startup.
- **Bigger row-batch**: `pick_rbatch` cap raised to 16 (override `PRL_RBATCH`).
- **`PRL_MODE=bench`** (`bench.py`): full-scan TMAC/s with no pool/early-exit noise.

## Next (optional, biggest remaining levers)
- Multi-chip sharding across a v5e-8/v6e-8 slice (~linear scaling) — Colab is 1 chip.
- Reduce the per-k (2T×N) int32 HBM traffic (the XOR-fold is memory-bound); tune
  rbatch vs N tiling; consider int8/int16 intermediate tricks where exactness holds.
- Difficulty: at very high static d= the pool idle-disconnects (rare shares); use
  vardiff (omit PEARL_DIFFICULTY) or a moderate value so shares stay steady. — `bash scripts/setup_tpu_vm.sh`, run the
   selftest under `JAX_PLATFORMS=tpu`, then mine and confirm a live AlphaPool
   `Share ACCEPTED`. Then optimize throughput (rbatch tuning, MXU/VPU overlap,
   multi-chip sharding across a v5e/v6e slice). — `JAX_PLATFORMS=tpu`, run full-scale
   M=N=131072 by tiling the (T_m × T_n) candidate grid to fit HBM; confirm a live
   `Share ACCEPTED` from AlphaPool.
4. **Throughput** — pipeline MXU GEMM with VPU fold/hash, multi-chip (`pmap`/
   sharding across a v5e/v6e slice), tune tile sizes; report TMAC/s vs duty.

## Open performance question
At full scale the candidate grid is huge (M/64·32 × N/64 ≈ 134M tiles/scan). The
correctness core materializes `(T_m, T_n, G, P, C)`; the TPU version must **tile**
the scan (stream row/col blocks, early-exit on a hit like the CUDA kernel) rather
than materialize it. That's milestone 3/4 work.
