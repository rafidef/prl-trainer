# PRL Miner — Google Cloud TPU (v4 / v5e / v6e)

An open-source [Pearl (PRL)](https://github.com/pearl-research-labs/pearl) miner
for **Google Cloud TPU** (v4, v5e, and v6e/Trillium), mining on **AlphaPool**.

Pearl's proof-of-useful-work (`NoisyGEMM`) is **100% integer math** — `int7×int7→int32`
matmul, integer XOR-folds, and a single keyed-BLAKE3 block. That maps perfectly onto
the TPU's int8 MXU + VPU and, because it's integer, the TPU produces **bit-identical**
results to the reference CUDA kernel. This miner reuses the host pipeline (AlphaPool
Stratum, Merkle, PlainProof) from the validated `prl-miner-turing` build and swaps the
CUDA search kernel for a JAX/XLA one.

> **Heads-up on economics.** PRL mining revenue has been sliding and Cloud TPU rental
> is not cheap (on-demand v4/v5e/v6e ≈ \$1–2+/chip-hr; cheaper with `--spot`). Model
> profitability before a long run. This project is about making it *work* on TPU; whether
> it *pays* is a separate question.

## Quick start (on a Cloud TPU VM)

```bash
# 1. Provision a TPU VM (from your workstation)
gcloud compute tpus tpu-vm create prl-miner \
    --zone=us-central1-a --accelerator-type=v5litepod-1 \
    --version=tpu-ubuntu2204-base
gcloud compute tpus tpu-vm ssh prl-miner --zone=us-central1-a

# 2. On the VM
git clone <your-repo-url> prl-miner-tpu && cd prl-miner-tpu
bash scripts/setup_tpu_vm.sh          # installs jax[tpu], runs the selftest

# 3. Mine
source .venv/bin/activate
JAX_PLATFORMS=tpu prl-miner-tpu \
    --address prl1pYOURWALLET... \
    --pool us1.alphapool.tech:5566 \
    --worker tpu1 \
    --password 'x;d=20000'
```

v6e (Trillium): `--accelerator-type=v6e-1 --zone=us-east5-a`. Add `--spot` to the
`create` call for preemptible (cheaper) capacity.

## Quick start (v4-64 pod)

A v4-64 pod has **8 worker VMs × 4 chips each = 32 chips**. All 8 VMs must be
running simultaneously (TPU pod requirement).

```bash
# 1. Provision a v4-64 pod (from your workstation)
gcloud compute tpus tpu-vm create prl-miner \
    --zone=us-central2-b \
    --accelerator-type=v4-64 \
    --version=tpu-ubuntu2204-base

# 2. Setup all workers (installs deps + runs selftest on every VM)
bash scripts/setup_v4_pod.sh prl-miner us-central2-b

# 3. Start mining on all workers
bash scripts/run_v4_pod.sh prl-miner us-central2-b prl1pYOURWALLET...

# 4. Check logs across all workers
gcloud compute tpus tpu-vm ssh prl-miner \
    --zone=us-central2-b \
    --worker=all \
    --command='tail -20 /tmp/prl-miner-tpu.log'

# 5. Stop mining on all workers
bash scripts/stop_v4_pod.sh prl-miner us-central2-b
```

> **Note:** All 8 VMs must be running simultaneously — this is a TPU pod
> requirement. Each VM mines independently after `jax.distributed.initialize()`
> with its own pool connection, matrices, and 4 local chips.

## Local dev / correctness validation (no TPU)

Everything except the live pool run is bit-exact-validated on CPU:

```bash
pip install -e ".[cpu,dev]"
JAX_PLATFORMS=cpu python -m pytest -q tests/        # full suite
JAX_PLATFORMS=cpu PRL_MODE=selftest python -m prl_miner_tpu.selftest
```

## How it works

```
AlphaPool ──Stratum──► StratumClient ──► TpuWorker ──► TpuMiner (JAX/TPU)
   ▲                       │                  │            │
   │  mining.submit        │ pearl.challenge  │            ├─ apply low-rank ±1 noise
   │  (base64 PlainProof)  │ (TPU solver)     │            ├─ tiled int8 NoisyGEMM scan (MXU)
   │                       │                  │            ├─ XOR-fold transcript (VPU)
   └───────────────────────┴── PlainProof ◄───┘            └─ keyed BLAKE3 + target (VPU)
                              (Merkle, host)
```

- **`tpu/noisy_gemm.py`** — dense single-shot scan (small / tests).
- **`tpu/tiled_scan.py`** — streaming scan for full 131072² scale: row-batches
  through the MXU, `lax.scan` over k-blocks, first-hit early-exit. `TpuMiner`
  auto-selects tiled vs dense by grid size (`PRL_RBATCH` overrides rows/batch).
- **`tpu/blake3_jax.py`** — single-block keyed/plain BLAKE3, vectorized on the VPU.
- **`tpu/challenge.py`** — TPU `pearl.challenge` (difficulty-32) solver.
- **`tpu/noise_jax.py`** — on-device dense noise generation.
- Host pipeline (`stratum.py`, `plain_proof.py`, `merkle*.py`, `noise.py`) is the
  validated AlphaPool path, reused unchanged.

## Configuration

CLI flags or env vars (see `.env.example`): `--address/WALLET_ADDRESS`,
`--pool/POOL`, `--worker/WORKER_NAME`, `--password/POOL_PASSWORD` (`x;d=N` for
static difficulty; AlphaPool minimum is 20000), `--devices/DEVICES`. Set
`JAX_PLATFORMS=tpu` on the VM.

## Performance tuning

| Variable | Description |
|---|---|
| `PRL_RBATCH` | Row-batch size (auto-selected). v4 default is larger due to 32 GB HBM per chip (vs v5e's 16 GB). |
| `PRL_NCBATCH` | Column-batch tiling for reduced HBM traffic. Tiles the N dimension so the accumulator fits in VMEM/L1 instead of streaming through HBM every k-step. Default: auto. |

## Status

| Component | State |
|---|---|
| Bit-exact JAX NoisyGEMM core + keyed BLAKE3 | ✅ CPU-validated |
| `TpuMiner` backend (mine / mine_seeded) | ✅ CPU-validated |
| Tiled/streaming scan (full scale) | ✅ CPU-validated |
| TPU challenge solver | ✅ CPU-validated |
| Host pipeline (Stratum/Merkle/PlainProof) | ✅ reused from turing (live AlphaPool-accepted) |
| Multi-chip device pinning | ✅ |
| Multi-host v4 pod support | ✅ |
| Live `Share ACCEPTED` on a real TPU VM | ⬜ run it and confirm |

Pool endpoints (AlphaPool): PPLNS `:5566`, SOLO `:5567`, regions
`us1/us2/eu1/eu2/ru1/sg1.alphapool.tech`.

## Pearl pattern assumptions

Specialized for the live profile `rows_pattern=[0,32]`, `cols_pattern=[0..63]`,
`rank=128`, `mma_type=Int7xInt7ToInt32`. The worker skips jobs that don't match.

## License

ISC, matching the upstream Pearl project.
