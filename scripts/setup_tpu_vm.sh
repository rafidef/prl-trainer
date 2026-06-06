#!/usr/bin/env bash
# One-command setup for the PRL TPU miner ON a Google Cloud TPU VM.
#
# First provision a TPU VM from your workstation (example, v5e single host):
#
#   gcloud compute tpus tpu-vm create prl-miner \
#       --zone=us-central1-a --accelerator-type=v5litepod-1 \
#       --version=tpu-ubuntu2204-base
#   gcloud compute tpus tpu-vm ssh prl-miner --zone=us-central1-a
#
#   # v6e (Trillium) example: --accelerator-type=v6e-1 --zone=us-east5-a
#   # Cheaper if you use --spot (preemptible).
#
# Then on the TPU VM:
#   git clone <your-repo-url> prl-miner-tpu && cd prl-miner-tpu
#   bash scripts/setup_tpu_vm.sh
#
set -euo pipefail

echo "==> Python: $(python3 --version)"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

echo "==> Installing JAX for TPU + the miner"
pip install -e ".[tpu,dev]"

echo "==> Verifying JAX sees the TPU"
JAX_PLATFORMS=tpu python3 - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
assert jax.default_backend() == "tpu", "JAX is not using the TPU — check libtpu install"
PY

echo "==> Running the on-device selftest (bit-exact correctness)"
JAX_PLATFORMS=tpu PRL_MODE=selftest python3 -m prl_miner_tpu.selftest

cat <<'EOF'

Setup complete. To mine (replace the wallet address):

  source .venv/bin/activate
  JAX_PLATFORMS=tpu prl-miner-tpu \
      --address prl1pYOURWALLET... \
      --pool us1.alphapool.tech:5566 \
      --worker tpu1 \
      --password 'x;d=20000'

  # Or run a quick bench/selftest any time:
  JAX_PLATFORMS=tpu PRL_MODE=selftest python3 -m prl_miner_tpu.selftest
EOF
