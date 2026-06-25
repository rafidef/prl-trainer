#!/usr/bin/env bash
# Launch PRL mining on all worker VMs of a v4-64 pod.
#
# Run this from your local workstation. Each VM auto-detects its process
# index via jax.process_index() and appends "-wN" to the worker name so
# the pool can distinguish workers.
#
# Mining runs under nohup so it survives SSH disconnection.
# Logs are written to /tmp/prl-miner-tpu.log on each VM.
#
# Usage:
#   bash scripts/run_v4_pod.sh <TPU_NAME> <ZONE> <WALLET_ADDRESS> [POOL] [PASSWORD] [WORKER_BASE]
#
# Example:
#   bash scripts/run_v4_pod.sh prl-miner us-central2-b prl1pABC123...
#
# Stop mining on all workers:
#   bash scripts/stop_v4_pod.sh prl-miner us-central2-b
#   # or manually:
#   gcloud compute tpus tpu-vm ssh prl-miner --zone=us-central2-b --worker=all \
#       --command="pkill -f prl-miner-tpu || true"
#
set -euo pipefail

TPU_NAME="${1:?Usage: $0 <TPU_NAME> <ZONE> <WALLET_ADDRESS> [POOL] [PASSWORD] [WORKER_BASE]}"
ZONE="${2:?Usage: $0 <TPU_NAME> <ZONE> <WALLET_ADDRESS>}"
WALLET="${3:?Usage: $0 <TPU_NAME> <ZONE> <WALLET_ADDRESS>}"
POOL="${4:-us1.alphapool.tech:5566}"
PASSWORD="${5:-x;d=20000}"
WORKER_BASE="${6:-tpu-v4}"

echo "==> Starting mining on all workers of ${TPU_NAME}..."
echo "    Pool:        ${POOL}"
echo "    Wallet:      ${WALLET}"
echo "    Worker base: ${WORKER_BASE} (each VM appends -wN)"

# --worker=all sends the command to every VM concurrently.
# nohup + backgrounding (&) keeps the miner running after SSH exits.
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" --worker=all --tunnel-through-iap \
    --command="cd prl-trainer && source .venv/bin/activate && \
        nohup env JAX_PLATFORMS=tpu \
        prl-trainer \
            --address '${WALLET}' \
            --pool '${POOL}' \
            --worker '${WORKER_BASE}' \
            --password '${PASSWORD}' \
        > /tmp/prl-trainer.log 2>&1 < /dev/null &"

echo "==> Mining launched on all workers. Check logs with:"
echo "    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all --tunnel-through-iap --command='tail -20 /tmp/prl-trainer.log'"
