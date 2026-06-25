#!/usr/bin/env bash
# Stop the background Monero python miner across all VMs in the TPU pod.
#
# Usage:
#   bash scripts/stop_monero.sh <TPU_NAME> <ZONE>
#
# Example:
#   bash scripts/stop_monero.sh node-3 us-central2-b

set -euo pipefail

TPU_NAME="${1:?Usage: $0 <TPU_NAME> <ZONE>}"
ZONE="${2:?Usage: $0 <TPU_NAME> <ZONE>}"

echo "==> Stopping trainer (tpu-tensor.py) on all workers of ${TPU_NAME}..."

# We kill both the python script and the detached screen session.
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" --worker=all --tunnel-through-iap \
    --command="pkill -f tpu-tensor.py || true; screen -S tpu_worker -X quit || true"

echo "==> Trainer stopped on all VMs. The CPU is now fully available for Pearl!"
