#!/usr/bin/env bash
# Stop PRL mining on all worker VMs of a v4-64 pod.
#
# Run this from your local workstation. It sends SIGTERM to prl-miner-tpu
# processes on every VM. The "|| true" ensures we get a clean exit even if
# no miner process is running on a given VM.
#
# Usage:
#   bash scripts/stop_v4_pod.sh [TPU_NAME] [ZONE]
#
# Example:
#   bash scripts/stop_v4_pod.sh prl-miner us-central2-b
#
set -euo pipefail

TPU_NAME="${1:-prl-miner}"
ZONE="${2:-us-central2-b}"

echo "==> Stopping mining on all workers of ${TPU_NAME}..."

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" --worker=all \
    --command="pkill -f prl-miner-tpu || true"

echo "==> Stopped."
