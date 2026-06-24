#!/usr/bin/env bash
# Setup all worker VMs in a v4-64 pod from your local workstation.
#
# This script SSHs into every worker VM of a TPU v4-64 pod (8 VMs, 4 chips
# each) and runs the on-VM setup script (scripts/setup_tpu_vm.sh) on all of
# them simultaneously.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - TPU pod already created, e.g.:
#       gcloud compute tpus tpu-vm create prl-miner \
#           --zone=us-central2-b --accelerator-type=v4-64 \
#           --version=tpu-ubuntu2204-base
#   - This repo cloned on every worker VM:
#       gcloud compute tpus tpu-vm ssh prl-miner --zone=us-central2-b \
#           --worker=all --command="git clone <repo-url> prl-miner-tpu"
#
# Usage:
#   bash scripts/setup_v4_pod.sh [TPU_NAME] [ZONE]
#
# Example:
#   bash scripts/setup_v4_pod.sh prl-miner us-central2-b
#
set -euo pipefail

TPU_NAME="${1:-prl-trainer}"
ZONE="${2:-us-central2-b}"

echo "==> Setting up all workers on ${TPU_NAME} in ${ZONE}..."

# --worker=all fans out the SSH command to every VM in the pod concurrently.
gcloud alpha compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" --worker=all --tunnel-through-iap \
    --command="if [ ! -d 'prl-trainer' ]; then git clone https://github.com/rafidef/prl-trainer.git; fi && cd prl-trainer && bash scripts/setup_tpu_vm.sh"

echo "==> Setup complete on all workers."
echo "To start mining:"
echo "  bash scripts/run_v4_pod.sh ${TPU_NAME} ${ZONE} prl1pYOURWALLET..."
