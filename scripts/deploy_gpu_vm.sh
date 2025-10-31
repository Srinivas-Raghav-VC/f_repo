#!/usr/bin/env bash
set -euo pipefail

# Simple helper to copy this repo to a GPU VM and launch the HPC runner.
# Requirements on the VM: Python3, pip, GPU drivers/CUDA, tmux or nohup, outbound email if you want notifications.
# Usage:
#   export VM_SSH="user@host"             # required
#   export VM_DIR="/home/user/mmie_run"   # required
#   export EMAIL_TO=you@example.com EMAIL_FROM=you@gmail.com GMAIL_APP_PASSWORD=app_password  # optional, for emails
#   ./scripts/deploy_gpu_vm.sh --model Qwen/Qwen2.5-1.5B-Instruct --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl --auto --auto_plots --device cuda

if [[ -z "${VM_SSH:-}" || -z "${VM_DIR:-}" ]]; then
  echo "Please set VM_SSH and VM_DIR environment variables." >&2
  exit 1
fi

cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE="mmie_${STAMP}.tar.gz"

echo "[deploy] Packing repo -> ${ARCHIVE}"
tar --exclude-vcs -czf "$ARCHIVE" .

echo "[deploy] Creating VM dir: ${VM_DIR}"
ssh "$VM_SSH" "mkdir -p '$VM_DIR'"

echo "[deploy] Uploading archive"
scp "$ARCHIVE" "$VM_SSH":"$VM_DIR"/

echo "[deploy] Unpacking on VM"
ssh "$VM_SSH" "cd '$VM_DIR' && tar -xzf '$ARCHIVE' && rm -f '$ARCHIVE' && chmod +x run_mmie_hpc.sh"

echo "[deploy] Launching run on VM"
ssh "$VM_SSH" "cd '$VM_DIR' && EMAIL_TO='${EMAIL_TO:-}' EMAIL_FROM='${EMAIL_FROM:-}' GMAIL_APP_PASSWORD='${GMAIL_APP_PASSWORD:-}' \
  ./run_mmie_hpc.sh $*"

echo "[deploy] Done. Check your email (if configured) or VM logs under logs/."

