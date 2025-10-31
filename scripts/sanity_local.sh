#!/usr/bin/env bash
set -euo pipefail

# Minimal local sanity check for MMIE on Qwen 1.5B.
# - Verifies environment
# - Checks dataset presence
# - Runs a tiny selection-only pass (no training)
# - Optionally runs a very short SAE gate smoke test if CUDA is available

cd "$(dirname "$0")/.."

# Resolve Python interpreter (prefer python3)
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "[error] No Python interpreter found (need python3 or python)" >&2
  exit 1
fi

echo "[sanity] Environment preflight"
"$PY" scripts/preflight.py || true

echo "[sanity] Dataset check"
"$PY" scripts/check_datasets.py --paths \
  data/forget_hi.jsonl data/retain_en.jsonl data/mixed.jsonl \
  data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl || true

# Disable judge by default for speed in sanity runs
export GEMINI_API_KEY="${GEMINI_API_KEY:-}"

# Prefer 8-bit if bitsandbytes is available
export LOAD_IN_8BIT="${LOAD_IN_8BIT:-1}"
export SAFETENSORS_FAST="${SAFETENSORS_FAST:-0}"

DEVICE=$($PY - << 'PY'
try:
    import torch
    print('cuda' if (hasattr(torch,'cuda') and torch.cuda.is_available()) else 'cpu')
except Exception:
    print('cpu')
PY
)

echo "[sanity] Using device: ${DEVICE}"

echo "[sanity] Selection-only smoke (no training)"
"$PY" mmie.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --lora_steps 0 --reft_steps 0 --train_sae_steps 0 \
  --sample_cap 20 --max_len 64 \
  --select_mode semantic --min_layer 6 --select_top_k 2 \
  --print_layer_scores \
  --device "${DEVICE}" \
  --out sanity_selection.json

if [[ "${DEVICE}" == "cuda" ]]; then
  echo "[sanity] Optional SAE gate smoke (very short)"
  "$PY" mmie.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
    --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
    --ckpt_dir ckpt_sanity \
    --train_sae_steps 50 --sae_k 32 --sae_expansion 8 \
    --lora_steps 0 --reft_steps 0 \
    --select_mode semantic --min_layer 6 --select_top_k 1 \
    --sample_cap 40 --max_len 64 \
    --device cuda \
    --out sanity_sae_gate.json || true
fi

echo "[sanity] Done. Outputs: sanity_selection.json (and sanity_sae_gate.json if CUDA)."
