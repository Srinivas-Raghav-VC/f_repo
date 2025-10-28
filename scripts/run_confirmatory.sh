#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
FORGET=${FORGET:-data/forget_hi.jsonl}
RETAIN=${RETAIN:-data/retain_en.jsonl}
MIXED=${MIXED:-data/mixed.jsonl}
XLANG_UR=${XLANG_UR:-data/urdu.jsonl}
XLANG_PA=${XLANG_PA:-data/punjabi.jsonl}
XLANG_BN=${XLANG_BN:-data/bengali.jsonl}
OUT=${OUT:-eval_report.json}

python mmie.py \
  --model "$MODEL" \
  --forget "$FORGET" --retain "$RETAIN" --mixed "$MIXED" \
  --xlang "$XLANG_UR" "$XLANG_PA" "$XLANG_BN" \
  --ckpt_dir ${CKPT_DIR:-ckpt_lora_final} \
  --train_sae_steps 2000 --sae_k 32 \
  --sae_gate --sae_gate_alpha 0.5 --sae_gate_topk 64 \
  --semantic_features --dynamic_gate --semantic_dynamic_gate \
  --report_token_kl \
  --device ${DEVICE:-cuda} \
  --out "$OUT"

echo "[ok] Wrote $OUT"

# Generate plots
python tools/plots_from_report.py "$OUT" || true
