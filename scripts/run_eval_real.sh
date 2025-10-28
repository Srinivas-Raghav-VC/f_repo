#!/usr/bin/env bash
set -euo pipefail

# Evaluate on small real sets if you place them under data/real/
# Expected files: data/real/forget_hi.jsonl, data/real/retain_en.jsonl, data/real/mixed.jsonl

MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
FORGET=${FORGET:-data/real/forget_hi.jsonl}
RETAIN=${RETAIN:-data/real/retain_en.jsonl}
MIXED=${MIXED:-data/real/mixed.jsonl}
XLANG_UR=${XLANG_UR:-data/urdu.jsonl}
XLANG_PA=${XLANG_PA:-data/punjabi.jsonl}
XLANG_BN=${XLANG_BN:-data/bengali.jsonl}
OUT=${OUT:-eval_real.json}

python mmie.py \
  --model "$MODEL" \
  --forget "$FORGET" --retain "$RETAIN" --mixed "$MIXED" \
  --xlang "$XLANG_UR" "$XLANG_PA" "$XLANG_BN" \
  --ckpt_dir ${CKPT_DIR:-ckpt_lora_final} \
  --train_sae_steps 0 --sae_k 32 --sae_expansion 8 \
  --sae_gate --sae_gate_alpha 0.5 --sae_gate_topk 64 \
  --semantic_features --semantic_tau 0.0 \
  --dynamic_gate --semantic_dynamic_gate \
  --gate_es_forget_ratio 0.5 --gate_es_mixed_ratio 0.7 --gate_ppl_ratio 1.10 \
  --seeds 42 \
  --device ${DEVICE:-cuda} \
  --out "$OUT"

python tools/plots_from_report.py "$OUT" || true
echo "[ok] Real-set evaluation saved to $OUT"

