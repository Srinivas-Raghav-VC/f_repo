#!/usr/bin/env bash
set -euo pipefail

# Qwen 1.5B Instruct (multilingual signal; fits on 8–12 GB with fp16 or int4)
# If HF model id differs in your setup, override MODEL=...
MODEL=${MODEL:-Qwen/Qwen2-1.5B-Instruct}
FORGET=${FORGET:-data/forget_hi.jsonl}
RETAIN=${RETAIN:-data/retain_en.jsonl}
MIXED=${MIXED:-data/mixed.jsonl}
XLANG_UR=${XLANG_UR:-data/urdu.jsonl}
XLANG_PA=${XLANG_PA:-data/punjabi.jsonl}
XLANG_BN=${XLANG_BN:-data/bengali.jsonl}
OUT=${OUT:-eval_qwen15b.json}

python mmie.py \
  --model "$MODEL" \
  --forget "$FORGET" --retain "$RETAIN" --mixed "$MIXED" \
  --xlang "$XLANG_UR" "$XLANG_PA" "$XLANG_BN" \
  --ckpt_dir ${CKPT_DIR:-ckpt_lora_final} \
  --train_sae_steps 1500 --sae_k 32 --sae_expansion 8 \
  --sae_gate --sae_gate_alpha 0.5 --sae_gate_topk 64 \
  --semantic_features --semantic_tau 0.0 \
  --dynamic_gate --semantic_dynamic_gate \
  --gate_es_forget_ratio 0.5 --gate_es_mixed_ratio 0.7 --gate_ppl_ratio 1.10 \
  --seeds 42 43 44 \
  --device ${DEVICE:-cuda} \
  --out "$OUT"

echo "[ok] Wrote $OUT"

# Generate plots
python tools/plots_from_report.py "$OUT" || true
