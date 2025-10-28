#!/usr/bin/env bash
set -euo pipefail

# Llama 3.1 8B Instruct — assumes you have HF_TOKEN with access.
# Recommend QLoRA or device_map=auto for memory; we keep SAE expansion small.
MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
FORGET=${FORGET:-data/forget_hi.jsonl}
RETAIN=${RETAIN:-data/retain_en.jsonl}
MIXED=${MIXED:-data/mixed.jsonl}
XLANG_UR=${XLANG_UR:-data/urdu.jsonl}
XLANG_PA=${XLANG_PA:-data/punjabi.jsonl}
XLANG_BN=${XLANG_BN:-data/bengali.jsonl}
OUT=${OUT:-eval_llama3_8b.json}

# For tight VRAM, you can set: 
#   export TRANSFORMERS_BITS=4  (if you integrate 4-bit loading) 
# The default path uses fp16/bf16 via `mmie.py` device selection.

python mmie.py \
  --model "$MODEL" \
  --forget "$FORGET" --retain "$RETAIN" --mixed "$MIXED" \
  --xlang "$XLANG_UR" "$XLANG_PA" "$XLANG_BN" \
  --ckpt_dir ${CKPT_DIR:-ckpt_lora_final} \
  --train_sae_steps 1500 --sae_k 32 --sae_expansion 4 \
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
