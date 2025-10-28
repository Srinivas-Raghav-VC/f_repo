# Quick Commands (TinyLlama + Qwen) — 2025-10-28

TinyLlama (RTX 3050, fast path)
- Selection‑only:
  - `python mmie.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl --ckpt_dir ckpt_lora_final --lora_steps 0 --reft_steps 0 --train_sae_steps 0 --sample_cap 120 --max_len 128 --device cuda --select_mode semantic --min_layer 6 --select_top_k 3 --judge_assist_selection --judge_cap 24 --judge_pool 6 --judge_alpha 0.5 --judge_beta 0.5 --judge_scale 0.85 --out selection_only.json`
- SAE train only:
  - `python mmie.py ... --lora_steps 0 --reft_steps 0 --train_sae_steps 1000 --sae_k 32 --sae_expansion 16 --sae_quality_eval --sae_eval_cap 256 --out eval_sae_train.json`
- Gating‑only sanity:
  - `python mmie.py ... --train_sae_steps 0 --lora_steps 0 --reft_steps 0 --sae_gate --sae_gate_alpha 0.5 --sae_gate_topk 64 --semantic_features --semantic_tau 0.05 --dynamic_gate --semantic_dynamic_gate --out eval_sae_gate_only.json`
- Full arms (NPO):
  - `python mmie.py ... --lora_steps 500 --reft_steps 500 --rank 4 --forget_obj npo --sae_gate --dynamic_gate --semantic_dynamic_gate --sae_quality_eval --out eval_full.json`

Qwen 1.5B (Colab/T4 or 12GB+ VRAM; quantize if needed)
- Same as above, change `--model` to `Qwen/Qwen2.5-1.5B-Instruct` (or Qwen2) and consider `--train_sae_steps 1200 --sample_cap 120`.
- If VRAM is tight, install bitsandbytes and set `LOAD_IN_4BIT=1`.

Environment (Windows)
- `SAFETENSORS_FAST=0`; `OFFLOAD_DIR=./offload`; use Python 3.11 venv; install torch with cu121/cu124 wheels; run `python scripts/preflight.py` to confirm CUDA.

