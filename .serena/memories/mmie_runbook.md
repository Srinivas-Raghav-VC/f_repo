Base-only sanity (CPU; robust):
source .venv/bin/activate
export HF_TOKEN="<your_token>"
python mmie.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget forget_hi.jsonl --retain retain_en.jsonl --mixed mixed.jsonl \
  --xlang urdu.jsonl punjabi.jsonl bengali.jsonl \
  --lora_steps 0 --reft_steps 0 --train_sae_steps 0 \
  --seeds 42 --use_anc --sample_cap 100 --max_len 128 --device cpu \
  --out eval_tiny_base.json
python scripts/summarize_report.py eval_tiny_base.json

Quick semantic SAE steering (fast):
python mmie.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget forget_hi.jsonl --retain retain_en.jsonl --mixed mixed.jsonl \
  --xlang urdu.jsonl punjabi.jsonl bengali.jsonl \
  --train_sae_steps 500 --sae_k 32 --sae_expansion 16 \
  --sae_gate --sae_gate_alpha 0.5 --semantic_features \
  --dynamic_gate --semantic_dynamic_gate \
  --seeds 42 --use_anc --sample_cap 100 --max_len 128 --device cpu \
  --out eval_tiny_sem.json
python scripts/summarize_report.py eval_tiny_sem.json

Optional baselines/tools:
- Linear script scrub: add --script_scrub --scrub_k 1
- SAELens: --sae_lens_dir <dir> (instead of training)
- TLens analysis (small model): python analysis_tlens.py --model EleutherAI/pythia-70m --a forget_hi.jsonl --b retain_en.jsonl --out tlens_report.json

Notes:
- If GPU (4 GiB): you can try --device cuda; model may offload to CPU; inputs are auto-aligned now.
- If RAM-constrained: lower --sample_cap to 80 and close background apps.
