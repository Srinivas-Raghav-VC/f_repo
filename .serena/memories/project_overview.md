Project purpose: Minimal, falsifiable experiments for multilingual unlearning/steering of LLMs (Hindi vs English focus) comparing LoRA and ReFT-style adapters, with metrics: Extraction Strength (LID-based), PPL, probes, cross-lingual leakage, and membership inference.

Tech stack: Python 3; PyTorch; Hugging Face transformers/peft/accelerate; scikit-learn; numpy; tqdm; google-generativeai (optional for dataset/judging); langid for lightweight LID; (planned) fastText/CLD3.

Entrypoints:
- mmie.py — main experiment (training adapters; metrics; report JSON)
- build_mmie_datasets.py — synthetic corpora via Gemini
- lid_ensemble.py — lightweight LID ensemble

Recent upgrades added:
- SAE-based feature gating (inference intervention)
- NPO forget objective (switchable) for LoRA/ReFT
- Token-level KL metric on retain
- Unicode normalization in LID

Rough structure: single-root Python scripts; adapters/hooks live in mmie.py; LID in lid_ensemble.py; sample data JSONL in repo for quick runs.

Conventions: functional style scripts; minimal type hints; prints for logging; JSONL IO; reproducibility via seeds.

Run commands:
- python -m venv .venv && source .venv/bin/activate
- pip install -U pip && pip install -r requirements.txt
- python mmie.py --model <hf_id> --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl --adversarial data/adversarial.jsonl --device cuda --out eval_report.json

Validation commands:
- Optional token-KL: add --report_token_kl
- SAE-gating eval: add --sae_gate --sae_gate_alpha 0.5 --sae_gate_topk 64
- NPO forget objective: --forget_obj npo

Stylistic guidelines: keep patches minimal; avoid global refactors; preserve CLI UX; add flags instead of breaking defaults.

Task completion checklist: run sanity experiment on TinyLlama; verify eval_report.json has ES, PPL, token-KL if requested; scan logs for SAE gate picks. Save changes; avoid committing secrets; keep .env out of VCS.

Notes: Next tasks proposed—analysis pack (entanglement metrics), ES-romanized, LID fusion (fastText/CLD3), Gemini judge integration.