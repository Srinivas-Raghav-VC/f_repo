Environment
- python -m venv .venv && source .venv/bin/activate
- pip install -U pip && pip install -r requirements.txt

Datasets (optional, Gemini)
- export GEMINI_API_KEY=...
- python build_mmie_datasets.py --out_dir data

Baseline run
- python mmie.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl \
  --mixed data/mixed.jsonl --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --adversarial data/adversarial.jsonl --device cuda --out eval_report.json

SAE gating / token-KL / NPO
- python mmie.py ... --sae_gate --sae_gate_alpha 0.5 --sae_gate_topk 64 --report_token_kl --forget_obj npo

Quick LID check
- python - << 'PY'\nfrom lid_ensemble import LIDEnsemble,LIDConfig\nprint(LIDEnsemble(LIDConfig()).infer('यह एक उदाहरण वाक्य है'))\nprint(LIDEnsemble(LIDConfig()).infer('yeh ek udaharan vakya hai'))\nPY

Code search
- rg "SAEGate|token_kl_to_base|npo_loss|generate\(" -n

Analysis (to be added)
- python analysis.py --model <id> --forget ... --retain ... (Jaccard, principal angles, gradient alignment)
