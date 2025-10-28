# Codebase Map (Serena Memory)

Purpose: Minimal, falsifiable experiments for semantic unlearning in multilingual LMs. Compare LoRA vs ReFT (+ optional SAE gating and script scrub) under decision gates using script-aware and script-blind (semantic) Extraction Strength.

Repo Overview
- Language: Python; entrypoint `mmie.py`.
- Datasets: JSONL with `{ "text": ... }` in `data/`.
- Outputs: `eval_report.json` (+ base activations `acts_*.npz`), plots in `tools/`.

Key Modules
- `mmie.py` — main CLI/workflow: layer triage (CKA/Procrustes/optional ANC), optional SAE train/load, LoRA/ReFT arms, SAE gating (static/dynamic), optional script scrub, metrics (ES, ES_semantic, PPL, token-KL, redistribution probes, x‑ling leakage, MIA), gating + final decision JSON.
- `lid_ensemble.py` — lightweight LID ensemble (langid + script guard; optional CLD3/fastText). Tweak/add backends here.
- `transliteration_utils.py` — Devanagari→Latin (for semantic ES). Uses `indic_transliteration` if available.
- `backends/sae_lens_loader.py` — import SAELens weights, map to TopK SAE (E[m,d], D[d,m]).
- `tools/` — analysis helpers: `analysis_tlens.py`, `sweep_alpha.py`, `reversibility_harness.py`, `build_training_pairs.py`, plotting.
- `scripts/` — runnable presets: `run_confirmatory.sh`, `run_*model*.sh`, `summarize_report.py`, `build_controls.py`, `migrate_data.sh`.

Entrypoints
- Primary: `python mmie.py --model <hf_id> --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl --out eval_report.json [flags]`
- One‑liners: `scripts/run_tinyllama.sh`, `scripts/run_qwen_1_5b.sh`, `scripts/run_llama3_8b.sh`.
- Summarize: `python scripts/summarize_report.py eval_report.json`.

Core Flags (where to tweak)
- Unlearning arms: `--lora_steps`, `--reft_steps`, `--forget_obj {ga|npo}`.
- SAE: `--train_sae_steps`, `--sae_k`, `--sae_expansion`, `--sae_lens_dir` (load instead of training).
- Gating: `--sae_gate`, `--sae_gate_alpha`, `--sae_gate_topk`, `--dynamic_gate`, `--semantic_dynamic_gate`, `--semantic_features`, `--semantic_tau`.
- Script scrub baseline: `--script_scrub`, `--scrub_k`.
- Layer selection: `--probe_layers`, `--use_anc`.
- Metrics ext: `--report_token_kl`, `--es_romanized`.
- Decision thresholds: `--gate_es_forget_ratio` (G1/G1S), `--gate_es_mixed_ratio` (G3/G3S), `--gate_ppl_ratio` (G2).
- IO/compute: `--sample_cap`, `--max_len`, `--device`, `--ckpt_dir`.

Data Flow (high level)
1) Read JSONL sets (forget/retain/mixed/x‑ling + adversarial).
2) Load tokenizer + base LM; select layers via alignment metrics.
3) Train/load per‑layer SAEs; optionally pick semantic features and attach SAE gate; optional linear script scrub per layer.
4) For each arm (LoRA/ReFT), generate on eval sets; compute ES (script‑aware) and ES_semantic (romanized), PPL/Token‑KL, redistribution probes, x‑ling leakage, MIA.
5) Aggregate across seeds → CIs; apply gates → decisions; write `eval_report.json`.

Hardware Notes
- Runs on CPU for base sanity (set `--sample_cap 100 --max_len 128`).
- Small VRAM works with tiny models; for larger models, reduce steps/caps or use device_map offload.

Common Tasks
- Base sanity: disable training (`--lora_steps 0 --reft_steps 0 --train_sae_steps 0 --device cpu`).
- Semantic SAE gating: enable `--sae_gate --semantic_features --dynamic_gate --semantic_dynamic_gate` and set `--train_sae_steps` or `--sae_lens_dir`.
- Dose–response: `tools/sweep_alpha.py` to vary `--sae_gate_alpha`.
- Reversibility check: `tools/reversibility_harness.py` to tiny‑finetune recovery and compare ES/Token‑KL pre/post.
- TLens analysis: `tools/analysis_tlens.py` on a small model to inspect A/B layer similarity.

Where To Change Things
- LID behavior: `lid_ensemble.py` (`_script_vote`, `_roman_hi_vote`, optional CLD3/fastText hooks).
- Transliteration: `transliteration_utils.py`.
- SAE loading: `backends/sae_lens_loader.py`.
- Gate math / thresholds: bottom of `mmie.py` (`gate(...)`).

Prereqs
- `pip install -r requirements.txt`; `.env` with `HF_TOKEN` (for gated HF models). Optional: `lid.176.bin` for fastText.

Quick Smoke Test (CPU)
```bash
python mmie.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --lora_steps 0 --reft_steps 0 --train_sae_steps 0 \
  --seeds 42 --use_anc --sample_cap 100 --max_len 128 --device cpu \
  --out eval_tiny_base.json
python scripts/summarize_report.py eval_tiny_base.json
```

