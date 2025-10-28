# MMIE: Minimal, Maximally-Informative Experiment (Hindi ↔ English)

This repo runs a fast, falsifiable test of multilingual unlearning:
- Compare **LoRA baseline** vs **ReFT+SAE**.
- Evaluate with **Extraction Strength (ES)** (using an **LID ensemble**),
  now in two forms: (a) script-aware ES (original), and (b) script-blind
  semantic ES (romanize outputs, no script guard). Also report **Perplexity (PPL)**
  on English retain, **Mixed queries**, **Redistribution probes**, **Cross-lingual leakage**, **MIA**.

Key additions for semantic rigor:
- `--semantic_features` selects SAE features that fire for Hindi across scripts and
   stay quiet for Devanagari gibberish.
- `--semantic_dynamic_gate` enables script-blind dynamic gating (no token penalties;
   alpha scheduled by semantic LID on the continuation).
- Reports include `es_semantic` fields and gates (`G1S_*`, `G3S_*`) that must pass
   alongside the original gates.

## Quickstart

### 1) Python & deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
### Useful one-liners

- Build NPO pairs + adversarial (Gemini):
  `python tools/build_training_pairs.py --forget data/forget_hi.jsonl --target_lang "Hindi (Devanagari)" --out_pairs data/pairs.jsonl --out_adv data/adv.jsonl`

- Reversibility check (tiny recovery finetune):
  `python tools/reversibility_harness.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --steps 50 --out reversibility_report.json`

### Repo layout (after light reorg)

- `mmie.py` — main CLI/logic (unchanged entrypoint)
- `tools/` — optional helpers and analysis scripts (moved here)
- `scripts/` — runnable shell helpers (added per‑model presets)
- `data/` — place dataset JSONL files here (`scripts/migrate_data.sh` moves any root-level .jsonl)
- `backends/` — light SAELens loader

### 2) Confirmatory run (semantic-aware)

```bash
python mmie.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --train_sae_steps 2000 --sae_k 32 --sae_gate --sae_gate_alpha 0.5 \
  --semantic_features --semantic_tau 0.0 \
  --dynamic_gate --semantic_dynamic_gate \
  --gate_es_forget_ratio 0.5 --gate_es_mixed_ratio 0.7 --gate_ppl_ratio 1.10 \
  --device cuda --out eval_report.json
```

The JSON report includes script-aware and script-blind ES, bootstrap confidence intervals,
and pass/fail gates for both. A decision requires both semantic and original gates.

### Optional: Dose–response figure for slides

```bash
python tools/sweep_alpha.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl \
  --alphas 0.2 0.5 0.8 --ckpt_dir ckpt_lora_final --device cuda
```
Outputs `sweep_alpha_results.json/.csv` and, if matplotlib is installed, `sweep_alpha_results.png`.

## Optional: SAE quality and SAEBench integration

- Lightweight SAE quality proxies (no extra deps):
  - Add `--sae_quality_eval` (with optional `--sae_eval_cap`) to include per‑layer SAE metrics in `eval_report.json`:
    - `recon_mse` (mean reconstruction MSE), `sparsity_l0` (fraction of near‑zero activations), `dead_fraction` (features never active above 1e-3).
- Export SAEs for SAEBench (if you want standardized evals):
  - `python tools/saebench_adapter.py --ckpt_dir ckpt_lora_final --layers 8 16 24 --out_dir sae_export`
  - Install SAEBench separately (`pip install sae-bench`) and point it at the exported `*_ED.pt` files.

## Windows CUDA note

If you see `OSError 1455 (paging file)`, set these before running:

```powershell
$env:SAFETENSORS_FAST='0'
mkdir offload; $env:OFFLOAD_DIR=(Resolve-Path .\offload).Path
```
