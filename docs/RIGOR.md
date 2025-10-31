# MMIE Research Rigor (Concise)

## Objective
Causally suppress Hindi semantics (meaning) in a frozen LLM while preserving English capability, without merely blocking Devanagari script.

## Hypotheses
- H1 (Localization): Mid layers encode shared semantics; early layers encode form; late layers map to tokens.
- H2 (Control): Attenuating mid‑layer semantic features reduces script‑blind Hindi generation with minimal English degradation.

## Data & Controls
- Forget: Hindi (Devanagari).
- Retain: English.
- Mixed (EN↔HI) prompts.
- Cross‑ling neighbors: Urdu/Punjabi/Bengali.
- Controls: Hindi‑Romanized; English‑in‑Devanagari; Devanagari gibberish.

## Metrics (with BCa 95% CIs)
- ES_script (original), ES_semantic (romanized continuation + script‑blind detection), English PPL + token‑KL to base, Redistribution probes, Cross‑ling leakage, MIA.

## Layer Selection
- Debiased CKA + Procrustes + optional ANC; choose ~3 mid layers by a combo score.

## Features (SAEs)
- Train per‑layer SAEs or load SAELens SAEs.
- Semantic picker keeps features active for Hindi across scripts and quiet on English‑in‑Deva/gibberish.

## Interventions
- Unlearning: LoRA (q/v), ReFT (residual adapters), with GA or NPO.
- Steering: SAE‑gate (delta‑blend; alpha sweep); semantic dynamic gating (no token penalties).
- Baseline erasure: Linear "script scrub" (INLP/LEACE‑lite subspace projection) per chosen layer.

## Decision Gates (PASS/FAIL)
- G1S: ES_semantic(Forget) ≤ 50% base;
- G2: PPLretain ≤ +10% base;
- G3S: ES_semantic(Mixed) ≤ 70% base;
- G4: no redistribution;
- G5: no cross‑ling leakage;
- G6: MIA near random.

## Exact Steps
1. Base‑only sanity (frozen model):
   ```bash
   python mmie.py \
     --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --forget forget_hi.jsonl --retain retain_en.jsonl --mixed mixed.jsonl \
     --xlang urdu.jsonl punjabi.jsonl bengali.jsonl \
     --lora_steps 0 --reft_steps 0 --train_sae_steps 0 \
     --seeds 42 --use_anc --sample_cap 100 --max_len 128 --device cpu \
     --out eval_tiny_base.json
   python scripts/summarize_report.py eval_tiny_base.json
   ```
2. SAE training or SAELens load:
   - Train quick SAEs: `--train_sae_steps 500 --sae_k 32 --sae_expansion 16`
   - Or load SAELens: `--sae_lens_dir <dir>`
3. Semantic features: add `--semantic_features`.
4. Arms (unlearning & steering):
   - LoRA: `--lora_steps 500 --forget_obj npo` (or `ga`)
   - ReFT: `--reft_steps 500 --forget_obj npo` (or `ga`)
   - SAE‑gate: `--sae_gate --sae_gate_alpha 0.5`
   - Semantic dynamic gating: `--dynamic_gate --semantic_dynamic_gate`
   - Optional: `--script_scrub --scrub_k 1`
5. Evaluate arms → JSON report (means + CIs) and summarize gates.
6. Optional dose‑response: sweep `--sae_gate_alpha`.
7. Optional TLens analysis on a small model: `tools/analysis_tlens.py`.

## Files
- Core: `mmie.py` (semantic ES/gates; device‑safe encodes; scrub; ANC)
- SAELens loader: `backends/sae_lens_loader.py`
- TLens analysis: `tools/analysis_tlens.py`
- Controls: `scripts/build_controls.py`
- Gate table: `scripts/summarize_report.py`
- One‑liner: `scripts/run_confirmatory.sh`

## Defaults for constrained hardware
- Use CPU for end‑to‑end sanity; set `--sample_cap 100`, `--max_len 128`.
- GPU (4 GiB): model may offload to CPU; inputs follow actual device automatically.
