# From Transformers to Unlearning — A Principles‑First Study Guide

Audience: Practitioners who want a clear, efficient, and intuitive path from “Attention Is All You Need” to modern representation‑level edits (ReFT), sparse autoencoders (SAEs), linear erasure (INLP/LEACE), and unlearning (NPO, robustness, reversibility). Optimized for building mental models you can apply in this repo.

Goals
- Efficiency: Read only what compounds understanding. Skim strategically; practice where it matters.
- Intuition: Build vivid mental pictures of the residual stream, attention, and interventions.
- Principles: Derive/verify key equations (attention, CKA, nullspace projection, preference losses).

How to use this guide
- Pick a track (10h Crash, 30h Core, or 60h Deep). Each module has: Why it matters here → Read/Watch/Blog → Exercises → Principles.
- Tie every module to this repo: concrete code anchors and optional run commands.
- Keep a running “One‑pager” (template below). Rebuild it as your mental model evolves.

Repo anchors (what you will connect to)
- `mmie.py`: `select_layers()`, `TopKSAE`, `SAEGate`, `LinearProjectHook`, `train_lora()`, `train_reft()`, `generate*()`, `DynamicGatingLogitsProcessor`, `pick_*_sae_features()`, metrics (ES/PPL/Probes/MIA), gates/summary.
- `tools/reversibility_harness.py`: Tiny recovery finetune + pre/post metrics.
- `tools/build_mmie_datasets.py`, `scripts/build_controls.py`: Mixed/adversarial prompts and semantic controls.
- `lid_ensemble.py`: Script + language detection ensemble.
- `scripts/summarize_report.py`: Gate table from `eval_report.json`.

Quick Map of Modules
1) Transformer Core → 2) Encoder/Decoder & Finetuning → 3) Efficiency/Positions → 4) Representation Similarity (CKA/Procrustes/ANC) → 5) PEFT→ReFT → 6) Preference Learning Context → 7) SAEs & Steering → 8) Linear Concept Erasure → 9) Unlearning (NPO, robustness, reversibility) → 10) Scaling Intuition.

—

## Module 1 — Transformer Core (Attention Is All You Need)
Why it matters here
- All interventions in this repo hook into the residual stream of Transformer blocks. Understanding self‑attention + residual pathways makes SAE gating and projection scrubs intuitive.

Read (primary)
- Vaswani et al., “Attention Is All You Need” (2017): Abstract; §3 Model; Figs 1–2.

Watch
- Karpathy — “Let’s build GPT from scratch” (self‑attention and block wiring segments).

Blogs (intuitive and underrated)
- The Annotated Transformer (Harvard NLP) — code walkthrough you can step through.
- Peter Bloem — Transformers from Scratch (math + design tradeoffs).

Exercises
- Draw a single decoder block and mark: MH‑Attention → Add → Norm → MLP → Add → Norm. Annotate exactly where a forward hook would see activations and where a projector would modify them.
- Explain (in three bullets) why pre‑norm residual blocks make gentle interventions easier to stabilize.

Principles
- Scaled dot‑product attention, multi‑head parallel subspaces, residual mixing. Memorize the residual‑stream mental picture.

Repo tie‑ins
- Hooks that edit hidden states: `SAEGate`, `LinearProjectHook` (file: `mmie.py`).

—

## Module 2 — Encoder vs Decoder; Pretrain → Finetune
Why it matters here
- You evaluate generation (decoder‑only) with ES/PPL; LoRA/ReFT arms adapt a decoder‑only LM.

Read (skim)
- BERT (masked‑LM encoders): pretraining objective.
- GPT‑3 (decoder‑only): few‑shot and scaling intuition.
- T5 (text‑to‑text): unifying everything as a text transformation.

Exercise
- Contrast encoder vs decoder attention masks and why decoder‑only is natural for ES generation.

Repo tie‑ins
- Generation paths: `generate()`, `generate_with_*gating()` (file: `mmie.py`).

—

## Module 3 — Efficiency and Positional Encodings
Why it matters here
- Confirms that optimized kernels (e.g., FlashAttention) don’t change logits semantics your processors rely on; RoPE/ALiBi reasoning about length generalization.

Read (skim for intuition)
- FlashAttention (exact, IO‑aware attention).
- RoPE, ALiBi (positional schemes for longer contexts).

Exercise
- Given RoPE, argue why a residual‑stream hook remains position‑agnostic: what does the hook “see” vs what RoPE changes?

Repo tie‑ins
- Logits processors: `DynamicGatingLogitsProcessor` schedules SAE alpha based on LID risk.

—

## Module 4 — Representation Similarity (Layer Picking)
Why it matters here
- `select_layers()` chooses mid‑layers via CKA/Procrustes/ANC — the measured “semantic overlap” between EN/HI drives where you intervene.

Read (primary)
- Kornblith et al., CKA (Similarity of NN Representations Revisited, 2019).
- (Concept) Orthogonal Procrustes alignment.

Exercises
- Compute linear CKA on toy matrices (or two adjacent model layers) and interpret scores.
- Explain how combining CKA + Procrustes (+ optional ANC) reduces sensitivity to sample noise.

Repo tie‑ins
- `select_layers()`, `linear_cka_debiased()`, `procrustes_sim()`, `anc_similarity()` (file: `mmie.py`).

—

## Module 5 — PEFT → ReFT (How we adapt models)
Why it matters here
- This repo compares LoRA (weight‑adapter) vs ReFT‑style hidden‑state interventions.

Read (primary)
- LoRA: Low‑Rank Adapters for LMs (idea + parameter counts).
- ReFT: Representation Finetuning (2024): edits in hidden space, base frozen.

Exercises
- List two pros/cons of editing weights (LoRA) vs editing representations (ReFT) for targeted forgetting and redistribution risk.

Repo tie‑ins
- `train_lora()` / `resume_lora()`, `train_reft()` (file: `mmie.py`).

—

## Module 6 — Preference Learning Context
Why it matters here
- NPO is a “negative preference” cousin; knowing RLHF/DPO gives context for stability and collapse dynamics.

Read (skim)
- InstructGPT (RLHF pipeline) — why preferences help.
- DPO (Direct Preference Optimization) — offline preference finetuning.

Repo tie‑ins
- `npo_loss()` and `forget_obj` flag (file: `mmie.py`).

—

## Module 7 — Sparse Autoencoders (SAEs) and Steering
Why it matters here
- You train/load SAEs, then select semantic features and gate them on the fly. Understanding feature geometry is key.

Read (primary)
- “Sparse Autoencoders find interpretable features in LMs.”
- “Scaling and evaluating sparse autoencoders” (2024).
- (Optional) SAEBench (2025) — metrics and pitfalls.

Practice/Blogs
- SAELens tutorials for training/loading/visualizing features.

Exercises
- Describe Top‑K SAE vs Gated/JumpReLU variants; when does Top‑K help steering?
- Why does the semantic picker take `min(|z| on HI‑Deva, HI‑Roman) − |z| on Deva‑gibberish`?

Repo tie‑ins
- `TopKSAE`, `train_sae()`, `pick_semantic_sae_features()`, `SAEGate` (file: `mmie.py`), `backends/sae_lens_loader.py`.

—

## Module 8 — Linear Concept Erasure (Controls/Baselines)
Why it matters here
- Your projector is an INLP/LEACE‑lite control: scrub script‑only directions.

Read (primary)
- INLP (Iterative Nullspace Projection).
- LEACE (Closed‑form linear erasure).

Exercise
- Derive the projection: H ← H − H P with P = W (WᵀW)⁻¹ Wᵀ; explain ridge/pinv stabilization.

Repo tie‑ins
- `LinearProjectHook` and `learn_script_subspace()` (file: `mmie.py`).

—

## Module 9 — Unlearning & Robustness (NPO, MIA, Leakage, Reversibility)
Why it matters here
- This repo evaluates suppression via ES (script‑aware/semantic), PPL/token‑KL, redistribution probes, cross‑ling leakage, MIA, and a reversibility harness.

Read (primary)
- NPO: Negative Preference Optimization (stability vs GA collapse).
- “Unlearning isn’t deletion” (reversibility; representation‑level analysis).
- Quantization can recover “forgotten” info (int8/int4 tests).
- RWKU benchmark (MIA and adversarial probes).

Exercises
- Design a tiny reversibility experiment: LoRA recovery on forget only for 50 steps; measure ES pre/post and token‑KL on retain.
- Propose a post‑quantization ES check (int8 vs int4) and thresholds to flag failure.

Repo tie‑ins
- Metrics: `extraction_strength()`, `perplexity()`, `probes_auc()`, `mia_loss()`; Gates: `gate()`; Harness: `tools/reversibility_harness.py`.

—

## Module 10 — Scaling Intuition
Why it matters here
- Helps choose sample caps, sequence lengths, and read results realistically.

Read (skim)
- Scaling laws (Kaplan et al., 2020) and “Chinchilla” (token‑optimal training).

—

## Study Tracks

10‑Hour Crash (breadth first)
- M1 (AIAUY + Annotated Transformer skim) — 2h
- M4 (CKA/Procrustes) — 1.5h
- M5 (LoRA vs ReFT) — 1.5h
- M7 (SAE concept + SAELens tour) — 2h
- M9 (NPO + reversibility + robustness headlines) — 2h
- Tie to repo: read code anchors in each section — 1h

30‑Hour Core Mastery
- M1–M3 deeper; M4–M9 with exercises; run one tiny experiment (see below).

60‑Hour Deep Dive
- Add SAEBench/advanced SAE variants; implement a correctness‑scored cross‑ling QA; add quantization robustness script and write a short report.

—

## “One‑Pager” Template (rewrite each module in your words)
1) What problem does this module solve?
2) The mental model (sketch; 2–4 bullet points).
3) The core equation/mechanism (write it once from memory).
4) Where it lives in this repo (file + function/class names).
5) One small experiment to confirm I understand it.

—

## Optional Tiny Experiment (1 evening)
Goal: Produce a first `eval_report.json`, see gates, and relate them to your reading.

1) Base sanity (no training)
```
python mmie.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --lora_steps 0 --reft_steps 0 --train_sae_steps 0 \
  --seeds 42 --sample_cap 100 --max_len 128 --device cpu \
  --out base.json
python scripts/summarize_report.py base.json
```

2) Semantic‑aware arms
```
python mmie.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --train_sae_steps 2000 --sae_k 32 --sae_gate --sae_gate_alpha 0.5 \
  --semantic_features --semantic_tau 0.0 \
  --dynamic_gate --semantic_dynamic_gate \
  --gate_es_forget_ratio 0.5 --gate_es_mixed_ratio 0.7 --gate_ppl_ratio 1.10 \
  --device cuda --out eval_report.json
python scripts/summarize_report.py eval_report.json
```

3) Reversibility probe
```
python tools/reversibility_harness.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget forget_hi.jsonl --retain retain_en.jsonl \
  --steps 50 --out reversibility_report.json
```

Interpretation checklist
- Did G1/G1S (ES reductions) pass without breaking G2 (PPL)?
- Mixed prompts (G3/G3S) reduction without cross‑ling leakage (G5)?
- Redistribution flag off and MIA near random (G4/G6)?
- Reversibility: ES_post ≫ ES_pre implies obfuscation; small rebound implies robust edits.

—

## Curated, High‑Signal Resources (papers/blogs/videos)
Papers (seminal and modern)
- Attention Is All You Need (2017)
- BERT (2018), GPT‑3 (2020), T5 (2020)
- FlashAttention (2022); RoPE/ALiBi (positional)
- CKA (2019)
- LoRA (2021); ReFT (2024)
- SAEs (2023), Scaling/Evaluating SAEs (2024), SAEBench (2025)
- INLP (2020), LEACE (2023)
- NPO (2024), SimNPO (2024), RWKU (2024), Unlearning ≠ Deletion (2025), Quantization‑Recovery (2024/25)

Underrated/great blogs
- The Annotated Transformer (Harvard NLP)
- Peter Bloem — Transformers from Scratch
- VectorFold — Transformers from Scratch (NumPy‑first)
- Jay Alammar — Illustrated series
- Neel Nanda — Mechanistic interpretability quickstarts/readings

Videos
- Karpathy — Build GPT from scratch
- CS224n Transformer/Attention lectures (Manning)
- Yannic Kilcher — AIAUY walkthrough
- 3Blue1Brown — What is a GPT?

Search recipes to find niche gems
- General: `[paper name] + "from scratch" OR "intuition" OR "explained" site:substack.com OR site:medium.com OR site:github.io`
- CKA/Procrustes code: `CKA Kornblith implementation from scratch blog`
- SAEs: `sparse autoencoder interpretability TopK tutorial SAELens`
- Unlearning: `negative preference optimization intuition blog`, `LLM unlearning reversibility quantization`
- Linear erasure: `INLP nullspace projection tutorial`, `LEACE closed form explained`

—

## Glossary (mental models in one line)
- Residual stream: The highway that all blocks add into; hooks edit it.
- Self‑attention: Content‑based mixing of token representations via QKᵀV.
- CKA: Centered kernel alignment — robust similarity of representations.
- LoRA: Low‑rank adapters on weights (parameter‑efficient finetune).
- ReFT: Interventions in hidden states (frozen base, edit the flow).
- SAE: Sparse bottleneck decoder reconstructing activations to expose features.
- INLP/LEACE: Linear subspace removal to erase a concept from representations.
- NPO: Negative preference objective to push away targeted behavior without collapse.
- ES: Extraction Strength — how quickly target language/script appears in continuation.

—

## Your “Do‑Better” Checklist (apply learning to this repo)
1) Add correctness‑scored cross‑ling QA (Hindi→English factual answers) alongside ES.
2) Save and interpret reversibility_report.json for each arm.
3) Add post‑quantization ES checks (int8/int4); flag failures in a table.
4) Make gate thresholds part of the CLI (already added: `--gate_*`); add CI‑aware alternatives if needed.
5) SAE feature stability test: re‑run picker on paraphrases/checkpoints and report overlap.

—

## One‑Page Notes Template (copy per module)
```
Module: ______________________________
What problem does it solve?
Mental model (3 bullets):
Core equation / mechanic:
Repo anchor (file:function):
One micro‑experiment:
What I still don’t get (1–2 questions):
```
