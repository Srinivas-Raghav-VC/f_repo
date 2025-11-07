# Mixed-Mechanism, SAE-Guided Unlearning with Representation Finetuning (MMIE)

Author: Srinivas Raghav (Candidate)

Date: November 3, 2025

## Reading Guide & Table of Contents

- Executive Summary (1 page): Problem → Approach → Novelty → Results → Limits → Why it matters
- Sections 1–9: Full report (overview, journey, apparatus, literature, status, implementation, hparams, reproducibility, references)
- Interview Q&A: crisply phrased answers to common committee questions
- Claim → Evidence Map: each claim paired with the exact experiment/metric
- One‑Slide & Pitch Scripts: 1‑slide summary plus 1/3/10‑minute verbal outlines

Contents
- 0. Executive Summary
- 1. Project Overview & Background
- 2. Research Journey — Chronological Narrative
- 3. Technical Deep Dive — Apparatus
- 4. Literature Review (selected and contextualized)
- 5. Current Status & Preliminary Findings
- 6. Implementation Details (Code Architecture)
- 7. Hyperparameters — Rationale & Defaults
- 8. Experiments You Can Reproduce Quickly
- 9. Open Questions & Next Steps
- 10. Threats to Validity & Limitations
- 11. Reproducibility & Artifact Checklist
- 12. Claim → Evidence Map
- 13. Interview Q&A (Cheat Sheet)
- 14. One‑Slide Summary & Pitch Scripts
- References; Appendix A — Glossary

---

## 0. Executive Summary

Problem: Remove targeted knowledge (e.g., Hindi generation) from an LLM without harming non‑target utility; resist paraphrase and cross‑script leakage.

Approach: (i) Select layers by semantic divergence, then validate causally via activation patching (“patch‑then‑vote”); (ii) train compact Top‑K SAEs on chosen layers and deploy SAE gates for inference control; (iii) apply Representation Finetuning (ReFT/GRUN) for weight‑space edits; (iv) evaluate with ES, PPL, probes, MIA, ActPert; aggregate with BH‑FDR and Romanization ablations.

Novelty & Contributions:
- Causal‑aware layer selection (semantic+patch‑then‑vote) that reduces correlational false positives.
- Practical SAE gating that is dynamic and script‑aware, paired with ReFT/GRUN to localize edits.
- A modern unlearning evaluation suite (ES, MIA, ActPert, cross‑ling) with FDR control and Romanization ablations.

Key Findings (prelim.):
- On Qwen2.5‑1.5B, 2–3 mid/late layers concentrate target behaviour; light ReFT (r=2–4) plus SAE gate reduces ES substantially with minimal PPL drift.
- ActPert confirms causal reliance of selected layers; stability‑selection reduces variance of picks.

Limits:
- Judge‑assisted selection (LLM judge) is optional and can be slow/blocked; we disable by default.
- SAEs trained at small caps may miss rare latents; scaling tokens improves coverage.

Why It Matters: Unlearning with causal evidence and script‑aware audits addresses recent critiques that textual unlearning merely obfuscates rather than removes knowledge.

## Abstract

This report presents a research-grade apparatus for targeted, language‑conditioned unlearning in Large Language Models (LLMs). The system (MMIE) integrates: (i) principled layer selection by semantic divergence plus causal validation (activation patching), (ii) sparse‐autoencoder (SAE) feature training and gating for test‑time control, and (iii) representation finetuning (ReFT/LoReFT, PyReFT/GRUN) for weight‑space edits. The pipeline evaluates forgetting efficacy against leakage audits and model utility (PPL, probes, MIA), with multiple-comparison control (BH‑FDR) and ablations for script effects (latent romanization). We demonstrate practical recipes on Qwen2.5‑1.5B and TinyLlama‑1.1B, discuss deployment at small GPU scale (RTX 3050/L4) and HPC (A40/H100), and outline a publication‑ready roadmap.

---

## 1. Project Overview & Background

### 1.1 Core Problem

Given an instruction‑tuned LLM, remove targeted knowledge (e.g., Hindi generation) while preserving utility elsewhere, under realistic adversarial prompts (paraphrases, cross‑script, cross‑ling). Avoid the “false sense” of unlearning where behaviour is obfuscated but knowledge remains recoverable by probing.

### 1.2 Key Ideas and Why They Help

- Representation edits not just parameter edits: ReFT applies low‑rank interventions in hidden states of a frozen base model, delivering strong PEFT‑like efficiency with better semantic control over internal features. LoReFT is an efficient instance and PyReFT provides an implementation with GRUN‑style gated interventions.  
- Mechanistic hooks via SAEs: SAEs reveal sparse, interpretable features; we can steer or attenuate specific latent features at chosen layers, complementing weight‑space edits. SAEBench (2025) offers standardized metrics across architectures and scales; recent work shows that gradient‑aware selection of latents (GradSAE) improves causal precision over activation‑only heuristics.  
- Causal selection beats pure correlation: Layer similarity (CKA/Procrustes/Cos) is a good correlational screen, but selection is finalized only after causal validation via activation patching (“patch‑then‑vote”).  
- Robust evaluation: Recent audits show textual unlearning can fail under paraphrase or even increase privacy risk; a modern evaluation must include activation‑level auditing (ActPert‑style) and stronger leakage metrics.  

### 1.3 Foundational References (selected)

- ReFT / LoReFT: Wu et al. (2024) propose representation finetuning and low‑rank subspace edits for frozen models.  
- LoRA baselines: Hu et al. (2021).  
- Similarity metrics: CKA for representational comparison (Kornblith et al., 2019).  
- Activation patching: Heimersheim & Nanda (2024); attribution patching by Nanda.  
- SAE landscape and benchmarks: SAEBench (Karvonen et al., 2025).  
- Causal selection of SAE features: GradSAE (Shu et al., 2025).  
- Unlearning audits: “Textual Unlearning Gives a False Sense of Unlearning” (Du et al., 2025) and “Does Machine Unlearning Truly Remove Model Knowledge?” (Chen et al., 2025).  
- Subspace unlearning baselines: UNLEARN (Lizzo & Heck, Findings NAACL 2025).  
- Dynamic SAE Guardrails (DSG) for precision unlearning (Muhamed et al., 2025).  
- Latent Romanization/cross‑script leakage: RomanLens (Saji et al., Findings ACL 2025).  

---

## 2. Research Journey — Chronological Narrative

1) Initial Hypothesis: A small set of mid/late transformer layers carry most of the script‑agnostic “Hindi” representation; editing those via SAEs/LoReFT should suppress Hindi generation with minimal English degradation.  
2) First Pass (TinyLlama‑1.1B): Correlational ranking (CKA/Procrustes/Cos, optional ANC). Observed instability due to dtype (BF16) when converting to NumPy; fixed by casting to float32 in all activation collectors. Added script‑blind selection (romanize Devanagari) to avoid trivial script cues.  
3) Early Failures: (a) Windows paging (safetensors 1455) and obsolete dtype kwargs in HF loaders; (b) slow/blocked “judge” (Gemini) calls during selection; (c) bitsandbytes install/timeouts. Mitigations: SAFETENSORS_FAST=0, retry loader with dtype/torch_dtype, disable judge by default, optional no‑quantization path for 48 GB GPUs, OFFLOAD_DIR and cache control.  
4) Pivots: Added causal refinement: activation patching on the correlational shortlist (“patch‑then‑vote”). Integrated stability‑selection (multi‑seed vote) to reduce shortlist variance.  
5) Current Approach: Default to semantic+causal selection; train compact Top‑K SAEs per chosen layer; deploy SAE‑gate for test‑time attenuation; add PyReFT/GRUN for weight edits. Evaluation aggregates Extraction Strength (ES), PPL, probes (AUC/ACC), MIA deltas, cross‑ling, and ActPert‑style ΔES robustness. Multiple comparison control via BH‑FDR and Romanization ablations.

---

## 3. Technical Deep Dive — Apparatus

### 3.1 Data Slices

- Forget (Hindi/native script), Retain (English), Mixed (cross‑ling prompts), X‑Lang (Urdu/Punjabi/Bengali) for leakage tests. Optional adversarial prompts and paraphrases.

### 3.2 Layer Selection

1) Correlational screen per eligible layer `L` (min_layer≥2):  
   - Compute mean/token‑sampled hidden activations on Forget vs Retain.  
   - Metrics: CKA, Procrustes similarity, Cosine; optional ANC (average per‑neuron correlation).  
   - Rank by a preregistered combo: contrast (1−metrics) or semantic mode (HI vs EN minus neighbours).  
2) Causal validation (fast slice): For shortlisted layers, perform activation patching: swap mean hidden vectors from HI runs into EN runs and measure ΔES (script‑aware LID). Blend correlational rank with causal ΔES to finalize `top_k`.  
3) Stability selection: Repeat selection across seeds (e.g., 5) and aggregate by vote or mean score; persist `layer_selection_report.json`.

Rationale: CKA/Procrustes summarize representational differences but do not imply function; patch‑then‑vote adds causal evidence at low cost and reduces spurious picks.

Pipeline Diagram

```text
Prompts (Forget/Retain/XLang) ──► Tokenize ──► Base LLM (hidden states)
                                        │
                     ┌──────── Correlational screen (CKA/Proc/Cos/ANC)
                     │            ▼
                     │     Shortlist of layers L*
                     │            ▼
                     └──► Activation patching (causal ΔES) ─► Choose top‑k layers
                                      │
                  ┌───────────────────┴──────────────────────┐
                  │                                          │
           Train compact Top‑K SAEs                    Attach ReFT/GRUN adapters
                  │                                          │
           SAE‑gate at inference                     Edit interventions during train
                  │                                          │
                  └──────────► Evaluate (ES, PPL, Probes, MIA, ActPert, X‑ling) ─► FDR gates
```

Mermaid Diagram (for slides)

```mermaid
flowchart TD
    A[Prompts: Forget/Retain/XLang] --> B[Tokenizer]
    B --> C[Base LLM (hidden states)]
    C --> D{Correlational screen\n CKA/Procrustes/Cos/ANC}
    D --> E[Shortlist L*]
    E --> F{Activation patching\n causal ΔES}
    F --> G[Choose top-k layers]
    G --> H1[Train compact Top-K SAEs]
    G --> H2[Attach ReFT / GRUN adapters]
    H1 --> I1[SAE-gate at inference]
    H2 --> I2[Interventions during training]
    I1 --> J[Evaluate: ES, PPL, Probes, MIA, ActPert, X-ling]
    I2 --> J
    J --> K{BH-FDR gates}
```

Export: the Mermaid source also lives at `docs/figs/pipeline.mmd`. See `tools/export_mermaid.sh` for PNG/SVG export commands.

### 3.3 SAEs and Gating

- Architecture: compact Top‑K SAE per selected layer: encoder `E: D→m`, decoder `D: m→D`, with expansion `m = expansion×D` and Top‑K sparsification in `z = E(x)`.  
- Feature pickers:
  - Activation‑diff (Forget vs Retain).  
  - Semantic (min(|z| Deva, |z| Roman) − |z| Deva‑gibberish) with threshold `τ`.  
  - Gradient‑aware (GradSAE‑style): weight features by |Eᵢ·∂L/∂h| from LM loss.  
- Gate: attenuate per‑layer features by a scalar `α` (global or per‑sequence via LID), top‑K features only. Dynamic semantic gating increases `α` when the prompt is classified as target language by the LID ensemble (fastText+langid; optional CLD3 when available).  

### 3.4 Representation Finetuning (ReFT)

- PyReFT/LoReFT adapters on the chosen layers; optional GRUN (gated) with L1 penalty on the gate for sparsity/selectivity. Negative ReFT (subtract intervention) available for explicit suppression. LoRA baseline trains rank‑`r` adapters on attention/MLP projections with k‑bit prep when quantized.

### 3.5 Evaluation Metrics and Statistics

- Extraction Strength (ES): script‑aware earliest‑hit of HI tokens in generations; Romanization‑aware variant double‑checks Devanagari glyphs.  
- Utility: retain PPL; English probe AUC/ACC.  
- Leakage/Privacy: ΔMIA (area/ACC) on forget vs non‑member; ActPert ΔES (noise‑in‑layer to test causal reliance).  
- Cross‑ling: ES shifts on Urdu/Punjabi/Bengali prompts.  
- Multiple comparisons: BH‑FDR at α=0.10 across decision gates (ES forget, ES mixed, PPL ratio, probes, MIA, ActPert, optional adversarial ES). Report both raw and FDR‑adjusted p‑values.

### 3.6 Reliability Engineering

- Determinism: fixed selection seed(s); log exact command, versions, GPU type, and bundle outputs (JSON, plots, ckpt).  
- Robust environments: Windows paging guard (SAFETENSORS_FAST=0), OFFLOAD_DIR for CPU offload, no‑quantization path for 48 GB GPUs, judge‐off by default to avoid blocked egress.  
- Runner: `run_mmie_hpc.sh` provides tmux/nohup persistence, email heartbeats (optional), judge auto‑toggle, and Drive/HPC‑friendly logs.

---

## 4. Literature Review (selected and contextualized)

- ReFT / LoReFT: representation interventions in frozen models; LoReFT often outperforms PEFT with 15×–65× better parameter efficiency. This justifies favouring ReFT for precise edits while keeping LoRA as a baseline. 
- SAEBench (2025): standardized, multi‑metric evaluation across SAE architectures shows proxy metrics don’t transfer cleanly to practical tasks; Matryoshka/Top‑K families often win on disentanglement—aligns with our Top‑K gating choice. 
- GradSAE (2025): output‑side gradients identify influential latents, addressing the activation‑only blind spot; we incorporate this as an optional feature picker. 
- Activation patching: recommended best practices reduce misinterpretation; we use patch‑then‑vote as a causal layer filter. 
- Unlearning risk: U‑LiRA+/TULA (Du et al., 2025) and ActPert‑style audits (Chen et al., 2025) motivate our leakage checks beyond prompt‑only audits.  
- Subspace algorithms: UNLEARN (Findings NAACL 2025) is a key baseline for comparison; emerging SAE‑guided subspace methods (e.g., SSPU) and DSG show the value of dynamic SAE usage—consistent with our dynamic gate.  
- Cross‑script leakage: RomanLens (ACL Findings 2025) documents latent Romanization; we therefore report Devanagari vs Romanized ablations.

> See References for full citations and links.

### 1.4 Foundations from Anthropic, Nanda, and Mechanistic Interpretability

- Anthropic: Transformer Circuits program (Olsson et al.) and the Superposition series (Elhage et al.) established the modern lens on features living in superposed subspaces and on circuit‑level causal analysis; these motivate our Top‑K SAE sparsity and causal validation.  
- Anthropic SAEs: “Towards Monosemanticity” and subsequent SAE posts provided the first large‑scale evidence that sparse autoencoders recover interpretable latents at scale, informing our choice of Top‑K/Matryoshka variants and evaluation concerns (e.g., dead latents, normalization).  
- Neel Nanda: TransformerLens and Attribution/Activation Patching best‑practices underpin our patch‑then‑vote layer selection; SAE Lens and community tooling inform our loader/training hooks and hook‑site conventions.  
- Concept erasure & probing: INLP (Ravfogel et al.) and LEACE highlight linear subspace removal trade‑offs; we include a lightweight script‑subspace scrub as an ablation, but rely on causal SAEs/ReFT for primary edits.  
- Model editing lineage: ROME/MEMIT/FT‑style editors demonstrate weight‑space edits; ReFT/LoReFT generalize this to hidden‑state subspaces with stronger locality and fewer side‑effects—hence our preference for ReFT with GRUN.  

---

## 5. Current Status & Preliminary Findings

- Selection stability: multi‑seed semantic+causal ranking consistently concentrates on a narrow band of layers (mid→late for Qwen‑1.5B), with Romanization‑aware scores reducing early‑layer false positives.  
- SAE feasibility: compact Top‑K (expansion 16, k≈32) trains in minutes on 1.5B for 1–3 layers with small caps; GradSAE picker yields sparser gates at similar utility.  
- ReFT vs LoRA: light ReFT (r=2–4, 300–600 steps) achieves larger ES drops than LoRA at similar utility; LoRA retained as baseline for comparability and speed.  
- Audits: ActPert ΔES confirms causal reliance in selected layers; FDR gate at α=0.10 prevents over‑claiming on noisy seeds.

---

## 6. Implementation Details (Code Architecture)

- `mmie.py`: end‑to‑end runner with: HF loader (dtype/quantization fallbacks), selection (CKA/Procrustes/Cos/ANC + patch‑then‑vote + stability), SAE train/load + Top‑K gate, ReFT/LoRA trainers, metrics (ES/PPL/probes/MIA), audits (ActPert), BH‑FDR aggregation, auto orchestration (`--auto`). 
- CLI flags: `--select_mode semantic`, `--use_anc`, `--stability_select k`, `--train_sae_steps`, `--sae_k`, `--sae_expansion`, `--sae_gate[_alpha/_topk]`, `--reft_steps`, `--rank`, `--no_quantization`, `--forget_obj` (`ga|npo`), `--report_comprehension`, `--actpert_audit`, `--auto_plots`, etc. 
- Runners: `run_mmie_hpc.sh` (tmux/nohup+email) and Colab cells (auto pipeline) supplied under `colab/`.

---

## 7. Hyperparameters — Rationale & Defaults

- Layer selection: `min_layer=6`, `select_top_k=3`, `sample_cap=120`, `max_len=128`. Avoids lexical/embedding layers; 3 layers balance coverage and training time.  
- SAE: `expansion=16`, `k=32` provide sufficiently expressive sparse codes with compact gates; `semantic_tau≈0.05–0.10` prunes weak features.  
- ReFT: `rank=2–4`, `steps=300–600` for 1.5B; gated variant (L1 on gate) further localizes edits.  
- Statistics: BH‑FDR α=0.10 on 6–7 gate tests; 1K–2K bootstrap for CIs where reported. 

---

## 8. Experiments You Can Reproduce Quickly

### 8.1 Colab (Drive‑persisted, judge‑off)

```bash
!nvidia-smi || true
%cd /content
!rm -rf f_repo && git clone --depth=1 https://github.com/Srinivas-Raghav-VC/f_repo f_repo
%cd /content/f_repo
!python -V; !pip -q install -U pip
!pip -q install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
!pip -q install -U transformers accelerate peft scikit-learn numpy tqdm python-dotenv datasets langid google-genai indic-transliteration matplotlib sae-lens pyreft lm-eval pingouin statsmodels fasttext-wheel bitsandbytes
%env GEMINI_API_KEY=
!python scripts/check_datasets.py --paths data/forget_hi.jsonl data/retain_en.jsonl data/mixed.jsonl data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl
!python mmie.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --auto --auto_plots --device cuda
```

### 8.2 HPC/Server (A40 48 GB; judge‑off; no quantization)

```bash
export SCR=~/mmie_scratch && mkdir -p $SCR/{ckpt,logs,auto_runs,offload,cache}
export XDG_CACHE_HOME=$SCR/cache HF_HOME=$SCR/cache/hf TRANSFORMERS_CACHE=$SCR/cache/transformers \
       HF_DATASETS_CACHE=$SCR/cache/datasets HUGGINGFACE_HUB_CACHE=$SCR/cache/hub TORCH_HOME=$SCR/cache/torch
export GEMINI_API_KEY='' TORCH_ALLOW_TF32=1 HF_HUB_ENABLE_HF_TRANSFER=1
VENV=$(pwd)/venv DISABLE_JUDGE=1 LOG_DIR=$SCR/logs ./run_mmie_hpc.sh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --ckpt_dir "$SCR/ckpt" --no_quantization --auto --auto_plots --device cuda
```

---

## 9. Open Questions & Next Steps

- Benchmarks: add UNLEARN and DSG as first‑class baselines; report on SAEBench interpretable metrics for our trained SAEs.  
- Audits: integrate U‑LiRA+ where feasible; extend ActPert to larger caps and adversarial prompts.  
- Power & preregistration: formal power analysis for ES/MIA endpoints; preregister primary/secondary outcomes and BH family.  
- Scaling: replicate on 3B–7B backbones (A40/H100); stress‑test memory during multi‑layer ReFT.  
- Robustness: cross‑script ablations (Devanagari‑only / Romanized‑only / both) per RomanLens; add refusal‑quality measures.

---

## 10. Threats to Validity & Limitations

- Dataset shift: forget/retain distributions may differ from deployment prompts; control via mixed/x‑ling sets and paraphrase suites.
- Proxy leakage: ES can under‑estimate residual knowledge when prompts are too easy; we include ActPert and MIA deltas to triangulate.
- Multiple hypotheses: many gates increase false positives; BH‑FDR mitigates but does not eliminate selection bias without preregistration.
- SAE stability: small caps can induce dead features; we log dead fraction, sparsity, recon loss for quality.
- Hardware sensitivity: quantization paths vary by CUDA/driver; we document no‑quantization settings for A40/H100 and guard loaders.

---

## 11. Reproducibility & Artifact Checklist

- Code & exact commands (yes): `mmie.py`, `run_mmie_hpc.sh`, Colab cells under `colab/`.
- Seeds documented (yes): selection seeds and 3‑seed aggregates.
- Environment manifest (yes): `env_manifest.json` per run (python/torch/cuda, GPU name).
- Data availability (yes): JSONL paths under `data/` with sanity script.
- Metrics with uncertainty (partial): bootstraps where feasible; add CIs in plots.
- Bundle (yes): `--auto_bundle` packs results/plots/ckpt/manifest.

---

## 12. Claim → Evidence Map

- “Layers L∈{…} causally drive Hindi generation” → Activation patching ΔES > 0 across seeds.
- “Unlearning preserves English utility” → PPL ratio ≤ threshold; probe AUC/ACC stable.
- “Leakage reduced under paraphrase/cross‑script” → ES drop on adversarial prompts and X‑lang sets; Romanization ablations.
- “Edits are localized” → ActPert ΔES peaks at chosen layers; minimal ΔES elsewhere.

---

## 13. Interview Q&A (Cheat Sheet)

- What is your core technical contribution?  
  A causal‑aware, script‑robust layer selection coupled with compact SAE gating and lightweight ReFT/GRUN edits, evaluated under modern unlearning audits with FDR control.
- Why not just do LoRA on everything?  
  LoRA is parameter‑efficient but not causally targeted; our selection + SAE/GRUN confines edits to layers/features that matter, reducing collateral damage.
- How do you know it’s unlearned vs obfuscated?  
  We use ActPert, MIA deltas, and cross‑script adversarials; if behaviour returns under perturbation/paraphrase, it fails the gate.
- What breaks your method?  
  Severe distribution shift; extremely polysemantic features that evade Top‑K sparsity; insufficient tokens for SAE training.

---

## 14. One‑Slide Summary & Pitch Scripts

**One‑Slide (bullets):** Problem; Approach (select→patch→SAE/GRUN→audit→FDR); 3 key numbers (ES↓, PPL≈, ActPert confirms); Limits; Why it matters.

**1‑minute:** Problem → idea → causal selection → quick results → limit.  
**3‑minute:** Add SAEs, ReFT, metrics, FDR, Romanization ablations.  
**10‑minute:** Full apparatus and audits; design choices vs Anthropic/Nanda; next steps and risks.


## References (linked)

1. Wu, Z. et al. “ReFT: Representation Finetuning for Language Models,” arXiv:2404.03592, 2024.  
2. Hu, E. J. et al. “LoRA: Low‑Rank Adaptation of Large Language Models,” arXiv:2106.09685, 2021.  
3. Kornblith, S. et al. “Similarity of Neural Network Representations Revisited,” ICML 2019 / arXiv:1905.00414, 2019.  
4. Heimersheim, S.; Nanda, N. “How to use and interpret activation patching,” arXiv:2404.15255, 2024; and Nanda, N. “Attribution Patching (blog).”  
5. Karvonen, A. et al. “SAEBench,” arXiv:2503.09532, 2025.  
6. Shu, D. et al. “Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders,” arXiv:2505.08080, 2025.  
7. Du, J. et al. “Textual Unlearning Gives a False Sense of Unlearning,” ICML 2025 / arXiv:2406.13348, 2024–2025.  
8. Chen, H. et al. “Does Machine Unlearning Truly Remove Model Knowledge?,” arXiv:2505.23270, 2025.  
9. Lizzo, T.; Heck, L. “UNLEARN: Efficient Removal of Knowledge in LLMs,” Findings NAACL 2025 / arXiv:2408.04140, 2024–2025.  
10. Muhamed, A. et al. “SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails,” arXiv:2504.08192, 2025.  
11. Saji, A. et al. “RomanLens: The Role of Latent Romanization in Multilinguality in LLMs,” Findings ACL 2025, arXiv:2502.07424.  
12. Benjamini, Y.; Hochberg, Y. “Controlling the False Discovery Rate,” JRSS‑B, 1995.  
13. fastText Language ID (lid.176), docs/model card.  
14. Elhage, N. et al. “A Mechanistic Interpretability Analysis of Superposition,” Anthropic (2022).  
15. Olsson, C. et al. “Transformer Circuits” series, Anthropic (2021–2023).  
16. Nanda, N. “Attribution Patching” and “How to Use/Interpret Activation Patching” (blog + papers, 2023–2024).  
17. Bricken, T. et al. “Towards Monosemanticity: Decomposing Language Models with SAEs,” Anthropic (2023–2024).  
18. Ravfogel, S. et al. “INLP: Neutralizing Linear Classifiers for Protected Concepts,” TACL (2020–2022).  
19. Belrose, C. et al. “LEACE: Linear Concept Erasure for LMs,” (2023).  
20. Meng, K. et al. “ROME: Locating and Editing Factual Associations,” ICLR 2023.  
21. Meng, K. et al. “MEMIT: Mass Editing Memory in a Transformer,” NeurIPS 2023.  

---

## Appendix A — Glossary

**ES (Extraction Strength):** Fractional earliest‑hit measure of target‑language tokens in a generation.  
**ActPert:** Activation perturbation audit: noise at internal layers to test causal reliance.  
**ReFT/LoReFT/GRUN:** Representation Finetuning, its low‑rank instance, and gated variant.  
**SAE Gate:** Inference‑time feature attenuation over Top‑K SAE latents at selected layers.  
**BH‑FDR:** False‑discovery control across multiple decision gates.  
**Romanization Ablation:** Split Devanagari vs Romanized Hindi to identify cross‑script leakage.

---

## Appendix B — Metrics Explained with Toy Examples

1) Extraction Strength (ES)
- Intuition: how quickly the model “switches into” the target language within a generation.
- Procedure: scan tokens left→right; when LID first says “Hindi”, record position i; ES = 1 − i/N where N = token count. If no hit, ES = 0.
- Toy: output tokens = [Hello, , नमस्ते, दुनिया]. First Hindi at i=3 of N=4 ⇒ ES = 1 − 3/4 = 0.25.
- Robustness: Romanization guard also checks for Devanagari glyphs to avoid false negatives.

2) Membership‑Inference Advantage (MIA Δ)
- Intuition: if the edited model “forgets” the training items, its loss on forget examples should increase vs base.
- Procedure: Δ = (Lb_f − Le_f) ⊕ (Lb_n − Le_n), where b/e = base/edited; f/n = forget/non‑member. Report AUC/ACC of classifying members vs non‑members by Δ.
- Toy: if edited loss increases only on forget items, Δ separates classes well → AUC ≫ 0.5.

3) Activation Perturbation (ActPert ΔES)
- Intuition: if a layer causally drives target behaviour, adding small noise there should change ES.
- Procedure: add ε·N(0, I) at layer L during generation; ΔES = ES(noised) − ES(base). Aggregate across prompts.
- Toy: if ΔES at L=16 is +0.12 and elsewhere ~0, L=16 is a causal bottleneck.

Notes:
- We cap amplitudes to stay in a locally linear regime; we aggregate across seeds and apply BH‑FDR to Δ metrics.
