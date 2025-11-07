# MMIE Methods Card (1‑Page)

Author: Srinivas Raghav • Date: Nov 3, 2025

Problem
- Remove targeted knowledge (e.g., Hindi generation) from an LLM while maintaining utility.
- Resist paraphrase and cross‑script leakage (latent Romanization).

Core Idea (Select → Patch → SAE/GRUN → Audit → FDR)
- Select: rank layers by semantic divergence (CKA/Procrustes/Cos/ANC); shortlist L*.
- Patch: causal validation via activation patching on L*; pick top‑k.
- SAE/GRUN: train compact Top‑K SAEs per layer and deploy inference‑time gates; train lightweight ReFT/GRUN adapters for weight‑space edits.
- Audit: ES, PPL, probes, MIA, ActPert, cross‑ling; apply BH‑FDR across gates.

Why It Works
- Causal selection trims correlational false positives; SAEs target latent features; GRUN localizes edits with gates; audits catch obfuscation.

Defaults (1.5B backbone)
- Selection: min_layer=6, top_k=3, sample_cap=120, max_len=128; stability_select=5.
- SAE: expansion=16, k=32, semantic_tau=0.05–0.10; dynamic gating on LID.
- ReFT: rank=2–4, steps=300–600; L1 on gate when GRUN.
- Stats: BH‑FDR α=0.10 on ES, PPL‑ratio, probes, MIA, ActPert (±AdvES).

Run (HPC, judge‑off)
- `VENV=$(pwd)/venv DISABLE_JUDGE=1 LOG_DIR=$SCR/logs ./run_mmie_hpc.sh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --ckpt_dir "$SCR/ckpt" --no_quantization --auto --auto_plots --device cuda`

Key Numbers (fill before interview)
- ES drop (Forget): ____  (base → edited)
- PPL ratio (Retain): ____  (≤ 1.10)
- ActPert ΔES peak layer(s): ____ (ΔES=____)
- Probe AUC/ACC (Retain): ____ / ____

Limits & Risks
- SAE coverage at small caps; distribution shift; judge latency if enabled; quantization variability.

Pointers
- Report: docs/PhD_Interview_Research_Report.md
- Mermaid: docs/figs/pipeline.mmd → tools/export_mermaid.sh
- Colab auto cell: colab/Colab_Qwen15B_Auto.txt

