# Pipeline Review (MMIE + ReFT/LoRA + SAE) — 2025-10-28

Objective: Causally suppress Hindi semantics (script‑blind) while preserving English and avoiding side‑effects. Multi‑gate decision: ES_semantic, PPL/token‑KL, mixed, redistribution, cross‑ling leakage, MIA.

End‑to‑end Steps
- Data & controls: forget (HI Deva), retain (EN), mixed, neighbors (UR/PA/BN), plus controls (HI‑roman, EN‑in‑Deva, Deva gibberish).
- Layer selection (script‑blind): `--select_mode semantic` recommended; default `use_anc=True`, `min_layer=2`, `print_layer_scores=True`.
  - Optional `--judge_assist_selection`: residual scaling + LLM judge to refine candidates.
- SAE stage:
  - Train or load SAEs per chosen layer. Typical: `--train_sae_steps 800–1200`, `--sae_k 32`, `--sae_expansion 8–16`.
  - Save per‑layer: `sae_layer{L}.pt` under `--ckpt_dir`.
  - Quality proxies (no extra deps): `--sae_quality_eval --sae_eval_cap 256` → recon_mse, sparsity_l0, dead_fraction. Export E/D for SAEBench: `tools/saebench_adapter.py`.
- Interventions:
  - SAE‑gate (primary): `--sae_gate --sae_gate_alpha 0.5` (+ `--dynamic_gate --semantic_dynamic_gate`); semantic feature picker: `--semantic_features --semantic_tau 0.05–0.1`; top‑K features: `--sae_gate_topk 64`.
  - ReFT (representation adapters): `--reft_steps 400–600`, `--rank 4–8`, `--forget_obj npo` preferred.
  - LoRA (baseline arm): `--lora_steps 400–600`, `--forget_obj npo` preferred.
  - Linear “script scrub” baseline: `--script_scrub --scrub_k 1` (ablative; not a gate input).
- Evaluation & gates:
  - ES (script‑aware) and ES_semantic (romanized); PPL + optional token‑KL; redistribution probes; x‑ling leakage; MIA near random.
  - Default gate thresholds: ES_forget ≤ 0.5×base; ES_mixed ≤ 0.7×base; PPL_retain ≤ 1.10×base; probes ≤ ~0.55 AUC; neighbors ΔES ≤ 0.10; MIA |AUC−0.5|,|ACC−0.5| ≤ 0.05.
  - Report: JSON `--out` with base + arm summaries, gates, and decisions; `scripts/summarize_report.py` prints table.

Checkpoints & Resuming
- Saved to `--ckpt_dir` (recommend per‑run folder):
  - `sae_layer{L}.pt`, `lora_adapters.pt`, `reft_adapters.pt`, activations `acts_*.npz`.
  - Resume: set steps=0 to load adapters (`--lora_steps 0`, `--reft_steps 0`) and reuse SAEs.

Windows/CUDA Notes
- Avoid paging file 1455: `SAFETENSORS_FAST=0`; set `OFFLOAD_DIR` to a writable folder; optional 4‑bit with bitsandbytes.
- Prefer Python 3.11 venv; install torch CUDA wheels via cu121/cu124 indexes; use `scripts/preflight.py` to confirm CUDA availability.

Sanity & Rigor Add‑ons
- Dose–response: `tools/sweep_alpha.py` (monotone ES_semantic↓ with α↑).
- Invert‑gate (optional): temporarily raise same features (α<0) to see ES_semantic↑ (causal sanity).
- LLM judge corroboration (post‑hoc): not a gate; confirms semantic drop on the same prompts.
- SAEBench slice: run on the exact SAEs used for gating to validate feature quality/utility.

Quick Commands (TinyLlama)
- Selection‑only: see `layer_selection_and_judge.md`.
- Train SAEs only: add `--train_sae_steps 1000 --lora_steps 0 --reft_steps 0`.
- Gating‑only check: `--train_sae_steps 0 --lora_steps 0 --reft_steps 0 --sae_gate --dynamic_gate --semantic_dynamic_gate`.
- Full arms: keep NPO, modest steps, and check gates with `scripts/summarize_report.py`.

