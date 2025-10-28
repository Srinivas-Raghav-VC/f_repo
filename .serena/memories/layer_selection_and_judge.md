# Layer Selection (Semantic + LLM Judge) — 2025-10-28

Purpose: pick intervention layers that carry Hindi semantics (not just script) before SAE training. Adds script-blind, semantic metrics and an optional LLM-judge causal nudge to stabilize picks, especially on small models or small caps.

What changed in code (mmie.py)
- Script-blind selection: romanizes Hindi during selection so scores target meaning, not script. Default: on (`--script_blind_selection/--no_script_blind_selection`).
- Modes: `--select_mode {semantic, contrast, similarity}`. Recommended: `semantic`.
  - semantic score = 0.6·(AUC(HI vs EN) − mean AUC(neighbors vs EN)) + 0.2·(1−CKA) + 0.2·(1−Procrustes)
  - contrast = weighted (1−similarities); similarity = weighted similarities.
- Defaults tuned for this project:
  - `use_anc=True`, `min_layer=2`, `print_layer_scores=True`.
- LLM judge assist (opt‑in): `--judge_assist_selection` with small residual scaling per candidate layer.
  - For top `--judge_pool` layers by metric, attach a temporary `ResidualScaleHook` (scale≈0.85), regenerate small HI/EN slices, and ask judge to score Hindi semantics.
  - Judge delta = (base_HI − scaled_HI) − β·max(0, scaled_EN − base_EN). Blend with metric: `blend = α·metric_norm + (1−α)·judge_norm`.
  - Flags: `--judge_cap 24`, `--judge_pool 6`, `--judge_scale 0.85`, `--judge_alpha 0.5`, `--judge_beta 0.5`, `--judge_model gemini-2.5-flash`, `--judge_timeout 15`.
  - If API/timeout fails, selection falls back to metrics only, with a console note.

Practical defaults (TinyLlama on RTX 3050)
- Selection-only (no training):
  - `--select_mode semantic --min_layer 6 --select_top_k 3 --judge_assist_selection --judge_cap 24 --judge_pool 6 --judge_alpha 0.5 --judge_beta 0.5 --judge_scale 0.85`
- If API slow: lower to `--judge_cap 12 --judge_pool 4 --judge_alpha 0.7 --judge_scale 0.9 --judge_timeout 12` or omit judge flag.

What to look for
- Printed header: `mode=semantic, script_blind=True`.
- Chosen layers pivot to mid/late‑mid; if still shallow, raise `--min_layer` to 8 and increase `--sample_cap` to 120–150.

Why this helps
- Script-blind + neighbor‑contrast focuses on semantics; LLM judge adds a small causal check that breaks metric ties when CKA/Procrustes saturate.

