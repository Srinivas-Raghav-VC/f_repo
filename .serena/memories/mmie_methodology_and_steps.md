Research Goal: Causally suppress Hindi semantics (meaning) in a frozen LLM while preserving English capability, not merely blocking Devanagari script.

Hypotheses:
- H1: Mid layers encode language-agnostic semantics; early layers encode form; late layers map to outputs.
- H2: Attenuating mid-layer semantic features reduces script-blind Hindi generation with minimal English degradation.

Data:
- Forget (Hindi Devanagari), Retain (English), Mixed prompts (EN↔HI), Cross-ling (Urdu/Punjabi/Bengali).
- Controls: Hindi-Romanized, English-in-Devanagari, Devanagari gibberish.

Metrics:
- ES_script (original), ES_semantic (romanize outputs + script-blind detection), English PPL + token-KL to base, Redistribution probes, Cross-ling leakage, MIA. BCa 95% CIs.

Layer Selection:
- Debiased linear CKA + Procrustes + optional ANC; pick ~3 mid layers.

Features (SAEs):
- Train per-layer SAEs (or load SAELens via --sae_lens_dir); semantic picker keeps features active for Hindi across scripts and quiet on English-in-Deva/gibberish.

Interventions:
- Unlearning: LoRA q/v (GA or NPO), ReFT residual adapters.
- Steering: SAE-gate (delta-blend; alpha sweep), semantic dynamic gating (no token penalties).
- Baseline erasure: Linear script scrub (INLP/LEACE-lite) per chosen layer.

Decision Gates:
- G1S: ES_semantic(Forget) ≤ 50% base; G2: PPLretain ≤ +10%; G3S: ES_semantic(Mixed) ≤ 70%; G4: no redistribution; G5: no cross-ling leakage; G6: MIA near random.

Exact Steps:
1) Base-only sanity: Run mmie.py to produce eval_report.json; summarize with scripts/summarize_report.py.
2) Layer triage: choose top ~3 layers by CKA/Procrustes/ANC.
3) SAEs: train quick SAEs (k≈32, expansion≈16) or load SAELens.
4) Semantic features: select with invariance criteria.
5) Arms: LoRA/ReFT, SAE-gate (alpha sweep), optional semantic dynamic gating and script scrub.
6) Evaluate: compute ES_script + ES_semantic + PPL/token-KL + safety; write JSON with CIs.
7) Gates: require all G1S…G6 PASS.
8) Optional: dose–response plots; TLens analysis; feature gallery.

Implementation Locations:
- mmie.py (core, semantic ES/gates/device-safe encodes/ANC/scrub)
- backends/sae_lens_loader.py (SAELens)
- analysis_tlens.py (TransformerLens analysis)
- scripts/build_controls.py (controls)
- scripts/summarize_report.py (gate table)
- scripts/run_confirmatory.sh (one-liner)