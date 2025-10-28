# Weekly Checklist — Transformers → SAEs → Unlearning (Principles + Practice)

Purpose: Make the knowledge stick by combining (1) by‑hand derivations, (2) code tracing in this repo, (3) tiny runnable experiments, and (4) “explain‑why” reflections. Check off each box; keep deliverables in a `notes/` folder.

Knowledge‑Sticks Toolkit (use every week)
- [ ] Teach‑Back (5 min): explain the week’s idea to a rubber duck (or a colleague) without notes.
- [ ] By‑Hand: derive or compute a small piece (attention weights, CKA, projection) on paper or in a tiny notebook cell.
- [ ] Retrieval: close everything and “blurt” the key equations/definitions; make 3–5 flashcards.
- [ ] Interleave: spend 15 min reviewing last week before starting new material.
- [ ] Micro‑Experiment: one short run or plot that produces a concrete number.
- [ ] Explain‑Why: write 3 bullets on why the method works (not how to run it).

—

## Week 1 — Transformer Core (Attention + Residual Stream)
Reading/Watching
- [ ] Paper: Attention Is All You Need — Abstract & §3 Model.
- [ ] Blog: The Annotated Transformer (skim end‑to‑end once).
- [ ] Video: Karpathy — GPT from scratch (attention + block wiring segments).

By‑Hand
- [ ] Compute scaled dot‑product attention for a 3‑token toy example by hand.
- [ ] Sketch a decoder block; mark exactly where a forward hook would see activations in the residual stream.

Repo Trace
- [ ] Identify where residual activations are edited: `mmie.py` → `SAEGate`, `LinearProjectHook`.

Micro‑Experiment
- [ ] Run a base sanity (no training) to see baseline ES/PPL.
```bash
python mmie.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --lora_steps 0 --reft_steps 0 --train_sae_steps 0 \
  --seeds 42 --sample_cap 100 --max_len 128 --device cpu --out base.json
python scripts/summarize_report.py base.json
```

Explain‑Why (3 bullets)
- [ ] Why attention beats RNNs for long‑range credit assignment.
- [ ] Why pre‑norm residual blocks tolerate small edits gracefully.
- [ ] Why logits processors don’t change the model’s knowledge, only its sampling.

Deliverables
- [ ] `notes/w1_decoder_block.pdf/png`
- [ ] `notes/w1_base_summary.md` (copy gate table + 3 takeaways)

—

## Week 2 — Encoder vs Decoder; Efficiency & Positions
Reading/Watching
- [ ] BERT (masked LM; skim objective), GPT‑3 (decoder‑only; motivation), T5 (task unification).
- [ ] Positional: RoPE & ALiBi (skim intuition).

By‑Hand
- [ ] Implement RoPE for a 1‑head toy attention in ~10 lines (NumPy/PyTorch); confirm rotation behavior.

Repo Trace
- [ ] Walk generation code paths: `generate()` and dynamic gating processors in `mmie.py`.

Micro‑Experiment
- [ ] Confirm that enabling dynamic gating doesn’t break generation formatting.
```bash
python mmie.py ... --dynamic_gate --device cpu --out dyn_gate.json
python scripts/summarize_report.py dyn_gate.json
```

Explain‑Why
- [ ] Why residual‑stream hooks are position‑agnostic (even with RoPE).

Deliverables
- [ ] `notes/w2_rope_toy.ipynb` or `.py`
- [ ] `notes/w2_generation_trace.md`

—

## Week 3 — Representation Similarity (CKA/Procrustes/ANC)
Reading
- [ ] CKA (Kornblith 2019); skim Procrustes (orthogonal alignment).

By‑Hand
- [ ] Compute linear CKA on two small matrices and verify invariances.

Repo Trace
- [ ] Read `select_layers()` and the trio: `linear_cka_debiased`, `procrustes_sim`, `anc_similarity` (in `mmie.py`).

Micro‑Experiment
- [ ] Run a tiny selection pass (`--sample_cap 100`) and record chosen layers and combo scores.

Explain‑Why
- [ ] Why mid‑layers tend to be best for semantic interventions.

Deliverables
- [ ] `notes/w3_layer_selection.md` (paste chosen_layers + your interpretation)

—

## Week 4 — PEFT vs ReFT (Where We Adapt)
Reading
- [ ] LoRA (idea, rank/params) and ReFT (hidden‑state interventions on a frozen base).

By‑Hand
- [ ] Derive how a rank‑r adapter adds a low‑rank update to the residual.

Repo Trace
- [ ] Compare `train_lora()` vs `train_reft()`; note where edits are applied (weights vs hidden states).

Micro‑Experiment
- [ ] Train 100 steps of LoRA and 100 steps of ReFT on CPU/GPU w/ small caps; compare PPL drift.

Explain‑Why
- [ ] Two bullets on why representation edits can target behavior with less collateral drift than weight‑space edits.

Deliverables
- [ ] `notes/w4_lora_reft_compare.md` (table: ES/PPL deltas, quick commentary)

—

## Week 5 — Sparse Autoencoders (SAEs) & Gating
Reading/Practice
- [ ] SAEs for LMs; Scaling/Evaluating SAEs; skim SAELens tutorial.

By‑Hand
- [ ] Write a 10–20 line Top‑K encoder/decoder and show how top‑k masking changes reconstruction.

Repo Trace
- [ ] `TopKSAE`, `train_sae()`, `pick_semantic_sae_features()`; understand the semantic score: `min(|z| HI‑Deva, HI‑Roman) − |z| Deva‑gib`.

Micro‑Experiment
- [ ] Train an SAE for one chosen layer (`--train_sae_steps 1000`), run semantic picker, and enable SAE‑gate (`--sae_gate`). Record ES change.

Explain‑Why
- [ ] Why feature‑targeted gating can reduce unwanted semantics with smaller PPL hit compared to broad penalties.

Deliverables
- [ ] `notes/w5_sae_gate_results.md` (before/after ES and PPL, with 2–3 qualitative generations)

—

## Week 6 — Linear Concept Erasure (INLP/LEACE‑Lite Baseline)
Reading
- [ ] INLP and LEACE (closed‑form erasure).

By‑Hand
- [ ] Derive H ← H − H P with P = W (WᵀW)⁻¹ Wᵀ; explain ridge/pinv for stability.

Repo Trace
- [ ] `LinearProjectHook` and `learn_script_subspace()`; understand logistic weight→subspace.

Micro‑Experiment
- [ ] Apply `--script_scrub --scrub_k 1` and compare ES/PPL vs SAE‑gate.

Explain‑Why
- [ ] When a linear scrub suffices, and when polysemanticity limits linear methods.

Deliverables
- [ ] `notes/w6_scrub_vs_gate.md`

—

## Week 7 — Unlearning: NPO, MIA, Leakage
Reading
- [ ] NPO (stability vs GA), RWKU (MIA/leakage probes).

By‑Hand
- [ ] Write the NPO loss from memory and annotate each term’s role.

Repo Trace
- [ ] `npo_loss()`, `mia_loss()`, mixed/adversarial datasets.

Micro‑Experiment
- [ ] Train LoRA/ReFT with `--forget_obj npo` (short run) and compare to GA.
- [ ] Summarize gates via `scripts/summarize_report.py`.

Explain‑Why
- [ ] Why NPO avoids catastrophic collapse on retain compared to naive GA.

Deliverables
- [ ] `notes/w7_unlearning_summary.md` (gates table + 3 takeaways)

—

## Week 8 — Robustness & Reversibility; Accuracy Add‑On
Reading
- [ ] “Unlearning isn’t deletion” (repr‑level reversibility); quantization‑recovery paper.

By‑Hand
- [ ] Outline a 3‑step protocol to distinguish suppression vs erasure.

Repo Trace
- [ ] `tools/reversibility_harness.py` (tiny recovery finetune); plan an int8/int4 ES re‑eval.

Micro‑Experiment
- [ ] Run the harness (50 steps) and compute ES pre/post; if you have a quantizer, re‑measure ES after int8.
- [ ] (Optional) Add a tiny correctness‑scored HI→EN QA set and log accuracy alongside ES.

Explain‑Why
- [ ] How reversibility and post‑quantization checks expose obfuscation.

Deliverables
- [ ] `notes/w8_reversibility_report.md` (copy JSON + interpretation)
- [ ] (Optional) `notes/w8_accuracy_addon.md`

—

## Capstone (1 day) — Synthesis & Design
- [ ] Write a 1‑page “How I’d improve the pipeline” (threshold flag choices, CI‑aware gates, correctness scoring, robustness).
- [ ] Record a 5‑minute screen‑share explaining one intervention (SAE‑gate or ReFT) and why it works.

—

Appendix — “Explain‑Why” Prompts (use any week)
- If this method works, what would I expect to see in ES, PPL, and probes? If it fails, what would I see instead?
- Why is the residual stream the right locus of control for targeted interventions?
- When does linear subspace removal approximate a true causal ablation, and when does it not?
- Why can preference‑based forgetting (NPO) be more stable than gradient ascent on the forget loss?
