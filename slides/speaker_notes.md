# Speaker Notes — SAE Steering (Plain Language)

Use these bullets as your talk track. One breath per bullet.

- Title
  - We target Hindi meaning inside the model, not just the script.
  - We keep English quality and safety steady with hard pass/fail gates.

- Executive Summary
  - Turn down one “meaning band” at mid layers using tiny valves (SAE features).
  - Measure success script-blind so romanization cannot fool us.
  - Proceed only if all gates pass: ES↓, PPL/KL≈, no redistribution/leakage/MIA.

- Why This Problem
  - Blocking scripts is easy to bypass; meaning is what matters.
  - We want a falsifiable, fast protocol usable on small models.

- First Principles
  - Early: form; Mid: meaning; Late: words.
  - Intervene where meaning lives (mid), not where tokens are chosen (late).

- Terminology Decoder
  - Residual stream = the main highway; features = knobs; α = how much to turn down.

- System Pipeline
  - Hooks sit on mid-layer activations; logits processor only schedules α.

- Data Flow
  - Four sets: forget (HI), retain (EN), mixed, cross-ling neighbors (Urdu/Punjabi/Bengali).

- Feynman-Style Picture
  - Conveyor belt with a small valve at the meaning stations.

- Layer Selection
  - Pick mid layers by CKA/Procrustes/ANC: “where EN and HI look most alike”.

- SAE Gate
  - Encode → attenuate selected latents → decode delta → add back.
  - Only a few features; small α first.

- Baselines (LoRA vs ReFT)
  - LoRA edits weights; ReFT edits representations. We compare both.

- Script Scrub (Control)
  - Remove script subspace linearly; good control but cannot remove semantics.

- Controllers
  - Dynamic (can penalize tokens) vs Semantic (script-blind, never penalizes tokens).

- LID & Romanization
  - Romanize the continuation and run LID so scripts can’t cheat the metric.

- Metrics → Gates
  - ES (script-aware + script-blind), PPL/KL, probes, cross-ling ES, MIA.
  - All must pass; otherwise, we do not claim success.

- ES Definition
  - ES = how quickly Hindi appears in the continuation; lower is better after edits.

- Worked Example 1 (Romanization)
  - Bad: “theek hai tum kaise ho?” appears even in Latin. Good: no Hindi semantics.

- Worked Example 2 (Mixed Prompt)
  - Keep the English explanation; downweight/decline Hindi parts.

- Worked Example 3 (Cross-ling)
  - Do not harm Urdu/Punjabi/Bengali; leakage fails the gate.

- Evidence Plots
  - Show ES bars (forget/mixed), PPL bars (retain), cross-ling bars.

- Dose–Response
  - α knob up → ES down; PPL roughly flat. Causal sanity check.

- Probes (Redistribution)
  - Ensure information didn’t just move layers; AUC shouldn’t spike elsewhere.

- Compute & Knobs
  - TinyLlama/Qwen fit on 8–12GB; 8B with 4-bit + offload on 24GB.

- Gate Thresholds
  - G1/G1S ≤50%; G3/G3S ≤70%; G2 ≤1.10; plus G4–G6 safety.

- SAE Memory Math
  - Expansion 4–8 and 2–3 layers keeps RAM/VRAM in check.

- Assumptions & Threats
  - Mid≈meaning (measured), features steerable (TopK + α-sweep), ES proxy (script-blind), no leakage (tests), unlearning≠deletion (recovery probe).

- Limitations
  - SAEs can be polysemantic; synthetic data is tidy; add real hold-out.

- Takeaways
  - Intervene in meaning space; prove script-blind; trust the gate table.

