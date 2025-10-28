Tools in this folder are optional helpers and analyses. They import the main logic from the top-level `mmie.py` without changing the CLI.

- analysis.py — diagnostics (SAE overlap, principal angles, gradient alignment)
- analysis_tlens.py — optional TransformerLens analysis
- reversibility_harness.py — tiny recovery finetune to test reversibility
- build_mmie_datasets.py — synthetic corpora builder (Gemini)
- build_training_pairs.py — preference pairs + adversarial prompts (Gemini)
- gemini_judge.py — LLM-as-a-judge semantic language scoring
- throughput_bench.py — quick model throughput benchmark

Run examples (from repo root):
- python tools/analysis.py --model <id> --forget forget_hi.jsonl --retain retain_en.jsonl
- python tools/reversibility_harness.py --model <id> --forget forget_hi.jsonl --retain retain_en.jsonl
