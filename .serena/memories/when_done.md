When a task is completed:
- Run a tiny baseline on TinyLlama to verify no syntax/runtime regressions.
- If metrics were changed, run mmie.py with --report_token_kl (and --sae_gate if applicable) and inspect eval_report.json.
- If LID logic changed, test romanized and scripted strings.
- Keep secrets out of VCS (.env ignored). Refresh README snippets if CLI flags changed.
- If adding new features (analysis pack, ES-romanized), include a short usage example in README.md.
