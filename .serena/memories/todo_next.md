Implement two new scripts:
1) reversibility_harness.py: load edited arm (or attach LoRA), train briefly on forget set (NLL descent, e.g., 25–100 steps), and report ES and token-KL pre vs post relative to base.
2) build_training_pairs.py: use Gemini 2.5 Flash to synthesize preferred/dispreferred pairs for NPO, including romanized/homoglyph/code-switch adversaries; write pairs.jsonl.
Keep CLI minimal, robust JSON parsing, and tight logging.