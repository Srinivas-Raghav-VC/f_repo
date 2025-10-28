#!/usr/bin/env python3
# moved to tools/ (repo reorg 2025-10)
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def bench(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cpu", seq_len=256, batch_size=8, steps=20):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    inp = tok(["hello"]*batch_size, return_tensors="pt", padding=True, truncation=True, max_length=seq_len).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(5): mdl(**inp)
    # bench
    s = time.time()
    with torch.no_grad():
        for _ in range(steps): mdl(**inp)
    dt = time.time() - s
    tokens = steps * batch_size * seq_len
    print(f"device={device} model={model_id}")
    print(f"seq_len={seq_len} batch={batch_size} steps={steps}")
    print(f"throughput ≈ {tokens/dt:.1f} tokens/sec")
    print(f"tokens={tokens}, seconds={dt:.2f}")

if __name__ == "__main__":
    bench()
