
import torch
import os
import numpy as np
from mmie import load_causal_lm, AutoTokenizer, select_layers, read_jsonl, _resolve_blocks

def debug_selection():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_id} on {device}...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Fix tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    model = load_causal_lm(model_id, tok, device, None, eval_mode=True)
    
    # Create dummy data if files don't exist, or use real data if available
    # We'll try to read from data/ but fallback to synthetic
    try:
        forget = read_jsonl("data/forget_hi.jsonl")[:20]
        retain = read_jsonl("data/retain_en.jsonl")[:20]
        print(f"Loaded real data: {len(forget)} forget, {len(retain)} retain")
    except:
        print("Using synthetic data")
        forget = ["नमस्ते दुनिया" for _ in range(20)]
        retain = ["Hello world" for _ in range(20)]
        
    n_layers = len(_resolve_blocks(model))
    print(f"Model has {n_layers} layers")
    
    print("Running select_layers with verbose=True...")
    # We patch print in mmie to capture output or just rely on its internal prints
    # But better, let's modify the internal loop of select_layers in mmie.py to print raw values
    # Actually, we can just run it and see what happens since I enabled verbose in the call
    
    chosen, scores = select_layers(model, tok, forget, retain, n_layers, device,
                                  cap=20, top_k=5, min_layer=2,
                                  select_mode='semantic', 
                                  verbose=True)
    
    print("\n--- Detailed Scores ---")
    for li in sorted(scores.keys()):
        s = scores[li]
        print(f"L{li:02d}: Combo={s['combo']:.4f} | CKA={s['cka']:.4f} | Proc={s['proc']:.4f} | ANC={s['anc']:.4f}")

if __name__ == "__main__":
    debug_selection()
