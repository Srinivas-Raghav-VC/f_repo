#!/usr/bin/env python3
"""
Sweep SAE-gate alpha and plot ES vs PPL for slides.

Reasoning: a dose–response curve is the clearest, compact way to show
the mechanism works as intended — stronger gating (higher alpha)
reduces Extraction Strength (ES) while Perplexity (PPL) on retain
stays within gate thresholds. This gives a single visual that ties
intervention strength to outcomes, and quickly surfaces regressions.

If matplotlib is installed, saves a PNG; otherwise writes a JSON/CSV.

Usage:
  python tools/sweep_alpha.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --forget data/forget_hi.jsonl --retain data/retain_en.jsonl \
    --alphas 0.2 0.5 0.8 --ckpt_dir ckpt_lora_final --device cuda

Notes:
  - Reuses mmie.py utilities (no training). If SAE weights for chosen
    layers exist in ckpt_dir, loads them; else trains briefly.
  - Uses static SAEGate with fixed alpha per sweep point and mmie.generate().
"""
import os, json, argparse, math
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt  # optional
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

from dotenv import load_dotenv
from transformers import AutoTokenizer

# Import repo utilities
from mmie import (
    LIDEnsemble, LIDConfig,
    select_layers, TopKSAE, SAEGate,
    pick_semantic_sae_features, pick_sae_features_forget_vs_retain,
    load_causal_lm, generate, extraction_strength, perplexity,
    read_jsonl,
)

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--forget', required=True)
    ap.add_argument('--retain', required=True)
    ap.add_argument('--mixed', default=None)
    ap.add_argument('--ckpt_dir', default='ckpt_lora_final')
    ap.add_argument('--alphas', nargs='+', type=float, default=[0.2, 0.5, 0.8])
    ap.add_argument('--sae_k', type=int, default=32)
    ap.add_argument('--sae_expansion', type=int, default=8)
    ap.add_argument('--semantic_features', action='store_true')
    ap.add_argument('--semantic_tau', type=float, default=0.0)
    ap.add_argument('--sample_cap', type=int, default=800)
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='sweep_alpha_results.json')
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Data
    forget = read_jsonl(args.forget)
    retain = read_jsonl(args.retain)
    mixed = read_jsonl(args.mixed) if args.mixed else []

    # LID
    lid = LIDEnsemble(LIDConfig(vote_require_majority=True))

    # Model/tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.padding_side = 'left'
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    mdl = load_causal_lm(args.model, tok, args.device, os.environ.get('HF_TOKEN'), eval_mode=True)
    [p.requires_grad_(False) for p in mdl.parameters()]
    n_layers = len(mdl.model.layers)

    # Choose layers
    chosen, _scores = select_layers(mdl, tok, forget, retain, n_layers, args.device, cap=args.sample_cap, top_k=3, use_anc=False)

    # Prepare/load SAEs + pick features
    sae_modules = {}
    feat_idx = {}
    for li in chosen:
        path = os.path.join(args.ckpt_dir, f'sae_layer{li}.pt')
        sae = TopKSAE(d=mdl.config.hidden_size, k=args.sae_k, expansion=args.sae_expansion).to(args.device)
        if os.path.exists(path):
            try:
                sae.load_state_dict(__import__('torch').load(path, map_location=args.device), strict=False)
                sae.eval()
            except Exception:
                pass
        else:
            # quick train to get usable features
            from mmie import train_sae
            sae,_ = train_sae(mdl, tok, forget+retain, li, args.device, steps=500, bs=32, seq_len=args.max_len, k=args.sae_k, expansion=args.sae_expansion)
            __import__('torch').save(sae.state_dict(), path)
        # pick features
        if args.semantic_features:
            try:
                from mmie import _romanize_texts, _make_devanagari_gibberish
                idx = pick_semantic_sae_features(
                    sae, mdl, tok,
                    hindi_deva=forget,
                    hindi_roman=_romanize_texts(forget),
                    deva_gib=_make_devanagari_gibberish(retain, seed=0),
                    layer=li, device=args.device, max_len=args.max_len,
                    bs=32, cap_each=256, topk=64, tau=args.semantic_tau,
                )
            except Exception:
                idx = pick_sae_features_forget_vs_retain(sae, mdl, tok, forget, retain, li, args.device, max_len=args.max_len, bs=32, cap_each=256, topk=64)
        else:
            idx = pick_sae_features_forget_vs_retain(sae, mdl, tok, forget, retain, li, args.device, max_len=args.max_len, bs=32, cap_each=256, topk=64)
        sae_modules[li] = sae
        feat_idx[li] = idx

    results = []
    for a in args.alphas:
        gate = SAEGate(mdl, chosen, sae_modules, {li: feat_idx[li] for li in chosen}, alpha=float(a))
        gens_f = generate(mdl, tok, forget[:200], args.device)
        es_forget = extraction_strength(gens_f, lid, target_code='hi', use_script_guard=True)
        ppl_retain = perplexity(mdl, tok, retain[:200], args.device)
        entry = {'alpha': float(a), 'ES_forget': float(es_forget), 'PPL_retain': float(ppl_retain)}
        results.append(entry)
        gate.remove()

    # Save JSON + CSV
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 'layers': chosen, 'results': results}, f, indent=2)
    csv_path = Path(args.out).with_suffix('.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('alpha,ES_forget,PPL_retain\n')
        for r in results:
            f.write(f"{r['alpha']},{r['ES_forget']},{r['PPL_retain']}\n")

    if _HAS_PLT:
        xs = [r['alpha'] for r in results]
        es = [r['ES_forget'] for r in results]
        ppl = [r['PPL_retain'] for r in results]
        fig, ax1 = plt.subplots()
        c1 = 'tab:blue'; c2 = 'tab:red'
        ax1.set_xlabel('SAE-gate alpha')
        ax1.set_ylabel('ES (forget, script-aware)', color=c1)
        ax1.plot(xs, es, marker='o', color=c1, label='ES_forget')
        ax1.tick_params(axis='y', labelcolor=c1)
        ax2 = ax1.twinx()
        ax2.set_ylabel('PPL (retain)', color=c2)
        ax2.plot(xs, ppl, marker='s', linestyle='--', color=c2, label='PPL_retain')
        ax2.tick_params(axis='y', labelcolor=c2)
        plt.title(f"Dose–response: {args.model}")
        fig.tight_layout()
        png = Path(args.out).with_suffix('.png')
        plt.savefig(png, dpi=160)
        print(f"[ok] wrote {png}")

    print(f"[ok] wrote {args.out} and {csv_path}")

if __name__ == '__main__':
    main()

