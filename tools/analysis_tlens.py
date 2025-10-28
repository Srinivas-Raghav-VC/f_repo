#!/usr/bin/env python3
# moved to tools/ (repo reorg 2025-10)
"""
Optional TransformerLens-based analysis:
- Computes per-layer alignment (CKA/Procrustes/ANC) between two text sets
- Optionally runs simple ablations (zeroing heads/MLPs) to localize signal

Usage:
  python analysis_tlens.py --model EleutherAI/pythia-70m --a forget_hi.jsonl --b retain_en.jsonl --out tlens_report.json

Notes:
- Requires `transformer_lens` installed; if missing, prints a helpful note and exits.
- Recommended to use small GPT-2/Pythia class models for full compatibility.
"""
import json, argparse, numpy as np
from typing import List

try:
    from transformer_lens import HookedTransformer
    _HAS_TLENS = True
except Exception:
    _HAS_TLENS = False

def read_jsonl(p: str, lim: int = 1000) -> List[str]:
    out = []
    with open(p, 'r', encoding='utf-8') as f:
        for i, l in enumerate(f):
            if lim and i >= lim: break
            l = l.strip()
            if not l: continue
            try:
                obj = json.loads(l)
                if isinstance(obj, dict) and 'text' in obj:
                    out.append(str(obj['text']))
                else:
                    out.append(str(obj))
            except Exception:
                out.append(l)
    return out

def center(K):
    n=K.shape[0]; H=np.eye(n, dtype=np.float32)-np.ones((n,n), dtype=np.float32)/n
    return H@K@H

def cka(X, Y):
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    K = X @ X.T; L = Y @ Y.T
    Kc = center(K); Lc = center(L)
    hsic = float((Kc * Lc).sum())
    var1 = float((Kc * Kc).sum()) ** 0.5
    var2 = float((Lc * Lc).sum()) ** 0.5
    return float(hsic / (var1 * var2 + 1e-8))

def procrustes(X, Y):
    Xc = X - X.mean(0, keepdims=True); Yc = Y - Y.mean(0, keepdims=True)
    C = Xc.T @ Yc
    _, s, _ = np.linalg.svd(C, full_matrices=False)
    return float(s.sum() / (np.linalg.norm(Xc, 'fro') * np.linalg.norm(Yc, 'fro') + 1e-8))

def anc(X, Y):
    n = min(len(X), len(Y))
    if n < 100 or X.shape[1] != Y.shape[1]: return 0.0
    i = np.random.choice(len(X), n, replace=False)
    j = np.random.choice(len(Y), n, replace=False)
    Xs = X[i]; Ys = Y[j]
    Xs = Xs - Xs.mean(0, keepdims=True); Ys = Ys - Ys.mean(0, keepdims=True)
    sx = Xs.std(0, keepdims=True) + 1e-6; sy = Ys.std(0, keepdims=True) + 1e-6
    corr = (Xs * Ys).mean(0) / (sx * sy).ravel()
    return float(np.nanmean(np.abs(corr)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--a', required=True, help='jsonl A')
    ap.add_argument('--b', required=True, help='jsonl B')
    ap.add_argument('--cap', type=int, default=1000)
    ap.add_argument('--out', default='tlens_report.json')
    args = ap.parse_args()

    if not _HAS_TLENS:
        print('[tlens] transformer_lens not installed. Install it to use this script.')
        return

    A = read_jsonl(args.a, args.cap)
    B = read_jsonl(args.b, args.cap)

    model = HookedTransformer.from_pretrained(args.model)
    # Collect per-layer mean activations at final token position
    def collect(texts):
        feats = {}
        for t in texts:
            out = model.run_with_cache(t)
            # resid_pre: list of [1, seq, d_model]
            for l in range(model.cfg.n_layers):
                h = out[1][f'blocks.{l}.hook_resid_pre'][0, -1].detach().cpu().numpy()
                feats.setdefault(l, []).append(h)
        for l in feats:
            feats[l] = np.array(feats[l], dtype=np.float32)
        return feats

    Aacts = collect(A); Bacts = collect(B)
    report = {'layers': {}}
    for l in range(model.cfg.n_layers):
        Xa = Aacts.get(l); Xb = Bacts.get(l)
        if Xa is None or Xb is None or len(Xa)==0 or len(Xb)==0:
            report['layers'][str(l)] = {'cka': 0.0, 'proc': 0.0, 'anc': 0.0}
            continue
        report['layers'][str(l)] = {'cka': cka(Xa, Xb), 'proc': procrustes(Xa, Xb), 'anc': anc(Xa, Xb)}

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'ok': True, 'out': args.out}, indent=2))

if __name__ == '__main__':
    main()
