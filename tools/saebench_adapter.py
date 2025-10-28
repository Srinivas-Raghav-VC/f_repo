#!/usr/bin/env python3
"""
Light SAEBench adapter/export utility.

This script wraps the repo's TopKSAE into a minimal interface SAEBench can use,
and optionally exports per-layer E/D tensors to disk for external evaluation.

Usage:
  python tools/saebench_adapter.py --ckpt_dir ckpt_lora_final --layers 8 16 24 --out_dir sae_export

Notes:
  - If SAEBench is installed (pip install sae-bench), you can import this module
    and pass TopKSAEAdapter to its evaluators directly.
  - Export format: torch .pt files with keys {'E': E[m,d], 'D': D[d,m], 'k', 'expansion', 'd'}
"""
import os, argparse, json, torch
from typing import List

# Import TopKSAE from mmie
from mmie import TopKSAE

class TopKSAEAdapter:
    def __init__(self, E: torch.Tensor, D: torch.Tensor, d: int, k: int, expansion: int):
        self.E = torch.as_tensor(E).contiguous()   # [m, d]
        self.D = torch.as_tensor(D).contiguous()   # [d, m]
        self.d = int(d); self.k = int(k); self.expansion = int(expansion)
        self.m = int(self.E.shape[0])
    def encode(self, x: torch.Tensor) -> torch.Tensor:  # [*, d] -> [*, m]
        return x @ self.E.T
    def decode(self, z: torch.Tensor) -> torch.Tensor:  # [*, m] -> [*, d]
        return z @ self.D.T

def export_layer(ckpt_dir: str, layer: int, out_dir: str, d: int, k: int, expansion: int):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"sae_layer{layer}.pt")
    if not os.path.exists(path):
        return False, f"missing {path}"
    sae = TopKSAE(d=d, k=k, expansion=expansion)
    sd = torch.load(path, map_location='cpu')
    sae.load_state_dict(sd, strict=False)
    payload = {
        'E': sae.E.weight.detach().cpu(),
        'D': sae.D.weight.detach().cpu(),
        'k': int(k), 'expansion': int(expansion), 'd': int(d)
    }
    outp = os.path.join(out_dir, f"sae_layer{layer}_ED.pt")
    torch.save(payload, outp)
    return True, outp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--layers', nargs='+', type=int, required=True)
    ap.add_argument('--d', type=int, default=None, help='Hidden size (if omitted, tries to read from a layer file)')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--expansion', type=int, default=16)
    ap.add_argument('--out_dir', default='sae_export')
    args = ap.parse_args()

    # If d is not provided, try to infer from the first present layer
    d = args.d
    if d is None:
        for li in args.layers:
            p = os.path.join(args.ckpt_dir, f"sae_layer{li}.pt")
            if os.path.exists(p):
                sd = torch.load(p, map_location='cpu')
                # find D weight to get shape [d, m]
                for kkey in ('D.weight', 'W_dec', 'decoder.weight'):
                    if kkey in sd:
                        d = int(sd[kkey].shape[0]); break
                if d is not None: break
    if d is None:
        print(json.dumps({'ok': False, 'error': 'could not infer hidden size d; pass --d'}, indent=2))
        return

    outs = {}
    for li in args.layers:
        ok, msg = export_layer(args.ckpt_dir, li, args.out_dir, d, args.k, args.expansion)
        outs[str(li)] = {'ok': ok, 'path_or_error': msg}
    print(json.dumps({'ok': True, 'layers': outs, 'out_dir': args.out_dir}, indent=2))

if __name__ == '__main__':
    main()

