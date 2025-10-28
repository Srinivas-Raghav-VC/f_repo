#!/usr/bin/env python3
"""
Lightweight loader for SAELens-trained SAEs.
Tries to read encoder/decoder weights from a local directory and map them into
our TopKSAE layout (E: [m,d], D: [d,m]).

Supported key patterns (flexible):
- 'W_enc', 'encoder.weight', 'E.weight', 'W_E' (m x d or d x m)
- 'W_dec', 'decoder.weight', 'D.weight', 'W_D' (d x m or m x d)

Usage:
  from backends.sae_lens_loader import load_sae_from_dir
  sae = load_sae_from_dir(layer_id=12, d=hidden, expansion=16, directory='sae_ckpts')
"""
from __future__ import annotations
import os, re, torch
from typing import Optional, Tuple


def _find_candidate_file(directory: str, layer_id: int) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    patterns = [
        rf"sae_layer{layer_id}\.pt$",
        rf"layer_{layer_id}\.pt$",
        rf"L{layer_id}\.pt$",
        rf".*layer.*{layer_id}.*\.pt$",
        rf".*sae.*{layer_id}.*\.pt$",
    ]
    files = os.listdir(directory)
    for pat in patterns:
        for f in files:
            if re.search(pat, f, flags=re.IGNORECASE):
                return os.path.join(directory, f)
    # fallback: first .pt in dir
    for f in files:
        if f.lower().endswith('.pt'):
            return os.path.join(directory, f)
    return None


def _extract_weights(sd: dict) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Allow nested state_dict
    if 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    # common keys
    enc_keys = [
        'W_enc', 'encoder.weight', 'E.weight', 'W_E', 'enc.weight', 'encoder.W', 'E'
    ]
    dec_keys = [
        'W_dec', 'decoder.weight', 'D.weight', 'W_D', 'dec.weight', 'decoder.W', 'D'
    ]
    E = None; D = None
    for k in enc_keys:
        if k in sd:
            E = torch.as_tensor(sd[k])
            break
    for k in dec_keys:
        if k in sd:
            D = torch.as_tensor(sd[k])
            break
    # Sometimes dicts are nested under names like 'E'/'D'
    if E is None:
        for k,v in sd.items():
            if isinstance(v, dict) and 'weight' in v:
                if 'enc' in k.lower() or k.lower().startswith('e'):
                    E = torch.as_tensor(v['weight'])
                if 'dec' in k.lower() or k.lower().startswith('d'):
                    D = torch.as_tensor(v['weight']) if D is None else D
    return E, D


def _orient_to(m_by_d: torch.Tensor, target_rows: int, target_cols: int) -> Optional[torch.Tensor]:
    if m_by_d is None:
        return None
    r, c = m_by_d.shape[-2], m_by_d.shape[-1]
    if (r, c) == (target_rows, target_cols):
        return m_by_d
    if (c, r) == (target_rows, target_cols):
        return m_by_d.T
    # shape mismatch
    return None


def load_sae_from_dir(layer_id: int, d: int, expansion: int, directory: str, device: str = 'cpu') -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Returns (E, D) tensors oriented as E:[m,d], D:[d,m], or None if not found/compatible."""
    path = _find_candidate_file(directory, layer_id)
    if path is None:
        return None
    try:
        sd = torch.load(path, map_location='cpu')
    except Exception:
        return None
    E_raw, D_raw = _extract_weights(sd)
    if E_raw is None or D_raw is None:
        return None
    m = d * expansion
    E = _orient_to(E_raw, m, d)
    D = _orient_to(D_raw, d, m)
    if E is None or D is None:
        return None
    return E.to(device=device, dtype=torch.float32), D.to(device=device, dtype=torch.float32)

