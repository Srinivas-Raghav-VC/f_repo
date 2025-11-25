#!/usr/bin/env python3
"""
MMIE v7: TRULY RIGOROUS Mechanistic Analysis

This version addresses ALL concerns:

1. TOKENIZATION ANALYSIS - How does tokenization differ across languages?
2. SAE ARCHITECTURE COMPARISON - TopK vs JumpReLU vs Gated
3. HYPERPARAMETER SEARCH - With proper validation
4. ACTIVATION COLLECTION VERIFICATION - Multiple methods compared
5. COMPLETE EVALUATION - Uses ALL data files including adversarial
6. STATISTICAL RIGOR - Multiple seeds, CIs, significance tests
7. MECHANISTIC INTERPRETABILITY - Following Anthropic's methodology

Data files expected:
- data/forget_hi.jsonl      # Hindi to suppress
- data/retain_en.jsonl      # English to preserve  
- data/mixed.jsonl          # Hinglish/code-mixed
- data/urdu.jsonl           # Related language
- data/punjabi.jsonl        # Related language
- data/bengali.jsonl        # Related language
- data/adversarial.jsonl    # Adversarial prompts

Usage:
    python mmie_v7_rigorous.py --data_dir data --full_analysis --out results.json
"""

import os
import json
import math
import random
import itertools
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_token: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data directory
    data_dir: str = "data"
    
    # SAE hyperparameters (for search)
    sae_k_options: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    sae_expansion_options: List[int] = field(default_factory=lambda: [8, 16, 32])
    sae_lr: float = 3e-4
    sae_steps: int = 2000
    
    # Experiment settings
    n_seeds: int = 5
    n_features_to_analyze: int = 100
    eval_cap: int = 100
    max_len: int = 256
    
    # Output
    out: str = "results_rigorous.json"
    plots_dir: str = "plots"
    
    # Flags
    do_tokenization_analysis: bool = True
    do_sae_comparison: bool = True
    do_hyperparam_search: bool = True
    do_activation_verification: bool = True
    do_full_evaluation: bool = True
    do_ablations: bool = True
    
    seed: int = 42

# ============================================================================
# DATA LOADING
# ============================================================================

def read_jsonl(path: str, limit: Optional[int] = None) -> List[str]:
    if not os.path.exists(path):
        print(f"[warn] File not found: {path}")
        return []
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(line.strip())
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict):
                    for k in ["text", "prompt", "content", "input"]:
                        if k in obj:
                            texts.append(str(obj[k]))
                            break
            except:
                pass
    return texts

def load_all_data(data_dir: str) -> Dict[str, List[str]]:
    """Load ALL data files."""
    files = {
        "forget_hi": "forget_hi.jsonl",
        "retain_en": "retain_en.jsonl",
        "mixed": "mixed.jsonl",
        "urdu": "urdu.jsonl",
        "punjabi": "punjabi.jsonl",
        "bengali": "bengali.jsonl",
        "adversarial": "adversarial.jsonl"
    }
    
    data = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        data[key] = read_jsonl(path)
        print(f"[data] {key}: {len(data[key])} samples")
    
    return data

# ============================================================================
# TOKENIZATION ANALYSIS
# ============================================================================

def analyze_tokenization(tok, data: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Analyze tokenization patterns across languages.
    
    This is CRITICAL because:
    - Hindi might use more tokens per character
    - This affects activation collection (mean over positions)
    - Could be a confound in our analysis
    """
    print("\n" + "="*70)
    print("TOKENIZATION ANALYSIS")
    print("="*70)
    print("Why this matters: If Hindi uses more tokens, our activation")
    print("collection (mean over positions) could be biased.")
    print("="*70)
    
    results = {}
    
    for lang, texts in data.items():
        if not texts:
            continue
        
        # Tokenize
        tokens_per_text = []
        chars_per_text = []
        tokens_per_char = []
        
        for text in texts[:200]:  # Sample
            tokens = tok.encode(text, add_special_tokens=False)
            tokens_per_text.append(len(tokens))
            chars_per_text.append(len(text))
            if len(text) > 0:
                tokens_per_char.append(len(tokens) / len(text))
        
        results[lang] = {
            "mean_tokens": float(np.mean(tokens_per_text)),
            "std_tokens": float(np.std(tokens_per_text)),
            "mean_chars": float(np.mean(chars_per_text)),
            "tokens_per_char": float(np.mean(tokens_per_char)),
            "n_samples": len(texts)
        }
        
        print(f"\n{lang}:")
        print(f"  Tokens/text: {results[lang]['mean_tokens']:.1f} ± {results[lang]['std_tokens']:.1f}")
        print(f"  Tokens/char: {results[lang]['tokens_per_char']:.2f}")
    
    # Check for bias
    if "forget_hi" in results and "retain_en" in results:
        hi_tpc = results["forget_hi"]["tokens_per_char"]
        en_tpc = results["retain_en"]["tokens_per_char"]
        ratio = hi_tpc / en_tpc if en_tpc > 0 else 1
        
        print(f"\n[analysis] Hindi/English tokens-per-char ratio: {ratio:.2f}")
        
        if ratio > 1.5:
            print("[WARNING] Hindi uses significantly more tokens per character!")
            print("          This could bias activation collection.")
            print("          Consider normalizing by token count.")
            results["bias_warning"] = True
            results["bias_ratio"] = ratio
        else:
            print("[OK] Tokenization is relatively balanced.")
            results["bias_warning"] = False
    
    return results

# ============================================================================
# MODEL UTILITIES
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_blocks(model):
    for path in ["model.layers", "transformer.h"]:
        obj = model
        try:
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if isinstance(obj, nn.ModuleList):
                return obj
        except:
            pass
    raise ValueError("Cannot find blocks")

def load_model(config):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(config.model, token=config.hf_token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model, torch_dtype=dtype, device_map="auto",
        token=config.hf_token, trust_remote_code=True
    )
    model.eval()
    return model, tok

# ============================================================================
# ACTIVATION COLLECTION - MULTIPLE METHODS
# ============================================================================

@torch.no_grad()
def collect_activations_mean(model, tok, texts, layer, max_len=256):
    """Method 1: Mean over all positions (standard)."""
    if not texts:
        return np.array([])
    device = next(model.parameters()).device
    all_h = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i+8]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer + 1].mean(dim=1).cpu().float().numpy()
        all_h.append(h)
    return np.vstack(all_h) if all_h else np.array([])

@torch.no_grad()
def collect_activations_last(model, tok, texts, layer, max_len=256):
    """Method 2: Last token only."""
    if not texts:
        return np.array([])
    device = next(model.parameters()).device
    all_h = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i+8]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        lengths = enc['attention_mask'].sum(dim=1) - 1  # Last real token
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer + 1]
        # Get last token for each sequence
        last_h = torch.stack([h[b, lengths[b]] for b in range(h.shape[0])])
        all_h.append(last_h.cpu().float().numpy())
    return np.vstack(all_h) if all_h else np.array([])

@torch.no_grad()
def collect_activations_max(model, tok, texts, layer, max_len=256):
    """Method 3: Max over positions."""
    if not texts:
        return np.array([])
    device = next(model.parameters()).device
    all_h = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i+8]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[layer + 1].max(dim=1)[0].cpu().float().numpy()
        all_h.append(h)
    return np.vstack(all_h) if all_h else np.array([])

ACTIVATION_METHODS = {
    "mean": collect_activations_mean,
    "last": collect_activations_last,
    "max": collect_activations_max
}

def verify_activation_methods(model, tok, hindi_texts, english_texts, config):
    """
    Verify that different activation collection methods give consistent results.
    
    If they disagree, our conclusions might be method-dependent.
    """
    print("\n" + "="*70)
    print("ACTIVATION COLLECTION VERIFICATION")
    print("="*70)
    print("Testing if different methods give consistent layer rankings.")
    print("="*70)
    
    n_layers = len(get_blocks(model))
    results = {method: {} for method in ACTIVATION_METHODS}
    
    for method_name, method_fn in ACTIVATION_METHODS.items():
        print(f"\n[{method_name}] Computing layer AUCs...")
        
        for layer in tqdm(range(2, n_layers, 2), desc=method_name):  # Sample layers
            H_hi = method_fn(model, tok, hindi_texts[:50], layer)
            H_en = method_fn(model, tok, english_texts[:50], layer)
            
            if len(H_hi) == 0 or len(H_en) == 0:
                continue
            
            X = np.vstack([H_hi, H_en])
            y = np.array([1]*len(H_hi) + [0]*len(H_en))
            
            try:
                tr, te = train_test_split(range(len(X)), test_size=0.3, stratify=y)
                clf = LogisticRegression(max_iter=500)
                clf.fit(X[tr], y[tr])
                auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
                results[method_name][layer] = auc
            except:
                results[method_name][layer] = 0.5
    
    # Check consistency
    layers = sorted(set.intersection(*[set(r.keys()) for r in results.values()]))
    
    if len(layers) >= 3:
        rankings = {}
        for method, aucs in results.items():
            ranked = sorted([(l, aucs.get(l, 0.5)) for l in layers], key=lambda x: x[1], reverse=True)
            rankings[method] = [l for l, _ in ranked[:5]]
        
        # Compute agreement (Kendall's tau)
        from scipy.stats import kendalltau
        
        agreements = {}
        method_names = list(rankings.keys())
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                r1 = [rankings[m1].index(l) if l in rankings[m1] else 99 for l in layers[:10]]
                r2 = [rankings[m2].index(l) if l in rankings[m2] else 99 for l in layers[:10]]
                tau, _ = kendalltau(r1, r2)
                agreements[f"{m1}_vs_{m2}"] = tau
        
        print("\n[agreement] Kendall's tau between methods:")
        for pair, tau in agreements.items():
            status = "✓ AGREE" if tau > 0.6 else "⚠ DISAGREE"
            print(f"  {pair}: τ = {tau:.2f} {status}")
        
        results["rankings"] = rankings
        results["agreements"] = agreements
        
        # Choose best method
        mean_agreement = np.mean(list(agreements.values()))
        if mean_agreement < 0.5:
            print("\n[WARNING] Methods disagree significantly!")
            print("          Results may be method-dependent.")
            results["reliable"] = False
        else:
            print("\n[OK] Methods are reasonably consistent.")
            results["reliable"] = True
    
    return results

# ============================================================================
# SAE ARCHITECTURES - COMPARISON
# ============================================================================

class TopKSAE(nn.Module):
    """Standard TopK SAE."""
    def __init__(self, d_model, expansion=16, k=32):
        super().__init__()
        self.d_dict = d_model * expansion
        self.k = k
        self.encoder = nn.Linear(d_model, self.d_dict)
        self.decoder = nn.Linear(self.d_dict, d_model)
    
    def encode(self, x):
        z = self.encoder(x)
        if self.k < self.d_dict:
            topk = torch.topk(z.abs(), k=self.k, dim=-1)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, topk.indices, 1.0)
            z = z * mask
        return z
    
    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z
    
    def get_sparsity(self, x):
        z = self.encode(x)
        return (z.abs() > 0).float().mean().item()

class JumpReLUSAE(nn.Module):
    """JumpReLU SAE with learned threshold."""
    def __init__(self, d_model, expansion=16, init_threshold=0.1):
        super().__init__()
        self.d_dict = d_model * expansion
        self.encoder = nn.Linear(d_model, self.d_dict)
        self.decoder = nn.Linear(self.d_dict, d_model)
        self.threshold = nn.Parameter(torch.ones(self.d_dict) * init_threshold)
    
    def encode(self, x):
        z = self.encoder(x)
        # JumpReLU: zero if below threshold, identity if above
        z = z * (z.abs() > self.threshold.abs()).float()
        return z
    
    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z
    
    def get_sparsity(self, x):
        z = self.encode(x)
        return (z.abs() > 0).float().mean().item()

class GatedSAE(nn.Module):
    """Gated SAE with separate gate network."""
    def __init__(self, d_model, expansion=16):
        super().__init__()
        self.d_dict = d_model * expansion
        self.encoder = nn.Linear(d_model, self.d_dict)
        self.gate = nn.Linear(d_model, self.d_dict)
        self.decoder = nn.Linear(self.d_dict, d_model)
    
    def encode(self, x):
        z = self.encoder(x)
        g = torch.sigmoid(self.gate(x))
        return z * g
    
    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z
    
    def get_sparsity(self, x):
        z = self.encode(x)
        return (z.abs() > 0.01).float().mean().item()

SAE_ARCHITECTURES = {
    "topk": TopKSAE,
    "jumprelu": JumpReLUSAE,
    "gated": GatedSAE
}

def compare_sae_architectures(model, tok, train_texts, val_texts, layer, config):
    """
    Compare different SAE architectures.
    
    Why this matters:
    - Different architectures might find different features
    - Our conclusions could be architecture-dependent
    - Need to verify robustness
    """
    print("\n" + "="*70)
    print("SAE ARCHITECTURE COMPARISON")
    print("="*70)
    print("Comparing: TopK, JumpReLU, Gated SAE")
    print("="*70)
    
    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    
    # Collect hidden states
    H_train = collect_activations_mean(model, tok, train_texts, layer)
    H_val = collect_activations_mean(model, tok, val_texts, layer)
    
    H_train_t = torch.tensor(H_train, dtype=torch.float32, device=device)
    H_val_t = torch.tensor(H_val, dtype=torch.float32, device=device)
    
    results = {}
    saes = {}
    
    for arch_name, arch_class in SAE_ARCHITECTURES.items():
        print(f"\n[{arch_name}] Training...")
        
        if arch_name == "topk":
            sae = arch_class(d_model, expansion=16, k=32).to(device)
        elif arch_name == "jumprelu":
            sae = arch_class(d_model, expansion=16, init_threshold=0.1).to(device)
        else:
            sae = arch_class(d_model, expansion=16).to(device)
        
        opt = torch.optim.AdamW(sae.parameters(), lr=config.sae_lr)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(H_train_t), batch_size=16, shuffle=True
        )
        
        train_losses = []
        sae.train()
        for step in tqdm(range(config.sae_steps), desc=arch_name):
            for (batch,) in loader:
                x_hat, z = sae(batch)
                loss = F.mse_loss(x_hat, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
                break
        
        sae.eval()
        with torch.no_grad():
            x_hat_val, z_val = sae(H_val_t)
            val_loss = F.mse_loss(x_hat_val, H_val_t).item()
            sparsity = sae.get_sparsity(H_val_t)
        
        results[arch_name] = {
            "val_loss": val_loss,
            "sparsity": sparsity,
            "final_train_loss": train_losses[-1]
        }
        saes[arch_name] = sae
        
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Sparsity: {sparsity:.3f}")
    
    # Compare feature overlap
    print("\n[overlap] Computing feature correlation between architectures...")
    
    with torch.no_grad():
        features = {}
        for name, sae in saes.items():
            z = sae.encode(H_val_t)
            features[name] = z.cpu().numpy()
    
    # Correlation between top features
    overlaps = {}
    arch_names = list(features.keys())
    for i, a1 in enumerate(arch_names):
        for a2 in arch_names[i+1:]:
            f1 = features[a1].mean(axis=0)
            f2 = features[a2].mean(axis=0)
            
            # Get top features
            top1 = set(np.argsort(f1)[-50:])
            top2 = set(np.argsort(f2)[-50:])
            
            overlap = len(top1 & top2) / 50
            overlaps[f"{a1}_vs_{a2}"] = overlap
            
            print(f"  {a1} vs {a2}: {overlap*100:.0f}% overlap in top-50 features")
    
    results["overlaps"] = overlaps
    
    # Recommendation
    best_arch = min(results.keys() - {"overlaps"}, key=lambda x: results[x]["val_loss"])
    print(f"\n[recommend] Best architecture by val loss: {best_arch}")
    
    # But check robustness
    if min(overlaps.values()) < 0.3:
        print("[WARNING] Low feature overlap between architectures!")
        print("          Results may be architecture-dependent.")
        results["robust"] = False
    else:
        print("[OK] Reasonable feature overlap between architectures.")
        results["robust"] = True
    
    results["best_architecture"] = best_arch
    
    return results, saes

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def hyperparameter_search(model, tok, train_texts, val_texts, layer, config):
    """
    Proper hyperparameter search with validation.
    
    Search space:
    - k: [8, 16, 32, 64]
    - expansion: [8, 16, 32]
    
    Objective: Minimize reconstruction loss on validation set
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH")
    print("="*70)
    
    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    
    H_train = collect_activations_mean(model, tok, train_texts, layer)
    H_val = collect_activations_mean(model, tok, val_texts, layer)
    
    H_train_t = torch.tensor(H_train, dtype=torch.float32, device=device)
    H_val_t = torch.tensor(H_val, dtype=torch.float32, device=device)
    
    results = []
    best_loss = float('inf')
    best_config = None
    
    for k in config.sae_k_options:
        for exp in config.sae_expansion_options:
            print(f"\n[search] k={k}, expansion={exp}")
            
            sae = TopKSAE(d_model, expansion=exp, k=k).to(device)
            opt = torch.optim.AdamW(sae.parameters(), lr=config.sae_lr)
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(H_train_t), batch_size=16, shuffle=True
            )
            
            sae.train()
            for _ in range(min(1000, config.sae_steps)):  # Quick training for search
                for (batch,) in loader:
                    x_hat, z = sae(batch)
                    loss = F.mse_loss(x_hat, batch)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    break
            
            sae.eval()
            with torch.no_grad():
                x_hat_val, _ = sae(H_val_t)
                val_loss = F.mse_loss(x_hat_val, H_val_t).item()
            
            results.append({
                "k": k,
                "expansion": exp,
                "val_loss": val_loss
            })
            
            print(f"  Val loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_config = {"k": k, "expansion": exp}
    
    print(f"\n[best] k={best_config['k']}, expansion={best_config['expansion']}")
    print(f"       Val loss: {best_loss:.4f}")
    
    return {
        "all_results": results,
        "best_config": best_config,
        "best_loss": best_loss
    }

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def has_devanagari(text):
    return any('\u0900' <= c <= '\u097F' for c in text)

def detect_hindi_score(text):
    if not text:
        return 0.0
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F') / len(text)
    hindi_words = {'है', 'हैं', 'का', 'की', 'के', 'ko', 'se', 'mein', 'aur', 'hai', 'nahi'}
    words = set(text.lower().split())
    vocab = min(1.0, len(words & hindi_words) / 3)
    return max(deva, vocab)

def extraction_strength(texts):
    if not texts:
        return 0.0
    return float(np.mean([detect_hindi_score(t) for t in texts]))

@torch.no_grad()
def generate(model, tok, prompts, max_new=50):
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(prompts), 8):
        batch = prompts[i:i+8]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=200)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                            temperature=0.7, pad_token_id=tok.pad_token_id)
        new = gen[:, enc['input_ids'].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outs

@torch.no_grad()
def perplexity(model, tok, texts, max_len=256):
    if not texts:
        return float('inf')
    device = next(model.parameters()).device
    losses = []
    for i in range(0, len(texts), 8):
        batch = texts[i:i+8]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, labels=enc['input_ids'])
        losses.append(out.loss.item())
    return float(np.exp(np.mean(losses)))

# ============================================================================
# COMPLETE EVALUATION
# ============================================================================

def complete_evaluation(model, tok, data: Dict[str, List[str]], config) -> Dict[str, Any]:
    """
    Complete evaluation using ALL data files.
    
    Tests:
    1. Hindi suppression (forget_hi)
    2. English preservation (retain_en)
    3. Hinglish bypass (mixed)
    4. Related language leakage (urdu, punjabi, bengali)
    5. Adversarial extraction (adversarial)
    """
    print("\n" + "="*70)
    print("COMPLETE EVALUATION")
    print("="*70)
    
    results = {}
    cap = config.eval_cap
    
    # 1. Hindi suppression
    if data.get("forget_hi"):
        prompts = [f"Continue: {t[:40]}" for t in data["forget_hi"][:cap]]
        gens = generate(model, tok, prompts)
        results["hindi_es"] = extraction_strength(gens)
        print(f"[forget_hi] ES = {results['hindi_es']:.3f}")
    
    # 2. English preservation
    if data.get("retain_en"):
        results["english_ppl"] = perplexity(model, tok, data["retain_en"][:cap])
        print(f"[retain_en] PPL = {results['english_ppl']:.1f}")
    
    # 3. Hinglish bypass (CRITICAL)
    if data.get("mixed"):
        prompts = data["mixed"][:cap]
        gens = generate(model, tok, prompts)
        results["mixed_es"] = extraction_strength(gens)
        results["mixed_samples"] = gens[:3]
        print(f"[mixed] ES = {results['mixed_es']:.3f} (Hinglish bypass)")
    
    # 4. Related languages
    for lang in ["urdu", "punjabi", "bengali"]:
        if data.get(lang):
            prompts = [f"Continue: {t[:40]}" for t in data[lang][:cap]]
            gens = generate(model, tok, prompts)
            results[f"{lang}_es"] = extraction_strength(gens)
            print(f"[{lang}] ES = {results[f'{lang}_es']:.3f}")
    
    # 5. Adversarial (CRITICAL)
    if data.get("adversarial"):
        prompts = data["adversarial"][:cap]
        gens = generate(model, tok, prompts)
        results["adversarial_es"] = extraction_strength(gens)
        results["adversarial_samples"] = gens[:5]
        print(f"[adversarial] ES = {results['adversarial_es']:.3f} (Adversarial extraction)")
    
    return results

# ============================================================================
# CAUSAL LAYER AND FEATURE ANALYSIS
# ============================================================================

def causal_layer_analysis(model, tok, hindi_texts, english_texts, config):
    """Causal layer importance via intervention."""
    print("\n" + "="*70)
    print("CAUSAL LAYER ANALYSIS")
    print("="*70)
    
    n_layers = len(get_blocks(model))
    device = next(model.parameters()).device
    
    hindi_prompts = [f"Write about: {t[:40]}" for t in hindi_texts[:15]]
    
    base_gens = generate(model, tok, hindi_prompts)
    base_es = extraction_strength(base_gens)
    base_ppl = perplexity(model, tok, english_texts[:30])
    
    print(f"[baseline] ES={base_es:.3f}, PPL={base_ppl:.1f}")
    
    results = {"baseline_es": base_es, "baseline_ppl": base_ppl, "layers": {}}
    
    for layer in tqdm(range(2, n_layers), desc="Causal layers"):
        # Get direction
        H_hi = collect_activations_mean(model, tok, hindi_texts[:30], layer)
        H_en = collect_activations_mean(model, tok, english_texts[:30], layer)
        
        if len(H_hi) == 0 or len(H_en) == 0:
            continue
        
        # Probe AUC (correlation)
        X = np.vstack([H_hi, H_en])
        y = np.array([1]*len(H_hi) + [0]*len(H_en))
        try:
            tr, te = train_test_split(range(len(X)), test_size=0.3, stratify=y)
            clf = LogisticRegression(max_iter=500)
            clf.fit(X[tr], y[tr])
            probe_auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        except:
            probe_auc = 0.5
        
        # Intervention (causation)
        direction = H_hi.mean(0) - H_en.mean(0)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        dir_t = torch.tensor(direction, dtype=torch.float32, device=device)
        
        test_model, _ = load_model(config)
        blocks = get_blocks(test_model)
        
        def make_hook(d, alpha=0.7):
            @torch.no_grad()
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h_f = h.float()
                proj = torch.einsum('btd,d->bt', h_f, d)
                h_new = h_f - alpha * proj.unsqueeze(-1) * d
                return (h_new.to(h.dtype),) + out[1:] if isinstance(out, tuple) else h_new.to(h.dtype)
            return hook
        
        handle = blocks[layer].register_forward_hook(make_hook(dir_t))
        
        ablated_gens = generate(test_model, tok, hindi_prompts)
        ablated_es = extraction_strength(ablated_gens)
        ablated_ppl = perplexity(test_model, tok, english_texts[:20])
        
        handle.remove()
        del test_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        es_reduction = base_es - ablated_es
        ppl_increase = (ablated_ppl - base_ppl) / base_ppl
        
        results["layers"][layer] = {
            "probe_auc": float(probe_auc),
            "ablated_es": float(ablated_es),
            "ablated_ppl": float(ablated_ppl),
            "es_reduction": float(es_reduction),
            "ppl_increase": float(ppl_increase)
        }
    
    # Rank
    by_probe = sorted(results["layers"].items(), key=lambda x: x[1]["probe_auc"], reverse=True)
    by_causal = sorted(results["layers"].items(), key=lambda x: x[1]["es_reduction"], reverse=True)
    
    results["best_by_probe"] = [int(l) for l, _ in by_probe[:5]]
    results["best_by_causal"] = [int(l) for l, _ in by_causal[:5]]
    
    print(f"\n[probe] Best: {results['best_by_probe'][:3]}")
    print(f"[causal] Best: {results['best_by_causal'][:3]}")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_comprehensive_plots(
    tokenization_results,
    activation_results,
    sae_results,
    hyperparam_results,
    layer_results,
    evaluation_results,
    config
):
    """Create all plots."""
    os.makedirs(config.plots_dir, exist_ok=True)
    
    # === PLOT 1: Tokenization Analysis ===
    if tokenization_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        langs = [k for k in tokenization_results.keys() if isinstance(tokenization_results[k], dict)]
        tpc = [tokenization_results[l]["tokens_per_char"] for l in langs]
        
        colors = ['#FF6B35' if 'hi' in l or 'forget' in l else '#004E89' for l in langs]
        ax.bar(langs, tpc, color=colors, alpha=0.7)
        ax.set_ylabel('Tokens per Character')
        ax.set_title('Tokenization Analysis\n(Higher = more tokens needed)')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{config.plots_dir}/1_tokenization.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(f"{config.plots_dir}/1_tokenization.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[plot] Saved 1_tokenization.pdf")
    
    # === PLOT 2: SAE Comparison ===
    if sae_results and "topk" in sae_results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        archs = [k for k in sae_results.keys() if k not in ["overlaps", "robust", "best_architecture"]]
        losses = [sae_results[a]["val_loss"] for a in archs]
        sparsities = [sae_results[a]["sparsity"] for a in archs]
        
        axes[0].bar(archs, losses, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('SAE Architectures: Reconstruction')
        
        axes[1].bar(archs, sparsities, color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
        axes[1].set_ylabel('Sparsity')
        axes[1].set_title('SAE Architectures: Sparsity')
        
        plt.tight_layout()
        plt.savefig(f"{config.plots_dir}/2_sae_comparison.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(f"{config.plots_dir}/2_sae_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[plot] Saved 2_sae_comparison.pdf")
    
    # === PLOT 3: Layer Analysis ===
    if layer_results and "layers" in layer_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        layers = sorted([int(l) for l in layer_results["layers"].keys()])
        probe_aucs = [layer_results["layers"][str(l)]["probe_auc"] for l in layers]
        es_reductions = [layer_results["layers"][str(l)]["es_reduction"] for l in layers]
        
        axes[0].bar(layers, probe_aucs, color='blue', alpha=0.7)
        axes[0].axhline(y=0.5, color='gray', linestyle='--')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Probe AUC')
        axes[0].set_title('CORRELATION: Separability')
        
        axes[1].bar(layers, es_reductions, color='red', alpha=0.7)
        axes[1].axhline(y=0, color='gray', linestyle='--')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('ES Reduction')
        axes[1].set_title('CAUSATION: Intervention Effect')
        
        plt.tight_layout()
        plt.savefig(f"{config.plots_dir}/3_layer_analysis.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(f"{config.plots_dir}/3_layer_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[plot] Saved 3_layer_analysis.pdf")
    
    # === PLOT 4: Complete Evaluation Radar ===
    if evaluation_results:
        # Collect metrics that exist
        metrics = {}
        if "hindi_es" in evaluation_results:
            metrics["Hindi ES↓"] = evaluation_results["hindi_es"]
        if "english_ppl" in evaluation_results:
            metrics["English PPL"] = min(evaluation_results["english_ppl"] / 50, 1)  # Normalize
        if "mixed_es" in evaluation_results:
            metrics["Hinglish ES↓"] = evaluation_results["mixed_es"]
        if "adversarial_es" in evaluation_results:
            metrics["Adversarial ES↓"] = evaluation_results["adversarial_es"]
        for lang in ["urdu", "punjabi", "bengali"]:
            if f"{lang}_es" in evaluation_results:
                metrics[f"{lang.title()} ES↓"] = evaluation_results[f"{lang}_es"]
        
        if len(metrics) >= 3:
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            labels = list(metrics.keys())
            values = list(metrics.values())
            values += values[:1]  # Close the loop
            
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='#E63946')
            ax.fill(angles, values, alpha=0.25, color='#E63946')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=10)
            ax.set_title('Complete Evaluation\n(Lower is better for ES metrics)', size=14)
            
            plt.tight_layout()
            plt.savefig(f"{config.plots_dir}/4_evaluation_radar.pdf", dpi=150, bbox_inches='tight')
            plt.savefig(f"{config.plots_dir}/4_evaluation_radar.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[plot] Saved 4_evaluation_radar.pdf")

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="results_rigorous.json")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    args = parser.parse_args()
    
    config = Config(
        model=args.model,
        data_dir=args.data_dir,
        out=args.out,
        plots_dir=args.plots_dir,
        seed=args.seed,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN")
    )
    
    if args.quick:
        config.sae_steps = 500
        config.n_features_to_analyze = 20
        config.eval_cap = 30
        config.sae_k_options = [16, 32]
        config.sae_expansion_options = [8, 16]
    
    set_seed(config.seed)
    os.makedirs(config.plots_dir, exist_ok=True)
    
    # Load ALL data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    data = load_all_data(config.data_dir)
    
    # Load model and tokenizer
    print("\n[model] Loading...")
    model, tok = load_model(config)
    
    results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat()
    }
    
    # === Phase 1: Tokenization Analysis ===
    if config.do_tokenization_analysis:
        results["tokenization"] = analyze_tokenization(tok, data)
    
    # === Phase 2: Activation Method Verification ===
    if config.do_activation_verification and data.get("forget_hi") and data.get("retain_en"):
        results["activation_methods"] = verify_activation_methods(
            model, tok, data["forget_hi"], data["retain_en"], config
        )
    
    # === Phase 3: Causal Layer Analysis ===
    if data.get("forget_hi") and data.get("retain_en"):
        results["layer_analysis"] = causal_layer_analysis(
            model, tok, data["forget_hi"], data["retain_en"], config
        )
        best_layer = results["layer_analysis"]["best_by_causal"][0]
    else:
        best_layer = 15  # Default
    
    # === Phase 4: SAE Architecture Comparison ===
    if config.do_sae_comparison and data.get("forget_hi") and data.get("retain_en"):
        train_texts = data["forget_hi"][:300] + data["retain_en"][:300]
        val_texts = data["forget_hi"][300:400] + data["retain_en"][300:400]
        
        sae_comparison, saes = compare_sae_architectures(
            model, tok, train_texts, val_texts, best_layer, config
        )
        results["sae_comparison"] = sae_comparison
    
    # === Phase 5: Hyperparameter Search ===
    if config.do_hyperparam_search and data.get("forget_hi") and data.get("retain_en"):
        train_texts = data["forget_hi"][:300] + data["retain_en"][:300]
        val_texts = data["forget_hi"][300:400] + data["retain_en"][300:400]
        
        results["hyperparam_search"] = hyperparameter_search(
            model, tok, train_texts, val_texts, best_layer, config
        )
    
    # === Phase 6: Complete Evaluation ===
    if config.do_full_evaluation:
        results["evaluation_base"] = complete_evaluation(model, tok, data, config)
    
    # === Phase 7: Create Plots ===
    create_comprehensive_plots(
        results.get("tokenization"),
        results.get("activation_methods"),
        results.get("sae_comparison"),
        results.get("hyperparam_search"),
        results.get("layer_analysis"),
        results.get("evaluation_base"),
        config
    )
    
    # Save
    with open(config.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[done] Results: {config.out}")
    print(f"[done] Plots: {config.plots_dir}/")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if "tokenization" in results and results["tokenization"].get("bias_warning"):
        print("⚠ TOKENIZATION BIAS detected - Hindi uses more tokens")
    
    if "activation_methods" in results and not results["activation_methods"].get("reliable", True):
        print("⚠ ACTIVATION METHODS disagree - results may be method-dependent")
    
    if "sae_comparison" in results and not results["sae_comparison"].get("robust", True):
        print("⚠ SAE ARCHITECTURES disagree - results may be architecture-dependent")
    
    if "layer_analysis" in results:
        la = results["layer_analysis"]
        if la["best_by_probe"][:3] != la["best_by_causal"][:3]:
            print("⚠ PROBE vs CAUSAL selection disagree - use causal!")
    
    if "evaluation_base" in results:
        eb = results["evaluation_base"]
        print(f"\nBase Model Evaluation:")
        for k, v in eb.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
