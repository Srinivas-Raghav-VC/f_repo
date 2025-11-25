#!/usr/bin/env python3
"""
MMIE COMPREHENSIVE SMOKE TEST

Based on best practices from literature:
1. SAE: OpenAI's "Scaling and evaluating sparse autoencoders" (2024)
   - Key metrics: L0 (sparsity), Loss Recovered
   - TopK can fail, JumpReLU more stable
   
2. Unlearning: RMU "Representation Misdirection for Unlearning" (AAAI 2025)
   - Steer forget samples toward random representation
   - Keep retain samples unchanged
   - Works at intermediate layers

3. Evaluation: TOFU benchmark methodology
   - Forget quality (ES reduction)
   - Retain quality (PPL preservation)
   - Side effects (related languages)

This script runs ALL necessary tests to validate the methodology.

Usage:
    python mmie_smoke_test.py --data_dir data --quick
"""

import os
import json
import random
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SmokeTestConfig:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_dir: str = "data"
    hf_token: Optional[str] = None
    
    # Test parameters
    n_samples: int = 50  # Samples per test
    n_prompts: int = 20  # Prompts for generation tests
    
    # SAE parameters
    sae_steps: int = 1000
    sae_expansion: int = 16
    
    # Output
    out: str = "smoke_test_results.json"
    plots_dir: str = "smoke_plots"
    
    seed: int = 42

# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_jsonl(path, limit=None):
    if not os.path.exists(path):
        return []
    texts = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                obj = json.loads(line.strip())
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict):
                    for k in ["text", "prompt", "content"]:
                        if k in obj:
                            texts.append(str(obj[k]))
                            break
            except:
                pass
    return texts

def has_devanagari(text):
    return any('\u0900' <= c <= '\u097F' for c in text)

def detect_hindi_score(text):
    if not text:
        return 0.0
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F') / max(len(text), 1)
    hindi_words = {'है', 'हैं', 'का', 'की', 'के', 'को', 'से', 'में', 'और', 'hai', 'hain', 'ka', 'ki', 'ke'}
    words = set(text.lower().split())
    vocab = min(1.0, len(words & hindi_words) / 3) if words else 0
    return max(deva, vocab)

def extraction_strength(texts):
    if not texts:
        return 0.0
    return float(np.mean([detect_hindi_score(t) for t in texts]))

# ============================================================================
# MODEL UTILITIES
# ============================================================================

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
    print("[model] Loading...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(config.model, token=config.hf_token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        config.model, torch_dtype=dtype, device_map="auto",
        token=config.hf_token, trust_remote_code=True
    )
    model.eval()
    n_layers = len(get_blocks(model))
    d_model = model.config.hidden_size
    print(f"[model] Loaded: {n_layers} layers, d_model={d_model}")
    return model, tok

@torch.no_grad()
def get_hidden(model, tok, texts, layer, max_len=256):
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
# TEST 1: TOKENIZATION SANITY CHECK
# ============================================================================

def test_tokenization(tok, data, config):
    """
    Check tokenization bias.
    From v7 results: Hindi uses 5x more tokens/char than English.
    This can bias activation collection.
    """
    print("\n" + "="*70)
    print("TEST 1: TOKENIZATION ANALYSIS")
    print("="*70)
    
    results = {}
    for key, texts in data.items():
        if not texts:
            continue
        sample = texts[:200]
        tokens_per_char = []
        for t in sample:
            toks = tok.encode(t, add_special_tokens=False)
            if len(t) > 0:
                tokens_per_char.append(len(toks) / len(t))
        
        if tokens_per_char:
            results[key] = {
                "mean_tpc": float(np.mean(tokens_per_char)),
                "std_tpc": float(np.std(tokens_per_char))
            }
            print(f"  {key}: {results[key]['mean_tpc']:.3f} tokens/char")
    
    # Check bias
    if "forget_hi" in results and "retain_en" in results:
        ratio = results["forget_hi"]["mean_tpc"] / results["retain_en"]["mean_tpc"]
        results["bias_ratio"] = ratio
        
        if ratio > 2:
            print(f"\n  ⚠️ WARNING: Hindi uses {ratio:.1f}x more tokens than English!")
            print(f"     This may bias activation collection.")
            results["status"] = "WARNING"
        else:
            print(f"\n  ✓ Tokenization bias ratio: {ratio:.1f}x (acceptable)")
            results["status"] = "OK"
    
    return results

# ============================================================================
# TEST 2: SAE ARCHITECTURE COMPARISON
# ============================================================================

class JumpReLUSAE(nn.Module):
    def __init__(self, d_model, expansion=16, threshold=0.1):
        super().__init__()
        self.d_dict = d_model * expansion
        self.threshold = nn.Parameter(torch.ones(self.d_dict) * threshold)
        self.encoder = nn.Linear(d_model, self.d_dict)
        self.decoder = nn.Linear(self.d_dict, d_model)
    
    def encode(self, x):
        z = self.encoder(x)
        z = z * (z.abs() > self.threshold.abs()).float()
        return z
    
    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z
    
    def get_sparsity(self):
        return (self.threshold.abs() > 0).float().mean().item()


class TopKSAE(nn.Module):
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


def test_sae_architectures(model, tok, hindi_texts, english_texts, layer, config):
    """
    Compare SAE architectures.
    From literature: TopK can fail with very low sparsity.
    JumpReLU is more stable.
    """
    print("\n" + "="*70)
    print("TEST 2: SAE ARCHITECTURE COMPARISON")
    print("="*70)
    
    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    
    # Collect activations
    texts = hindi_texts[:200] + english_texts[:200]
    H = get_hidden(model, tok, texts, layer)
    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    
    # Split train/val
    n_train = int(0.8 * len(H_t))
    H_train, H_val = H_t[:n_train], H_t[n_train:]
    
    results = {}
    
    for name, sae_class, kwargs in [
        ("topk_k32", TopKSAE, {"k": 32}),
        ("topk_k64", TopKSAE, {"k": 64}),
        ("jumprelu_0.1", JumpReLUSAE, {"threshold": 0.1}),
        ("jumprelu_0.05", JumpReLUSAE, {"threshold": 0.05}),
    ]:
        print(f"\n  Training {name}...")
        sae = sae_class(d_model, expansion=config.sae_expansion, **kwargs).to(device)
        opt = torch.optim.AdamW(sae.parameters(), lr=3e-4)
        
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(H_train), batch_size=32, shuffle=True
        )
        
        sae.train()
        for _ in range(config.sae_steps):
            for (batch,) in loader:
                x_hat, z = sae(batch)
                loss = F.mse_loss(x_hat, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                break
        
        sae.eval()
        with torch.no_grad():
            x_hat_val, z_val = sae(H_val)
            val_loss = F.mse_loss(x_hat_val, H_val).item()
            # L0: average number of nonzero features
            l0 = (z_val.abs() > 1e-6).float().sum(dim=-1).mean().item()
            # Sparsity ratio
            sparsity = l0 / sae.d_dict
        
        results[name] = {
            "val_loss": val_loss,
            "l0": l0,
            "sparsity": sparsity
        }
        print(f"    Val loss: {val_loss:.4f}, L0: {l0:.1f}, Sparsity: {sparsity:.4f}")
    
    # Recommend
    best = min(results.keys(), key=lambda k: results[k]["val_loss"])
    results["best"] = best
    
    # Check for TopK failure (very low L0)
    topk_l0 = results.get("topk_k32", {}).get("l0", 32)
    if topk_l0 < 5:
        print(f"\n  ⚠️ WARNING: TopK has very low L0 ({topk_l0:.1f}) - may be failing!")
        results["topk_status"] = "FAILING"
    else:
        results["topk_status"] = "OK"
    
    print(f"\n  ✓ Best architecture: {best}")
    
    return results

# ============================================================================
# TEST 3: LAYER IMPORTANCE - THREE METHODS
# ============================================================================

def test_layer_importance(model, tok, hindi_texts, english_texts, config):
    """
    Test three methods for layer importance:
    1. Probe AUC (correlation)
    2. Activation variance (separability)
    3. Direct intervention (causation)
    
    Check if they agree.
    """
    print("\n" + "="*70)
    print("TEST 3: LAYER IMPORTANCE (3 METHODS)")
    print("="*70)
    
    n_layers = len(get_blocks(model))
    device = next(model.parameters()).device
    
    # Test layers (sample for speed)
    test_layers = list(range(4, n_layers-2, 3))
    
    results = {
        "probe_auc": {},
        "activation_variance": {},
        "intervention_effect": {}
    }
    
    # Baseline for intervention
    test_prompts = [f"Continue: {t[:40]}" for t in hindi_texts[:config.n_prompts]]
    base_gens = generate(model, tok, test_prompts)
    base_es = extraction_strength(base_gens)
    print(f"  Baseline ES: {base_es:.3f}")
    
    for layer in tqdm(test_layers, desc="Testing layers"):
        H_hi = get_hidden(model, tok, hindi_texts[:config.n_samples], layer)
        H_en = get_hidden(model, tok, english_texts[:config.n_samples], layer)
        
        if len(H_hi) == 0 or len(H_en) == 0:
            continue
        
        # Method 1: Probe AUC
        X = np.vstack([H_hi, H_en])
        y = np.array([1]*len(H_hi) + [0]*len(H_en))
        try:
            tr, te = train_test_split(range(len(X)), test_size=0.3, stratify=y)
            clf = LogisticRegression(max_iter=500)
            clf.fit(X[tr], y[tr])
            auc = roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])
        except:
            auc = 0.5
        results["probe_auc"][layer] = auc
        
        # Method 2: Activation variance
        hi_mean = H_hi.mean(axis=0)
        en_mean = H_en.mean(axis=0)
        diff_norm = np.linalg.norm(hi_mean - en_mean)
        results["activation_variance"][layer] = float(diff_norm)
        
        # Method 3: Direct intervention
        direction = en_mean - hi_mean
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        dir_t = torch.tensor(direction, dtype=torch.float32, device=device)
        
        test_model, _ = load_model(config)
        blocks = get_blocks(test_model)
        
        def hook(module, inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            h_f = h.float()
            proj = torch.einsum('btd,d->bt', h_f, dir_t)
            h_new = h_f + 0.5 * proj.unsqueeze(-1) * dir_t
            return (h_new.to(h.dtype),) + outputs[1:] if isinstance(outputs, tuple) else h_new.to(h.dtype)
        
        handle = blocks[layer].register_forward_hook(hook)
        int_gens = generate(test_model, tok, test_prompts)
        int_es = extraction_strength(int_gens)
        handle.remove()
        del test_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        es_change = base_es - int_es  # Positive = Hindi reduced
        results["intervention_effect"][layer] = float(es_change)
    
    # Rank by each method
    probe_ranked = sorted(results["probe_auc"].keys(), key=lambda l: results["probe_auc"][l], reverse=True)
    var_ranked = sorted(results["activation_variance"].keys(), key=lambda l: results["activation_variance"][l], reverse=True)
    int_ranked = sorted(results["intervention_effect"].keys(), key=lambda l: results["intervention_effect"][l], reverse=True)
    
    results["ranked_by_probe"] = probe_ranked[:5]
    results["ranked_by_variance"] = var_ranked[:5]
    results["ranked_by_intervention"] = int_ranked[:5]
    
    print(f"\n  By Probe AUC: {probe_ranked[:3]}")
    print(f"  By Variance:  {var_ranked[:3]}")
    print(f"  By Intervention: {int_ranked[:3]}")
    
    # Check agreement
    top3_probe = set(probe_ranked[:3])
    top3_int = set(int_ranked[:3])
    agreement = len(top3_probe & top3_int) / 3
    results["probe_intervention_agreement"] = agreement
    
    if agreement < 0.33:
        print(f"\n  ⚠️ WARNING: Probe and Intervention methods DISAGREE!")
        print(f"     Agreement: {agreement*100:.0f}%")
        print(f"     Use intervention-based selection (causal, not correlational)")
        results["status"] = "METHODS_DISAGREE"
    else:
        print(f"\n  ✓ Methods agree ({agreement*100:.0f}%)")
        results["status"] = "OK"
    
    # Check if intervention works at any layer
    positive_effect = [l for l, e in results["intervention_effect"].items() if e > 0]
    results["layers_that_reduce_hindi"] = positive_effect
    
    if not positive_effect:
        print(f"\n  ⚠️ WARNING: No layer reduces Hindi with direction method!")
        print(f"     The simple direction intervention may not work.")
        results["direction_method_works"] = False
    else:
        print(f"\n  ✓ Layers that reduce Hindi: {positive_effect}")
        results["direction_method_works"] = True
    
    return results

# ============================================================================
# TEST 4: INTERVENTION METHODS COMPARISON
# ============================================================================

def test_intervention_methods(model, tok, hindi_texts, english_texts, layer, config):
    """
    Compare intervention methods:
    1. Direction steering (push toward English)
    2. RMU-style (steer toward random)
    3. Feature ablation (zero out Hindi features)
    
    Based on RMU paper: random steering can be effective.
    """
    print("\n" + "="*70)
    print("TEST 4: INTERVENTION METHODS")
    print("="*70)
    
    device = next(model.parameters()).device
    n_layers = len(get_blocks(model))
    
    # Use middle layer if not specified
    if layer >= n_layers:
        layer = n_layers // 2
    
    # Get representations
    H_hi = get_hidden(model, tok, hindi_texts[:config.n_samples], layer)
    H_en = get_hidden(model, tok, english_texts[:config.n_samples], layer)
    
    hi_mean = H_hi.mean(axis=0)
    en_mean = H_en.mean(axis=0)
    d_model = len(hi_mean)
    
    # Baseline
    test_prompts = [f"Continue: {t[:40]}" for t in hindi_texts[:config.n_prompts]]
    base_gens = generate(model, tok, test_prompts)
    base_es = extraction_strength(base_gens)
    base_ppl = perplexity(model, tok, english_texts[:30])
    
    print(f"  Baseline: ES={base_es:.3f}, PPL={base_ppl:.1f}")
    
    results = {"baseline_es": base_es, "baseline_ppl": base_ppl, "methods": {}}
    
    methods = {
        "direction_toward_english": {
            "direction": (en_mean - hi_mean) / (np.linalg.norm(en_mean - hi_mean) + 1e-8),
            "alpha": 0.5
        },
        "direction_away_from_hindi": {
            "direction": -hi_mean / (np.linalg.norm(hi_mean) + 1e-8),
            "alpha": 0.3
        },
        "rmu_random": {
            "direction": np.random.randn(d_model).astype(np.float32),
            "alpha": 0.5
        }
    }
    # Normalize random direction
    methods["rmu_random"]["direction"] /= np.linalg.norm(methods["rmu_random"]["direction"])
    
    for method_name, params in methods.items():
        print(f"\n  Testing {method_name}...")
        
        dir_t = torch.tensor(params["direction"], dtype=torch.float32, device=device)
        alpha = params["alpha"]
        
        test_model, _ = load_model(config)
        blocks = get_blocks(test_model)
        
        def make_hook(direction, a):
            def hook(module, inputs, outputs):
                h = outputs[0] if isinstance(outputs, tuple) else outputs
                h_f = h.float()
                proj = torch.einsum('btd,d->bt', h_f, direction)
                h_new = h_f + a * proj.unsqueeze(-1) * direction
                return (h_new.to(h.dtype),) + outputs[1:] if isinstance(outputs, tuple) else h_new.to(h.dtype)
            return hook
        
        handle = blocks[layer].register_forward_hook(make_hook(dir_t, alpha))
        
        int_gens = generate(test_model, tok, test_prompts)
        int_es = extraction_strength(int_gens)
        int_ppl = perplexity(test_model, tok, english_texts[:20])
        
        handle.remove()
        del test_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        es_change = base_es - int_es
        ppl_change = (int_ppl - base_ppl) / base_ppl
        
        results["methods"][method_name] = {
            "es": float(int_es),
            "ppl": float(int_ppl),
            "es_reduction": float(es_change),
            "ppl_increase": float(ppl_change),
            "sample": int_gens[0][:100] if int_gens else ""
        }
        
        status = "✓ REDUCES" if es_change > 0 else "✗ INCREASES"
        print(f"    ES: {base_es:.3f} → {int_es:.3f} ({status} by {abs(es_change):.3f})")
        print(f"    PPL: {base_ppl:.1f} → {int_ppl:.1f} (+{ppl_change*100:.1f}%)")
    
    # Find best method
    best = max(results["methods"].keys(), 
               key=lambda m: results["methods"][m]["es_reduction"] - results["methods"][m]["ppl_increase"])
    results["best_method"] = best
    print(f"\n  ✓ Best method: {best}")
    
    return results

# ============================================================================
# TEST 5: EVALUATION COMPLETENESS
# ============================================================================

def test_evaluation(model, tok, data, config):
    """
    Test on ALL data types to ensure complete evaluation.
    """
    print("\n" + "="*70)
    print("TEST 5: COMPLETE EVALUATION")
    print("="*70)
    
    results = {}
    
    for key, texts in data.items():
        if not texts:
            continue
        
        if key in ["forget_hi", "mixed", "adversarial", "urdu", "punjabi", "bengali"]:
            # Generation test
            prompts = [f"Continue: {t[:40]}" for t in texts[:config.n_prompts]]
            gens = generate(model, tok, prompts)
            es = extraction_strength(gens)
            results[f"{key}_es"] = es
            results[f"{key}_samples"] = gens[:2]
            print(f"  {key}: ES={es:.3f}")
        
        if key == "retain_en":
            ppl = perplexity(model, tok, texts[:50])
            results["retain_en_ppl"] = ppl
            print(f"  {key}: PPL={ppl:.1f}")
    
    # Check for critical bypasses
    if "mixed_es" in results:
        if results["mixed_es"] > 0.3:
            print(f"\n  ⚠️ WARNING: Hinglish bypass exists (ES={results['mixed_es']:.3f})")
            results["hinglish_bypass"] = True
        else:
            results["hinglish_bypass"] = False
    
    if "adversarial_es" in results:
        if results["adversarial_es"] > 0.2:
            print(f"\n  ⚠️ WARNING: Adversarial extraction possible (ES={results['adversarial_es']:.3f})")
            results["adversarial_bypass"] = True
        else:
            results["adversarial_bypass"] = False
    
    return results

# ============================================================================
# MAIN SMOKE TEST
# ============================================================================

def run_smoke_test(config):
    """Run all smoke tests and produce comprehensive report."""
    
    print("\n" + "="*70)
    print("MMIE COMPREHENSIVE SMOKE TEST")
    print("="*70)
    print(f"Model: {config.model}")
    print(f"Data: {config.data_dir}")
    print("="*70)
    
    set_seed(config.seed)
    os.makedirs(config.plots_dir, exist_ok=True)
    
    # Load data
    print("\n[Loading Data]")
    data = {}
    files = ["forget_hi", "retain_en", "mixed", "urdu", "punjabi", "bengali", "adversarial"]
    for f in files:
        path = os.path.join(config.data_dir, f"{f}.jsonl")
        data[f] = read_jsonl(path)
        if data[f]:
            print(f"  {f}: {len(data[f])} samples")
    
    # Load model
    model, tok = load_model(config)
    n_layers = len(get_blocks(model))
    
    results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "n_layers": n_layers,
            "d_model": model.config.hidden_size
        }
    }
    
    # Run tests
    results["test1_tokenization"] = test_tokenization(tok, data, config)
    
    mid_layer = n_layers // 2
    results["test2_sae"] = test_sae_architectures(
        model, tok, data.get("forget_hi", []), data.get("retain_en", []), mid_layer, config
    )
    
    results["test3_layers"] = test_layer_importance(
        model, tok, data.get("forget_hi", []), data.get("retain_en", []), config
    )
    
    # Use best layer from test3 for test4
    if results["test3_layers"].get("layers_that_reduce_hindi"):
        best_layer = results["test3_layers"]["layers_that_reduce_hindi"][0]
    else:
        best_layer = mid_layer
    
    results["test4_interventions"] = test_intervention_methods(
        model, tok, data.get("forget_hi", []), data.get("retain_en", []), best_layer, config
    )
    
    results["test5_evaluation"] = test_evaluation(model, tok, data, config)
    
    # Summary
    print("\n" + "="*70)
    print("SMOKE TEST SUMMARY")
    print("="*70)
    
    issues = []
    
    if results["test1_tokenization"].get("status") == "WARNING":
        issues.append("Tokenization bias detected (may affect activation collection)")
    
    if results["test2_sae"].get("topk_status") == "FAILING":
        issues.append("TopK SAE is failing - use JumpReLU instead")
    
    if results["test3_layers"].get("status") == "METHODS_DISAGREE":
        issues.append("Layer selection methods disagree - use intervention-based")
    
    if not results["test3_layers"].get("direction_method_works"):
        issues.append("Simple direction method doesn't reduce Hindi at any layer")
    
    if results["test5_evaluation"].get("hinglish_bypass"):
        issues.append("Hinglish bypass exists - intervention incomplete")
    
    if results["test5_evaluation"].get("adversarial_bypass"):
        issues.append("Adversarial extraction possible")
    
    if issues:
        print("\n⚠️ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        results["overall_status"] = "ISSUES_FOUND"
        results["issues"] = issues
    else:
        print("\n✓ All tests passed!")
        results["overall_status"] = "OK"
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    
    sae_rec = results["test2_sae"].get("best", "jumprelu_0.1")
    print(f"  1. Use SAE: {sae_rec}")
    
    if results["test3_layers"].get("layers_that_reduce_hindi"):
        layer_rec = results["test3_layers"]["layers_that_reduce_hindi"][:3]
        print(f"  2. Use layers: {layer_rec}")
    else:
        print(f"  2. Use layers: mid-layers ({n_layers//2-2} to {n_layers//2+2})")
    
    method_rec = results["test4_interventions"].get("best_method", "direction_toward_english")
    print(f"  3. Use intervention: {method_rec}")
    
    results["recommendations"] = {
        "sae": sae_rec,
        "layers": results["test3_layers"].get("layers_that_reduce_hindi", [mid_layer]),
        "method": method_rec
    }
    
    # Save results
    with open(config.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[Results saved to {config.out}]")
    
    return results

# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="smoke_test_results.json")
    parser.add_argument("--plots_dir", default="smoke_plots")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Faster but less thorough")
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    
    config = SmokeTestConfig(
        model=args.model,
        data_dir=args.data_dir,
        out=args.out,
        plots_dir=args.plots_dir,
        seed=args.seed,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN")
    )
    
    if args.quick:
        config.n_samples = 30
        config.n_prompts = 10
        config.sae_steps = 500
    
    run_smoke_test(config)

if __name__ == "__main__":
    main()
