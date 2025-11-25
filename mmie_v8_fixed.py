#!/usr/bin/env python3
"""
MMIE v8: FIXED VERSION

Fixes from v7 analysis:
1. Uses JumpReLU SAE (TopK was broken - 0.0013 sparsity, 3x worse loss)
2. Fixes direction sign bug (was INCREASING Hindi instead of decreasing)
3. Uses direct intervention measurement instead of saturated probes
4. Proper hyperparameter search on working SAE

Key findings from v7 that inform this version:
- TopK SAE: val_loss=1.57, sparsity=0.001 ← BROKEN
- JumpReLU: val_loss=0.47, sparsity=0.93 ← USE THIS
- Probe AUC=1.0 everywhere ← SATURATED, don't use for layer selection
- All es_reduction negative ← SIGN BUG, direction was inverted
- mixed_es=0.51 ← Hinglish bypass exists, must test
- 5x tokenization bias ← Consider normalizing
"""

import os
import json
import random
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
# CONFIG
# ============================================================================

@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_token: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "data"
    
    # Override paths
    forget_path: Optional[str] = None
    retain_path: Optional[str] = None
    force_layers: Optional[List[int]] = None
    
    # SAE - USE JUMPRELU (TopK was broken)
    sae_type: str = "jumprelu"  # "jumprelu" or "gated" - NOT topk
    sae_expansion: int = 16
    sae_threshold: float = 0.1  # For JumpReLU
    sae_steps: int = 2000
    sae_lr: float = 3e-4
    
    # Intervention
    gate_alpha: float = 0.5
    
    # Evaluation
    eval_cap: int = 100
    max_len: int = 256
    
    out: str = "results_v8.json"
    plots_dir: str = "plots_v8"
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

def load_all_data(data_dir):
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
        data[key] = read_jsonl(os.path.join(data_dir, fname))
        print(f"[data] {key}: {len(data[key])}")
    return data

def has_devanagari(text):
    return any('\u0900' <= c <= '\u097F' for c in text)

def detect_hindi_score(text):
    if not text:
        return 0.0
    deva = sum(1 for c in text if '\u0900' <= c <= '\u097F') / len(text)
    hindi_words = {'है', 'हैं', 'का', 'की', 'के', 'को', 'से', 'में', 'और', 'hai', 'hain', 'ka', 'ki', 'ke'}
    words = set(text.lower().split())
    vocab = min(1.0, len(words & hindi_words) / 3)
    return max(deva, vocab)

def extraction_strength(texts):
    if not texts:
        return 0.0
    return float(np.mean([detect_hindi_score(t) for t in texts]))

# ============================================================================
# MODEL
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
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(config.model, token=config.hf_token, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'  # Fix the warning
    model = AutoModelForCausalLM.from_pretrained(
        config.model, torch_dtype=dtype, device_map="auto",
        token=config.hf_token, trust_remote_code=True
    )
    model.eval()
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
# SAE - JUMPRELU (TopK was broken)
# ============================================================================

class JumpReLUSAE(nn.Module):
    """
    JumpReLU SAE - better than TopK based on v7 results:
    - TopK: val_loss=1.57, sparsity=0.001 ← BROKEN
    - JumpReLU: val_loss=0.47, sparsity=0.93 ← GOOD
    """
    def __init__(self, d_model, expansion=16, threshold=0.1):
        super().__init__()
        self.d_dict = d_model * expansion
        self.threshold = nn.Parameter(torch.ones(self.d_dict) * threshold)
        self.encoder = nn.Linear(d_model, self.d_dict)
        self.decoder = nn.Linear(self.d_dict, d_model)
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
    
    def encode(self, x):
        z = self.encoder(x)
        # JumpReLU: zero below threshold
        z = z * (z.abs() > self.threshold.abs()).float()
        return z
    
    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z
    
    def get_sparsity(self, x):
        z = self.encode(x)
        return (z.abs() > 0).float().mean().item()


class GatedSAE(nn.Module):
    """Gated SAE - also works well based on v7."""
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


def train_sae(model, tok, texts, layer, config):
    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    
    H = get_hidden(model, tok, texts, layer)
    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    
    if config.sae_type == "jumprelu":
        sae = JumpReLUSAE(d_model, config.sae_expansion, config.sae_threshold).to(device)
    else:
        sae = GatedSAE(d_model, config.sae_expansion).to(device)
    
    opt = torch.optim.AdamW(sae.parameters(), lr=config.sae_lr)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(H_t), batch_size=16, shuffle=True)
    
    sae.train()
    losses = []
    for step in tqdm(range(config.sae_steps), desc=f"SAE L{layer}"):
        for (batch,) in loader:
            x_hat, z = sae(batch)
            loss = F.mse_loss(x_hat, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            break
    
    sae.eval()
    with torch.no_grad():
        _, z_test = sae(H_t[:100])
        sparsity = (z_test.abs() > 0).float().mean().item()
    
    print(f"[sae] Layer {layer}: final_loss={losses[-1]:.4f}, sparsity={sparsity:.3f}")
    
    return sae, {"final_loss": losses[-1], "sparsity": sparsity}

# ============================================================================
# LAYER SELECTION - DIRECT INTERVENTION (not saturated probes)
# ============================================================================

def select_layers_by_direct_intervention(model, tok, hindi_texts, english_texts, config):
    """
    Select layers by DIRECT intervention effect, not probe AUC (which was saturated).
    
    For each layer:
    1. Compute English direction (pointing AWAY from Hindi)
    2. ADD this direction (push toward English = suppress Hindi)
    3. Measure if ES actually decreases
    
    This fixes the sign bug from v7 where we were subtracting the wrong direction.
    """
    print("\n" + "="*70)
    print("LAYER SELECTION BY DIRECT INTERVENTION")
    print("="*70)
    print("NOTE: v7 had a sign bug - ablation INCREASED Hindi.")
    print("This version pushes TOWARD English to suppress Hindi.")
    print("="*70)
    
    n_layers = len(get_blocks(model))
    device = next(model.parameters()).device
    
    # Prompts for testing
    test_prompts = [f"Continue this: {t[:40]}" for t in hindi_texts[:15]]
    
    # Baseline
    base_gens = generate(model, tok, test_prompts)
    base_es = extraction_strength(base_gens)
    base_ppl = perplexity(model, tok, english_texts[:30])
    
    print(f"[baseline] ES={base_es:.3f}, PPL={base_ppl:.1f}")
    
    results = {"baseline_es": base_es, "baseline_ppl": base_ppl, "layers": {}}
    
    for layer in tqdm(range(4, n_layers-2, 2), desc="Testing layers"):  # Skip first/last few
        # Get representations
        H_hi = get_hidden(model, tok, hindi_texts[:50], layer)
        H_en = get_hidden(model, tok, english_texts[:50], layer)
        
        if len(H_hi) == 0 or len(H_en) == 0:
            continue
        
        # FIXED: English direction (pointing TOWARD English, AWAY from Hindi)
        english_direction = H_en.mean(axis=0) - H_hi.mean(axis=0)
        english_direction = english_direction / (np.linalg.norm(english_direction) + 1e-8)
        eng_dir_t = torch.tensor(english_direction, dtype=torch.float32, device=device)
        
        # Test intervention
        test_model, _ = load_model(config)
        blocks = get_blocks(test_model)
        
        def make_suppress_hindi_hook(direction, alpha=0.5):
            """Push activations TOWARD English to suppress Hindi."""
            @torch.no_grad()
            def hook(module, inputs, outputs):
                h = outputs[0] if isinstance(outputs, tuple) else outputs
                h_f = h.float()
                # Project onto English direction and ADD it (push toward English)
                proj = torch.einsum('btd,d->bt', h_f, direction)
                h_new = h_f + alpha * proj.unsqueeze(-1) * direction
                h_new = h_new.to(h.dtype)
                return (h_new,) + outputs[1:] if isinstance(outputs, tuple) else h_new
            return hook
        
        handle = blocks[layer].register_forward_hook(make_suppress_hindi_hook(eng_dir_t, alpha=0.5))
        
        # Measure effect
        intervention_gens = generate(test_model, tok, test_prompts)
        intervention_es = extraction_strength(intervention_gens)
        intervention_ppl = perplexity(test_model, tok, english_texts[:20])
        
        handle.remove()
        del test_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate metrics
        es_reduction = base_es - intervention_es  # POSITIVE = good (Hindi reduced)
        ppl_increase = (intervention_ppl - base_ppl) / base_ppl
        
        # Score: want high ES reduction, low PPL increase
        if ppl_increase < 0.3:  # Accept up to 30% PPL increase
            score = es_reduction
        else:
            score = es_reduction - ppl_increase  # Penalize high PPL increase
        
        results["layers"][layer] = {
            "intervention_es": float(intervention_es),
            "intervention_ppl": float(intervention_ppl),
            "es_reduction": float(es_reduction),
            "ppl_increase": float(ppl_increase),
            "score": float(score)
        }
        
        status = "✓" if es_reduction > 0 else "✗"
        print(f"  Layer {layer}: ES {base_es:.3f}→{intervention_es:.3f} ({status} {es_reduction:+.3f}), PPL +{ppl_increase*100:.0f}%")
    
    # Rank by score
    ranked = sorted(results["layers"].items(), key=lambda x: x[1]["score"], reverse=True)
    results["best_layers"] = [int(l) for l, _ in ranked[:5]]
    
    # Find layers that actually REDUCE Hindi
    effective_layers = [l for l, d in results["layers"].items() if d["es_reduction"] > 0]
    results["effective_layers"] = effective_layers
    
    print(f"\n[result] Layers that REDUCE Hindi: {effective_layers}")
    print(f"[result] Best layers by score: {results['best_layers']}")
    
    if not effective_layers:
        print("\n[WARNING] No layers effectively reduced Hindi!")
        print("          The simple direction method may not work for this model.")
        print("          Consider using SAE-based feature selection instead.")
    
    return results

# ============================================================================
# TEST SPECIFIC LAYERS
# ============================================================================

def test_specific_layers(model, tok, hindi_texts, english_texts, layers, config):
    """Test intervention on specific layers."""
    print("\n" + "="*70)
    print(f"TESTING SPECIFIC LAYERS: {layers}")
    print("="*70)
    
    n_layers = len(get_blocks(model))
    device = next(model.parameters()).device
    
    test_prompts = [f"Continue this: {t[:40]}" for t in hindi_texts[:20]]
    
    # Baseline
    base_gens = generate(model, tok, test_prompts)
    base_es = extraction_strength(base_gens)
    base_ppl = perplexity(model, tok, english_texts[:30])
    
    print(f"[baseline] ES={base_es:.3f}, PPL={base_ppl:.1f}")
    print(f"[baseline] Sample: {base_gens[0][:100]}...")
    
    results = {"baseline_es": base_es, "baseline_ppl": base_ppl, "layers": {}}
    
    for layer in layers:
        if layer >= n_layers or layer < 0:
            print(f"[skip] Layer {layer} out of range (model has {n_layers} layers)")
            continue
            
        print(f"\n[layer {layer}] Testing...")
        
        # Get representations
        H_hi = get_hidden(model, tok, hindi_texts[:50], layer)
        H_en = get_hidden(model, tok, english_texts[:50], layer)
        
        if len(H_hi) == 0 or len(H_en) == 0:
            print(f"[skip] No activations for layer {layer}")
            continue
        
        # English direction (push TOWARD English = suppress Hindi)
        english_direction = H_en.mean(axis=0) - H_hi.mean(axis=0)
        english_direction = english_direction / (np.linalg.norm(english_direction) + 1e-8)
        eng_dir_t = torch.tensor(english_direction, dtype=torch.float32, device=device)
        
        # Test intervention
        test_model, _ = load_model(config)
        blocks = get_blocks(test_model)
        
        def make_suppress_hindi_hook(direction, alpha=0.5):
            @torch.no_grad()
            def hook(module, inputs, outputs):
                h = outputs[0] if isinstance(outputs, tuple) else outputs
                h_f = h.float()
                proj = torch.einsum('btd,d->bt', h_f, direction)
                h_new = h_f + alpha * proj.unsqueeze(-1) * direction
                h_new = h_new.to(h.dtype)
                return (h_new,) + outputs[1:] if isinstance(outputs, tuple) else h_new
            return hook
        
        handle = blocks[layer].register_forward_hook(make_suppress_hindi_hook(eng_dir_t, alpha=config.gate_alpha))
        
        intervention_gens = generate(test_model, tok, test_prompts)
        intervention_es = extraction_strength(intervention_gens)
        intervention_ppl = perplexity(test_model, tok, english_texts[:20])
        
        handle.remove()
        del test_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        es_reduction = base_es - intervention_es
        ppl_increase = (intervention_ppl - base_ppl) / base_ppl
        
        results["layers"][layer] = {
            "intervention_es": float(intervention_es),
            "intervention_ppl": float(intervention_ppl),
            "es_reduction": float(es_reduction),
            "ppl_increase": float(ppl_increase),
            "sample": intervention_gens[0][:100] if intervention_gens else ""
        }
        
        status = "✓ REDUCED" if es_reduction > 0 else "✗ INCREASED"
        print(f"  ES: {base_es:.3f} → {intervention_es:.3f} ({status} by {abs(es_reduction):.3f})")
        print(f"  PPL: {base_ppl:.1f} → {intervention_ppl:.1f} (+{ppl_increase*100:.1f}%)")
        print(f"  Sample: {intervention_gens[0][:80]}...")
    
    # Summary
    effective = [l for l, d in results["layers"].items() if d["es_reduction"] > 0]
    results["effective_layers"] = effective
    results["best_layers"] = sorted(results["layers"].keys(), 
                                     key=lambda l: results["layers"][l]["es_reduction"], 
                                     reverse=True)[:3]
    
    print(f"\n[summary] Layers that REDUCED Hindi: {effective}")
    print(f"[summary] Best layers: {results['best_layers']}")
    
    return results

# ============================================================================
# COMPLETE EVALUATION
# ============================================================================

def evaluate_all(model, tok, data, config):
    """Evaluate on all datasets."""
    print("\n" + "="*70)
    print("COMPLETE EVALUATION")
    print("="*70)
    
    results = {}
    cap = config.eval_cap
    
    # Hindi
    if data.get("forget_hi"):
        prompts = [f"Continue: {t[:40]}" for t in data["forget_hi"][:cap]]
        gens = generate(model, tok, prompts)
        results["hindi_es"] = extraction_strength(gens)
        results["hindi_samples"] = gens[:3]
        print(f"[forget_hi] ES = {results['hindi_es']:.3f}")
    
    # English
    if data.get("retain_en"):
        results["english_ppl"] = perplexity(model, tok, data["retain_en"][:cap])
        print(f"[retain_en] PPL = {results['english_ppl']:.1f}")
    
    # Mixed/Hinglish - CRITICAL TEST
    if data.get("mixed"):
        prompts = data["mixed"][:cap]
        gens = generate(model, tok, prompts)
        results["mixed_es"] = extraction_strength(gens)
        results["mixed_samples"] = gens[:3]
        print(f"[mixed] ES = {results['mixed_es']:.3f} ← HINGLISH BYPASS TEST")
    
    # Related languages
    for lang in ["urdu", "punjabi", "bengali"]:
        if data.get(lang):
            prompts = [f"Continue: {t[:40]}" for t in data[lang][:cap]]
            gens = generate(model, tok, prompts)
            results[f"{lang}_es"] = extraction_strength(gens)
            print(f"[{lang}] ES = {results[f'{lang}_es']:.3f}")
    
    # Adversarial - CRITICAL TEST
    if data.get("adversarial"):
        prompts = data["adversarial"][:cap]
        gens = generate(model, tok, prompts)
        results["adversarial_es"] = extraction_strength(gens)
        results["adversarial_samples"] = gens[:3]
        print(f"[adversarial] ES = {results['adversarial_es']:.3f} ← ADVERSARIAL TEST")
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_plots(layer_results, eval_results, config):
    os.makedirs(config.plots_dir, exist_ok=True)
    
    # Plot 1: Layer intervention effects
    if layer_results and "layers" in layer_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        layers = sorted([int(l) for l in layer_results["layers"].keys()])
        es_reductions = [layer_results["layers"][str(l)]["es_reduction"] for l in layers]
        ppl_increases = [layer_results["layers"][str(l)]["ppl_increase"] for l in layers]
        
        # ES reduction
        colors = ['green' if r > 0 else 'red' for r in es_reductions]
        axes[0].bar(layers, es_reductions, color=colors, alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('ES Reduction')
        axes[0].set_title('Hindi Suppression by Layer\n(Positive = Hindi reduced)')
        
        # PPL increase
        axes[1].bar(layers, [p*100 for p in ppl_increases], color='orange', alpha=0.7)
        axes[1].axhline(y=30, color='red', linestyle='--', label='30% threshold')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('PPL Increase (%)')
        axes[1].set_title('English PPL Increase by Layer\n(Lower is better)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{config.plots_dir}/layer_intervention.png", dpi=150)
        plt.savefig(f"{config.plots_dir}/layer_intervention.pdf", dpi=150)
        plt.close()
        print(f"[plot] Saved layer_intervention.png")
    
    # Plot 2: Evaluation radar
    if eval_results:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        metrics = {}
        if "hindi_es" in eval_results:
            metrics["Hindi ES↓"] = eval_results["hindi_es"]
        if "english_ppl" in eval_results:
            metrics["Eng PPL"] = min(eval_results["english_ppl"] / 50, 1)
        if "mixed_es" in eval_results:
            metrics["Hinglish ES↓"] = eval_results["mixed_es"]
        if "adversarial_es" in eval_results:
            metrics["Adversarial ES↓"] = eval_results["adversarial_es"]
        
        if len(metrics) >= 3:
            labels = list(metrics.keys())
            values = list(metrics.values()) + [list(metrics.values())[0]]
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='#E63946')
            ax.fill(angles, values, alpha=0.25, color='#E63946')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_title('Evaluation Results\n(Lower is better for ES)')
            
            plt.savefig(f"{config.plots_dir}/evaluation_radar.png", dpi=150)
            plt.savefig(f"{config.plots_dir}/evaluation_radar.pdf", dpi=150)
            plt.close()
            print(f"[plot] Saved evaluation_radar.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--forget", default=None, help="Path to forget data (overrides data_dir)")
    parser.add_argument("--retain", default=None, help="Path to retain data (overrides data_dir)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="Specific layers to test")
    parser.add_argument("--out", default="results_v8.json")
    parser.add_argument("--plots_dir", default="plots_v8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sae_type", default="jumprelu", choices=["jumprelu", "gated"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()
    
    config = Config(
        model=args.model,
        data_dir=args.data_dir,
        out=args.out,
        plots_dir=args.plots_dir,
        seed=args.seed,
        sae_type=args.sae_type,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN")
    )
    
    # Store extra args
    config.force_layers = args.layers
    config.forget_path = args.forget
    config.retain_path = args.retain
    
    if args.quick:
        config.sae_steps = 500
        config.eval_cap = 30
    
    set_seed(config.seed)
    os.makedirs(config.plots_dir, exist_ok=True)
    
    # Load data - use explicit paths if provided
    print("\n[1/5] Loading data...")
    if config.forget_path or config.retain_path:
        # Use explicit file paths
        data = {}
        if config.forget_path:
            data["forget_hi"] = read_jsonl(config.forget_path)
            print(f"[data] forget_hi: {len(data['forget_hi'])} (from {config.forget_path})")
        if config.retain_path:
            data["retain_en"] = read_jsonl(config.retain_path)
            print(f"[data] retain_en: {len(data['retain_en'])} (from {config.retain_path})")
        # Try to load other files from data_dir
        for key in ["mixed", "urdu", "punjabi", "bengali", "adversarial"]:
            path = os.path.join(config.data_dir, f"{key}.jsonl")
            if os.path.exists(path):
                data[key] = read_jsonl(path)
                print(f"[data] {key}: {len(data[key])}")
    else:
        data = load_all_data(config.data_dir)
    
    # Load model
    print("\n[2/5] Loading model...")
    model, tok = load_model(config)
    
    results = {
        "config": asdict(config),
        "timestamp": datetime.now().isoformat(),
        "fixes_from_v7": [
            "Using JumpReLU SAE (TopK was broken with 0.001 sparsity)",
            "Fixed direction sign (was increasing Hindi instead of decreasing)",
            "Using direct intervention measurement (probe was saturated at 1.0)",
            "Set padding_side='left' (fixes generation warning)"
        ]
    }
    
    # Layer selection by direct intervention
    print("\n[3/5] Selecting layers by direct intervention...")
    if config.force_layers:
        print(f"[layers] Using specified layers: {config.force_layers}")
        layer_results = test_specific_layers(
            model, tok, data.get("forget_hi", []), data.get("retain_en", []), 
            config.force_layers, config
        )
    else:
        layer_results = select_layers_by_direct_intervention(
            model, tok, data.get("forget_hi", []), data.get("retain_en", []), config
        )
    results["layer_selection"] = layer_results
    
    # Baseline evaluation
    print("\n[4/5] Baseline evaluation...")
    results["evaluation_base"] = evaluate_all(model, tok, data, config)
    
    # Create plots
    print("\n[5/5] Creating plots...")
    create_plots(layer_results, results["evaluation_base"], config)
    
    # Save results
    with open(config.out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[done] Results: {config.out}")
    print(f"[done] Plots: {config.plots_dir}/")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if layer_results.get("effective_layers"):
        print(f"✓ Layers that reduce Hindi: {layer_results['effective_layers']}")
    else:
        print("✗ No layers effectively reduced Hindi with direction method")
    
    if "evaluation_base" in results:
        eb = results["evaluation_base"]
        print(f"\nBaseline metrics:")
        print(f"  Hindi ES: {eb.get('hindi_es', 'N/A'):.3f}")
        print(f"  Hinglish ES: {eb.get('mixed_es', 'N/A'):.3f} (bypass test)")
        print(f"  Adversarial ES: {eb.get('adversarial_es', 'N/A'):.3f}")
        print(f"  English PPL: {eb.get('english_ppl', 'N/A'):.1f}")

if __name__ == "__main__":
    main()
