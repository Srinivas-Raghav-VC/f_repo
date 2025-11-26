#!/usr/bin/env python3
"""
MINIMAL DIAGNOSTIC: Visualize Steering Inconsistency
=====================================================

This script shows EXACTLY what's happening:
1. Collects activations for Hindi/English prompts
2. Computes mean-difference direction
3. Tests steering on each prompt
4. Shows which prompts succeed vs fail
5. Plots the "money figure" showing the problem

Run: python diagnose_steering.py
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_jsonl(path: str, max_samples: int = 50) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def compute_hindi_score(text: str) -> float:
    """Fraction of characters that are Devanagari."""
    if not text:
        return 0.0
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total = len(text.replace(' ', '').replace('\n', ''))
    return hindi_chars / max(total, 1)


def main():
    print("=" * 60)
    print("STEERING INCONSISTENCY DIAGNOSTIC")
    print("=" * 60)
    
    # =========================================
    # STEP 1: Load model
    # =========================================
    print("\n[1/5] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"  ✓ Model loaded: {model.config.num_hidden_layers} layers")
    
    # =========================================
    # STEP 2: Load data
    # =========================================
    print("\n[2/5] Loading data...")
    
    # Find data files
    data_dir = Path(".")
    hindi_file = data_dir / "forget_hindi.jsonl"
    english_file = data_dir / "retain_english.jsonl"
    
    if not hindi_file.exists():
        print(f"  ✗ {hindi_file} not found!")
        return
    if not english_file.exists():
        print(f"  ✗ {english_file} not found!")
        return
    
    hindi_data = load_jsonl(hindi_file, max_samples=30)
    english_data = load_jsonl(english_file, max_samples=30)
    
    print(f"  ✓ Loaded {len(hindi_data)} Hindi, {len(english_data)} English prompts")
    
    # =========================================
    # STEP 3: Collect activations
    # =========================================
    print("\n[3/5] Collecting activations at layer 7...")
    
    layer_idx = 7  # Middle layer
    
    def get_activation(text: str) -> torch.Tensor:
        """Get activation at layer 7 for last token."""
        activation = None
        
        def hook(module, input, output):
            nonlocal activation
            hidden = output[0] if isinstance(output, tuple) else output
            activation = hidden[:, -1, :].detach().clone()
        
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)
        
        handle.remove()
        return activation.squeeze(0).float()
    
    # Collect Hindi activations
    hindi_activations = []
    hindi_prompts = []
    for item in tqdm(hindi_data, desc="  Hindi"):
        prompt = item.get("prompt", item.get("text", ""))
        hindi_prompts.append(prompt)
        act = get_activation(prompt)
        hindi_activations.append(act)
    
    # Collect English activations
    english_activations = []
    for item in tqdm(english_data, desc="  English"):
        prompt = item.get("prompt", item.get("text", ""))
        act = get_activation(prompt)
        english_activations.append(act)
    
    hindi_acts = torch.stack(hindi_activations)
    english_acts = torch.stack(english_activations)
    
    print(f"  ✓ Collected activations: Hindi {hindi_acts.shape}, English {english_acts.shape}")
    
    # =========================================
    # STEP 4: Compute steering direction
    # =========================================
    print("\n[4/5] Computing steering direction...")
    
    hindi_mean = hindi_acts.mean(dim=0)
    english_mean = english_acts.mean(dim=0)
    
    # Direction: Hindi → English (we want to REDUCE Hindi)
    steering_direction = F.normalize(english_mean - hindi_mean, dim=0)
    
    # Compute distances of each Hindi prompt from the means
    distances_to_hindi_mean = torch.norm(hindi_acts - hindi_mean, dim=1)
    distances_to_english_mean = torch.norm(hindi_acts - english_mean, dim=1)
    projections = (hindi_acts @ steering_direction)  # Projection onto direction
    
    print(f"  ✓ Direction computed (norm={torch.norm(english_mean - hindi_mean):.2f})")
    print(f"  ✓ Distance to Hindi mean: {distances_to_hindi_mean.mean():.2f} ± {distances_to_hindi_mean.std():.2f}")
    print(f"  ✓ Projection onto direction: {projections.mean():.2f} ± {projections.std():.2f}")
    
    # =========================================
    # STEP 5: Test steering on each prompt
    # =========================================
    print("\n[5/5] Testing steering on each Hindi prompt...")
    
    def generate_with_steering(prompt: str, coeff: float) -> str:
        """Generate text with optional steering."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        hooks = []
        if coeff > 0:
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[:, -1, :] = hidden[:, -1, :] + coeff * steering_direction.to(hidden.device)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            hooks.append(handle)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            for h in hooks:
                h.remove()
        
        return generated
    
    # Test each prompt
    results = []
    steering_coeff = 2.0
    
    for i, prompt in enumerate(tqdm(hindi_prompts, desc="  Testing")):
        # Baseline (no steering)
        baseline_output = generate_with_steering(prompt, coeff=0.0)
        baseline_es = compute_hindi_score(baseline_output)
        
        # With steering
        steered_output = generate_with_steering(prompt, coeff=steering_coeff)
        steered_es = compute_hindi_score(steered_output)
        
        delta = baseline_es - steered_es  # Positive = Hindi reduced (good)
        
        results.append({
            "prompt_idx": i,
            "prompt": prompt[:50],
            "baseline_es": baseline_es,
            "steered_es": steered_es,
            "delta": delta,
            "success": delta > 0,
            "distance_to_mean": distances_to_hindi_mean[i].item(),
            "projection": projections[i].item(),
        })
    
    # =========================================
    # ANALYSIS
    # =========================================
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total prompts: {len(results)}")
    print(f"  Successes (Hindi reduced): {len(successes)} ({100*len(successes)/len(results):.1f}%)")
    print(f"  Failures (Hindi increased): {len(failures)} ({100*len(failures)/len(results):.1f}%)")
    
    if successes:
        print(f"\n✅ Successful prompts:")
        print(f"  Mean baseline ES: {np.mean([r['baseline_es'] for r in successes]):.3f}")
        print(f"  Mean delta: {np.mean([r['delta'] for r in successes]):.3f}")
        print(f"  Mean distance from mean: {np.mean([r['distance_to_mean'] for r in successes]):.3f}")
    
    if failures:
        print(f"\n❌ Failed prompts:")
        print(f"  Mean baseline ES: {np.mean([r['baseline_es'] for r in failures]):.3f}")
        print(f"  Mean delta: {np.mean([r['delta'] for r in failures]):.3f}")
        print(f"  Mean distance from mean: {np.mean([r['distance_to_mean'] for r in failures]):.3f}")
    
    # =========================================
    # THE KEY INSIGHT
    # =========================================
    print("\n" + "=" * 60)
    print("🔑 KEY INSIGHT")
    print("=" * 60)
    
    # Correlation between projection and delta
    projections_list = [r["projection"] for r in results]
    deltas_list = [r["delta"] for r in results]
    
    correlation = np.corrcoef(projections_list, deltas_list)[0, 1]
    
    print(f"\nCorrelation between projection and delta: {correlation:.3f}")
    
    if correlation > 0.3:
        print("→ Prompts that project HIGHER on the direction get BETTER steering")
        print("→ This means prompts 'behind' the mean get pushed the WRONG way")
    elif correlation < -0.3:
        print("→ Negative correlation - unexpected pattern")
    else:
        print("→ Weak correlation - projection doesn't fully explain success")
    
    # =========================================
    # EXAMPLE OUTPUTS
    # =========================================
    print("\n" + "=" * 60)
    print("EXAMPLE OUTPUTS")
    print("=" * 60)
    
    print("\n✅ SUCCESSFUL CASE (Hindi reduced):")
    if successes:
        ex = max(successes, key=lambda x: x["delta"])
        print(f"  Prompt: {ex['prompt']}...")
        print(f"  Baseline ES: {ex['baseline_es']:.3f} → Steered ES: {ex['steered_es']:.3f}")
        print(f"  Delta: {ex['delta']:.3f} (REDUCED Hindi)")
    
    print("\n❌ FAILED CASE (Hindi increased):")
    if failures:
        ex = min(failures, key=lambda x: x["delta"])
        print(f"  Prompt: {ex['prompt']}...")
        print(f"  Baseline ES: {ex['baseline_es']:.3f} → Steered ES: {ex['steered_es']:.3f}")
        print(f"  Delta: {ex['delta']:.3f} (INCREASED Hindi!)")
    
    # =========================================
    # SAVE RESULTS
    # =========================================
    output_file = "steering_diagnostic_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": {
                "total": len(results),
                "successes": len(successes),
                "failures": len(failures),
                "success_rate": len(successes) / len(results),
                "correlation_projection_delta": correlation,
            },
            "per_prompt_results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    # =========================================
    # CREATE VISUALIZATION
    # =========================================
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Baseline ES vs Delta
        colors = ['green' if r['success'] else 'red' for r in results]
        axes[0].scatter(
            [r['baseline_es'] for r in results],
            [r['delta'] for r in results],
            c=colors, alpha=0.7
        )
        axes[0].axhline(y=0, color='black', linestyle='--', label='No change')
        axes[0].set_xlabel('Baseline Hindi Score (ES)')
        axes[0].set_ylabel('Delta (positive = Hindi reduced)')
        axes[0].set_title('THE PROBLEM: Same Steering → Opposite Effects')
        axes[0].legend(['Threshold', 'Success', 'Failure'])
        
        # Plot 2: Projection vs Delta
        axes[1].scatter(
            [r['projection'] for r in results],
            [r['delta'] for r in results],
            c=colors, alpha=0.7
        )
        axes[1].axhline(y=0, color='black', linestyle='--')
        axes[1].set_xlabel('Projection onto Steering Direction')
        axes[1].set_ylabel('Delta (positive = Hindi reduced)')
        axes[1].set_title(f'Correlation: {correlation:.3f}')
        
        plt.tight_layout()
        plt.savefig('steering_diagnostic.png', dpi=150)
        print(f"📊 Plot saved to: steering_diagnostic.png")
        plt.close()
        
    except ImportError:
        print("(matplotlib not available, skipping plot)")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
