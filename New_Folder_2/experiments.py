"""
Experiments Module
==================
All experiment implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from scipy import stats
from collections import defaultdict

from model_manager import (
    compute_hindi_score, compute_english_score, 
    compute_all_language_scores, compute_urdu_score,
    compute_bengali_score, compute_punjabi_score
)
from data_loader import get_prompt_text, COHERENCE_TEST_CASES


# ==============================================================================
# EXPERIMENT 1: Semantic Subspace Analysis
# ==============================================================================

def run_semantic_subspace_analysis(model_manager, data, config) -> Dict[str, Any]:
    """
    Test: Do parallel sentences (same meaning, different language) 
    have similar representations?
    
    This tests the "common semantic subspace" hypothesis.
    """
    print("  Testing semantic subspace hypothesis...")
    
    results = {
        "layer_similarities": {},
        "language_classification": {},
        "interpretation": {}
    }
    
    # For each layer, compute similarity between parallel sentences
    for layer_idx in tqdm(range(model_manager.num_layers), desc="  Layers"):
        
        parallel_sims = []
        for en_text, hi_text in data.parallel:
            en_act = model_manager.get_activation(en_text, layer_idx)
            hi_act = model_manager.get_activation(hi_text, layer_idx)
            
            sim = F.cosine_similarity(en_act.unsqueeze(0), hi_act.unsqueeze(0)).item()
            parallel_sims.append(sim)
        
        # Control: compare different-meaning sentences
        different_sims = []
        for i in range(len(data.parallel) - 1):
            en1 = data.parallel[i][0]
            hi2 = data.parallel[i + 1][1]
            
            en_act = model_manager.get_activation(en1, layer_idx)
            hi_act = model_manager.get_activation(hi2, layer_idx)
            
            sim = F.cosine_similarity(en_act.unsqueeze(0), hi_act.unsqueeze(0)).item()
            different_sims.append(sim)
        
        results["layer_similarities"][layer_idx] = {
            "parallel_mean": float(np.mean(parallel_sims)),
            "parallel_std": float(np.std(parallel_sims)),
            "different_mean": float(np.mean(different_sims)),
            "different_std": float(np.std(different_sims)),
            "gap": float(np.mean(parallel_sims) - np.mean(different_sims)),
        }
        
        # Language classification (can we tell Hindi from English?)
        from sklearn.linear_model import LogisticRegression
        
        en_acts = [model_manager.get_activation(t[0], layer_idx).cpu().numpy() for t in data.parallel]
        hi_acts = [model_manager.get_activation(t[1], layer_idx).cpu().numpy() for t in data.parallel]
        
        X = np.vstack(en_acts + hi_acts)
        y = np.array([0] * len(en_acts) + [1] * len(hi_acts))
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        
        results["language_classification"][layer_idx] = float(acc)
    
    # Find best layer for semantic similarity
    best_layer = max(
        results["layer_similarities"].keys(),
        key=lambda k: results["layer_similarities"][k]["gap"]
    )
    
    results["interpretation"] = {
        "best_semantic_layer": best_layer,
        "best_gap": results["layer_similarities"][best_layer]["gap"],
        "semantic_subspace_exists": results["layer_similarities"][best_layer]["gap"] > 0.1,
        "language_always_separable": all(v > 0.9 for v in results["language_classification"].values()),
    }
    
    print(f"    Best semantic layer: {best_layer} (gap={results['interpretation']['best_gap']:.3f})")
    print(f"    Semantic subspace exists: {results['interpretation']['semantic_subspace_exists']}")
    
    return results


# ==============================================================================
# EXPERIMENT 2: Steering Grid Search
# ==============================================================================

def run_steering_grid_search(model_manager, data, config) -> Dict[str, Any]:
    """
    Test: Which layer and coefficient combination works best for steering?
    """
    print("  Running steering grid search...")
    
    # Get source/target texts for direction
    hindi_texts = [get_prompt_text(item) for item in data.hindi[:10]]
    english_texts = [get_prompt_text(item) for item in data.english[:10]]
    
    # Test prompts
    test_prompts = [get_prompt_text(item) for item in data.hindi[:config.n_test_prompts]]
    
    results = {
        "grid_results": {},
        "best_combinations": [],
    }
    
    for layer_idx in tqdm(config.layers, desc="  Layers"):
        # Compute direction at this layer
        direction = model_manager.compute_steering_direction(
            hindi_texts, english_texts, layer_idx
        )
        
        results["grid_results"][layer_idx] = {}
        
        for coeff in config.coefficients:
            successes = 0
            deltas = []
            
            for prompt in test_prompts:
                # Baseline
                baseline = model_manager.generate(prompt, max_new_tokens=30)
                baseline_hindi = compute_hindi_score(baseline)
                
                # Steered
                steered = model_manager.generate(
                    prompt,
                    max_new_tokens=30,
                    steering_layer=layer_idx,
                    steering_direction=direction,
                    steering_coeff=coeff
                )
                steered_hindi = compute_hindi_score(steered)
                
                delta = baseline_hindi - steered_hindi
                deltas.append(delta)
                if delta > 0.05:
                    successes += 1
            
            results["grid_results"][layer_idx][coeff] = {
                "success_rate": successes / len(test_prompts),
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
            }
    
    # Find best combinations
    all_combos = []
    for layer, coeffs in results["grid_results"].items():
        for coeff, metrics in coeffs.items():
            all_combos.append({
                "layer": layer,
                "coeff": coeff,
                "success_rate": metrics["success_rate"],
                "mean_delta": metrics["mean_delta"],
            })
    
    all_combos.sort(key=lambda x: (x["success_rate"], x["mean_delta"]), reverse=True)
    results["best_combinations"] = all_combos[:5]
    
    print(f"    Best: Layer {results['best_combinations'][0]['layer']}, "
          f"coeff={results['best_combinations'][0]['coeff']}")
    
    return results


# ==============================================================================
# EXPERIMENT 3: Causality Test (Activation Patching)
# ==============================================================================

def run_causality_test(model_manager, data, config) -> Dict[str, Any]:
    """
    Test: If we patch English activation into Hindi prompt, 
    does output become English?
    
    This tests CAUSALITY vs correlation.
    """
    print("  Running causality test (activation patching)...")
    
    layer_idx = config.layers[len(config.layers) // 2]  # Use middle layer
    
    results = {
        "layer": layer_idx,
        "pairs": [],
        "summary": {}
    }
    
    for en_text, hi_text in tqdm(data.parallel[:10], desc="  Patching"):
        # Get English activation
        en_activation = model_manager.get_activation(en_text, layer_idx)
        
        # Normal Hindi output
        hi_output_normal = model_manager.generate(hi_text, max_new_tokens=40)
        hi_normal_hindi = compute_hindi_score(hi_output_normal)
        
        # Hindi with English activation patched
        hi_output_patched = model_manager.generate(
            hi_text,
            max_new_tokens=40,
            patch_activation=en_activation,
            patch_layer=layer_idx
        )
        hi_patched_hindi = compute_hindi_score(hi_output_patched)
        
        # English output for reference
        en_output = model_manager.generate(en_text, max_new_tokens=40)
        en_hindi = compute_hindi_score(en_output)
        
        change = hi_normal_hindi - hi_patched_hindi
        
        results["pairs"].append({
            "en_prompt": en_text,
            "hi_prompt": hi_text,
            "en_output_hindi": en_hindi,
            "hi_normal_hindi": hi_normal_hindi,
            "hi_patched_hindi": hi_patched_hindi,
            "change": change,
            "causal": change > 0.3,
            "hi_normal_sample": hi_output_normal[:60],
            "hi_patched_sample": hi_output_patched[:60],
        })
    
    # Summary
    changes = [p["change"] for p in results["pairs"]]
    causal_count = sum(1 for p in results["pairs"] if p["causal"])
    
    results["summary"] = {
        "avg_change": float(np.mean(changes)),
        "causal_pairs": causal_count,
        "total_pairs": len(results["pairs"]),
        "causal_rate": causal_count / len(results["pairs"]),
        "conclusion": "CAUSAL" if np.mean(changes) > 0.3 else "NOT CAUSAL",
    }
    
    print(f"    Avg change: {results['summary']['avg_change']:.3f}")
    print(f"    Conclusion: {results['summary']['conclusion']}")
    
    return results


# ==============================================================================
# EXPERIMENT 4: Coherence Test
# ==============================================================================

def run_coherence_test(model_manager, data, config) -> Dict[str, Any]:
    """
    Test: Does steering preserve answer correctness?
    Find the Goldilocks zone where language changes but meaning preserved.
    """
    print("  Running coherence test...")
    
    # Get direction at best layer
    layer_idx = config.layers[len(config.layers) // 2]
    
    hindi_texts = [get_prompt_text(item) for item in data.hindi[:10]]
    english_texts = [get_prompt_text(item) for item in data.english[:10]]
    direction = model_manager.compute_steering_direction(hindi_texts, english_texts, layer_idx)
    
    results = {
        "layer": layer_idx,
        "test_cases": [],
        "coeff_summary": defaultdict(lambda: {"correct": 0, "total": 0}),
    }
    
    for test_case in tqdm(COHERENCE_TEST_CASES, desc="  Testing"):
        case_results = {
            "prompt_hi": test_case["prompt_hi"],
            "expected": test_case["expected"],
            "coefficients": {}
        }
        
        for coeff in [0] + list(config.coefficients):
            output = model_manager.generate(
                test_case["prompt_hi"],
                max_new_tokens=50,
                steering_layer=layer_idx,
                steering_direction=direction,
                steering_coeff=coeff
            )
            
            hindi_score = compute_hindi_score(output)
            has_answer = test_case["expected"].lower() in output.lower()
            
            case_results["coefficients"][coeff] = {
                "output": output[:80],
                "hindi_score": hindi_score,
                "correct": has_answer,
            }
            
            results["coeff_summary"][coeff]["total"] += 1
            if has_answer:
                results["coeff_summary"][coeff]["correct"] += 1
        
        results["test_cases"].append(case_results)
    
    # Convert defaultdict
    results["coeff_summary"] = dict(results["coeff_summary"])
    
    # Find Goldilocks zone
    goldilocks = None
    for coeff in sorted(results["coeff_summary"].keys()):
        if coeff == 0:
            continue
        accuracy = results["coeff_summary"][coeff]["correct"] / results["coeff_summary"][coeff]["total"]
        
        # Check if this coeff reduces Hindi while maintaining accuracy
        avg_hindi = np.mean([
            tc["coefficients"][coeff]["hindi_score"] 
            for tc in results["test_cases"]
        ])
        
        if accuracy >= 0.6 and avg_hindi < 0.3:
            goldilocks = coeff
            break
    
    results["goldilocks_zone"] = goldilocks
    
    print(f"    Goldilocks zone: coeff={goldilocks}")
    
    return results


# ==============================================================================
# EXPERIMENT 5: Cross-Language Leak Test
# ==============================================================================

def run_cross_language_test(model_manager, data, config) -> Dict[str, Any]:
    """
    Test: When we steer away from Hindi, what happens to related languages?
    (Urdu, Bengali, Punjabi)
    """
    print("  Running cross-language leak test...")
    
    # Get direction
    layer_idx = config.layers[len(config.layers) // 2]
    
    hindi_texts = [get_prompt_text(item) for item in data.hindi[:10]]
    english_texts = [get_prompt_text(item) for item in data.english[:10]]
    direction = model_manager.compute_steering_direction(hindi_texts, english_texts, layer_idx)
    
    # Test on each language
    languages = {
        "hindi": data.hindi,
        "urdu": data.urdu,
        "bengali": data.bengali,
        "punjabi": data.punjabi,
    }
    
    score_funcs = {
        "hindi": compute_hindi_score,
        "urdu": compute_urdu_score,
        "bengali": compute_bengali_score,
        "punjabi": compute_punjabi_score,
    }
    
    results = {
        "layer": layer_idx,
        "coeff": 5.0,  # Use known effective coefficient
        "languages": {},
    }
    
    for lang_name, lang_data in languages.items():
        if not lang_data:
            print(f"    Skipping {lang_name} (no data)")
            continue
        
        print(f"    Testing {lang_name}...")
        
        score_func = score_funcs.get(lang_name, compute_hindi_score)
        
        baseline_scores = []
        steered_scores = []
        
        for item in lang_data[:config.n_test_prompts]:
            prompt = get_prompt_text(item)
            
            # Baseline
            baseline = model_manager.generate(prompt, max_new_tokens=30)
            baseline_score = score_func(baseline)
            baseline_scores.append(baseline_score)
            
            # Steered
            steered = model_manager.generate(
                prompt,
                max_new_tokens=30,
                steering_layer=layer_idx,
                steering_direction=direction,
                steering_coeff=5.0
            )
            steered_score = score_func(steered)
            steered_scores.append(steered_score)
        
        results["languages"][lang_name] = {
            "baseline_mean": float(np.mean(baseline_scores)),
            "steered_mean": float(np.mean(steered_scores)),
            "change": float(np.mean(baseline_scores) - np.mean(steered_scores)),
            "affected": abs(np.mean(baseline_scores) - np.mean(steered_scores)) > 0.1,
        }
    
    # Analyze leaks
    hindi_change = results["languages"].get("hindi", {}).get("change", 0)
    
    leaks = []
    for lang in ["urdu", "bengali", "punjabi"]:
        if lang in results["languages"]:
            other_change = results["languages"][lang]["change"]
            if other_change > 0.1:  # Also decreased
                leaks.append(lang)
    
    results["interpretation"] = {
        "hindi_change": hindi_change,
        "leaking_languages": leaks,
        "clean_separation": len(leaks) == 0,
    }
    
    print(f"    Leaking to: {leaks if leaks else 'None (clean)'}")
    
    return results


# ==============================================================================
# EXPERIMENT 6: SAE Feature Analysis
# ==============================================================================

def run_sae_analysis(model_manager, data, config) -> Dict[str, Any]:
    """
    Test: Can SAE features disentangle language from content?
    Find features that are language-specific.
    """
    print("  Running SAE analysis...")
    
    from sae import JumpReLUSAE, train_sae
    
    layer_idx = config.layers[len(config.layers) // 2]
    
    # Collect activations
    print("    Collecting activations...")
    
    hindi_acts = []
    for item in data.hindi[:config.n_samples_per_lang]:
        prompt = get_prompt_text(item)
        act = model_manager.get_activation(prompt, layer_idx)
        hindi_acts.append(act)
    
    english_acts = []
    for item in data.english[:config.n_samples_per_lang]:
        prompt = get_prompt_text(item)
        act = model_manager.get_activation(prompt, layer_idx)
        english_acts.append(act)
    
    hindi_acts = torch.stack(hindi_acts)
    english_acts = torch.stack(english_acts)
    all_acts = torch.cat([hindi_acts, english_acts], dim=0)
    
    # Train SAE
    print("    Training SAE...")
    sae, sae_metrics = train_sae(
        all_acts,
        hidden_dim=model_manager.hidden_dim,
        sae_dim=config.sae_hidden_dim,
        epochs=config.sae_epochs,
        lr=config.sae_lr,
        k=config.sae_k,
        threshold=config.sae_threshold,
    )
    
    # Find language-specific features
    print("    Finding language-specific features...")
    
    with torch.no_grad():
        hindi_encoded = sae.encode(hindi_acts)
        english_encoded = sae.encode(english_acts)
        
        # Activation frequency per feature
        hindi_freq = (hindi_encoded > 0).float().mean(dim=0)
        english_freq = (english_encoded > 0).float().mean(dim=0)
        
        # Hindi-specific: high on Hindi, low on English
        hindi_specificity = hindi_freq - english_freq
        hindi_features = torch.argsort(hindi_specificity, descending=True)[:50].tolist()
        
        # English-specific
        english_specificity = english_freq - hindi_freq
        english_features = torch.argsort(english_specificity, descending=True)[:50].tolist()
        
        # Monolinguality score
        mono_score = torch.abs(hindi_freq - english_freq) / (hindi_freq + english_freq + 1e-6)
    
    results = {
        "layer": layer_idx,
        "sae_metrics": sae_metrics,
        "hindi_specific_features": hindi_features[:20],
        "english_specific_features": english_features[:20],
        "top_monolinguality_scores": sorted(mono_score.tolist(), reverse=True)[:20],
        "interpretation": {
            "has_language_features": max(mono_score.tolist()) > 0.5,
            "num_highly_specific": sum(1 for s in mono_score if s > 0.5),
        }
    }
    
    print(f"    Found {results['interpretation']['num_highly_specific']} highly language-specific features")
    
    return results


# ==============================================================================
# EXPERIMENT 7: Adversarial Robustness
# ==============================================================================

def run_adversarial_test(model_manager, data, config) -> Dict[str, Any]:
    """
    Test: Can adversarial prompts bypass steering?
    """
    print("  Running adversarial robustness test...")
    
    # Get direction
    layer_idx = config.layers[len(config.layers) // 2]
    
    hindi_texts = [get_prompt_text(item) for item in data.hindi[:10]]
    english_texts = [get_prompt_text(item) for item in data.english[:10]]
    direction = model_manager.compute_steering_direction(hindi_texts, english_texts, layer_idx)
    
    results = {
        "layer": layer_idx,
        "coeff": 5.0,
        "by_type": defaultdict(list),
        "overall": {}
    }
    
    for adv_item in tqdm(data.adversarial, desc="  Adversarial"):
        prompt = adv_item.get("prompt", adv_item.get("text", ""))
        adv_type = adv_item.get("type", "unknown")
        
        # With steering
        output = model_manager.generate(
            prompt,
            max_new_tokens=40,
            steering_layer=layer_idx,
            steering_direction=direction,
            steering_coeff=5.0
        )
        
        hindi_score = compute_hindi_score(output)
        extracted = hindi_score > 0.1  # Hindi appeared despite steering
        
        results["by_type"][adv_type].append({
            "prompt": prompt[:50],
            "output": output[:50],
            "hindi_score": hindi_score,
            "extracted": extracted,
        })
    
    # Convert defaultdict
    results["by_type"] = dict(results["by_type"])
    
    # Summary by type
    type_summary = {}
    total_extracted = 0
    total_prompts = 0
    
    for adv_type, items in results["by_type"].items():
        extracted_count = sum(1 for i in items if i["extracted"])
        type_summary[adv_type] = {
            "total": len(items),
            "extracted": extracted_count,
            "rate": extracted_count / len(items) if items else 0,
        }
        total_extracted += extracted_count
        total_prompts += len(items)
    
    results["type_summary"] = type_summary
    results["overall"] = {
        "total_extracted": total_extracted,
        "total_prompts": total_prompts,
        "extraction_rate": total_extracted / total_prompts if total_prompts else 0,
        "vulnerability": "HIGH" if total_extracted / total_prompts > 0.3 else "MEDIUM" if total_extracted / total_prompts > 0.1 else "LOW",
    }
    
    print(f"    Overall extraction rate: {results['overall']['extraction_rate']:.1%}")
    print(f"    Vulnerability: {results['overall']['vulnerability']}")
    
    return results
