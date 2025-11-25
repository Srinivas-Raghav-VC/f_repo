"""
Experiment 2: SAE Features vs Direction-Based Steering
=======================================================

Hypothesis H2: SAE-derived language-specific features provide more consistent
steering than mean-difference vectors.

This experiment:
1. Trains SAEs on activations to extract language-specific features
2. Compares steering methods: mean-difference vs SAE top-K ablation
3. Measures consistency (variance) of steering across prompts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from scipy import stats
from tqdm import tqdm

from .base import BaseExperiment, compute_statistics
from utils.logging_utils import get_logger
from models.model_loader import load_model_and_tokenizer
from models.activation_collector import ActivationCollector
from models.sae import JumpReLUSAE, TopKSAE, train_sae

logger = get_logger(__name__)


def run_experiment_2_sae_vs_direction(config, results_manager) -> Dict[str, Any]:
    """
    Run Experiment 2: SAE vs Direction Comparison
    
    Tests H2: SAE features provide more consistent steering than mean-difference
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT 2: SAE vs Direction-Based Steering")
    logger.info("Testing H2: SAE features are more consistent")
    logger.info("=" * 50)
    
    experiment = SAEvsDirectionExperiment(config, results_manager)
    return experiment.run()


class SAEvsDirectionExperiment(BaseExperiment):
    """Compare SAE-based vs direction-based steering."""
    
    def run(self) -> Dict[str, Any]:
        self.set_seed()
        
        # Load model
        logger.info("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            dtype=self.config.model_dtype
        )
        
        # Load data
        logger.info("Loading data...")
        hindi_data = self.load_data(self.config.forget_file)[:self.config.num_samples]
        english_data = self.load_data(self.config.retain_file)[:self.config.num_samples]
        
        if not hindi_data or not english_data:
            return {"error": "No data loaded"}
        
        results = {
            "hypothesis": "H2: SAE features provide more consistent steering",
            "per_layer_results": {},
            "method_comparison": {},
            "aggregate_results": {},
        }
        
        # Analyze key layers
        for layer_idx in self.config.layers:
            logger.info(f"\n{'=' * 40}")
            logger.info(f"Analyzing Layer {layer_idx}")
            logger.info(f"{'=' * 40}")
            
            layer_results = self.analyze_layer(layer_idx, hindi_data, english_data)
            results["per_layer_results"][f"layer_{layer_idx}"] = layer_results
        
        # Aggregate comparison
        results["aggregate_results"] = self.aggregate_comparison(results["per_layer_results"])
        
        # Generate plots
        self.results_manager.save_sae_comparison_plots(results)
        
        return results
    
    def analyze_layer(self, layer_idx: int, hindi_data: List, english_data: List) -> Dict:
        """Analyze SAE vs direction steering at a specific layer."""
        
        collector = ActivationCollector(self.model, layer_idx)
        
        # Step 1: Collect all activations
        logger.info("  Step 1: Collecting activations...")
        
        hindi_acts, hindi_prompts = self.collect_activations(collector, hindi_data)
        english_acts, english_prompts = self.collect_activations(collector, english_data)
        
        # Step 2: Compute mean direction
        logger.info("  Step 2: Computing mean direction...")
        hindi_mean = hindi_acts.mean(dim=0)
        english_mean = english_acts.mean(dim=0)
        mean_direction = F.normalize(english_mean - hindi_mean, dim=0)
        
        # Step 3: Train SAE
        logger.info("  Step 3: Training SAE...")
        all_acts = torch.cat([hindi_acts, english_acts], dim=0)
        
        sae, sae_metrics = train_sae(
            activations=all_acts,
            sae_type=self.config.sae_type,
            hidden_dim=self.config.hidden_dim,
            sae_dim=self.config.sae_hidden_dim,
            epochs=self.config.sae_epochs,
            lr=self.config.sae_lr,
            device=self.device,
            k=self.config.sae_k,
            threshold=self.config.sae_threshold
        )
        
        logger.info(f"    SAE trained: val_loss={sae_metrics['val_loss']:.4f}, L0={sae_metrics['l0']:.1f}")
        
        # Step 4: Find language-specific SAE features
        logger.info("  Step 4: Finding language-specific features...")
        language_features = self.find_language_specific_features(sae, hindi_acts, english_acts)
        
        logger.info(f"    Found {len(language_features['hindi_features'])} Hindi-specific features")
        logger.info(f"    Found {len(language_features['english_features'])} English-specific features")
        
        # Step 5: Compare steering methods
        logger.info("  Step 5: Comparing steering methods...")
        
        methods = {
            "mean_direction": {
                "type": "direction",
                "direction": mean_direction,
            },
            "sae_top10_ablation": {
                "type": "sae_ablation",
                "sae": sae,
                "features_to_ablate": language_features["hindi_features"][:10],
            },
            "sae_top20_ablation": {
                "type": "sae_ablation",
                "sae": sae,
                "features_to_ablate": language_features["hindi_features"][:20],
            },
            "sae_direction": {
                "type": "sae_direction",
                "sae": sae,
                "hindi_features": language_features["hindi_features"][:20],
                "english_features": language_features["english_features"][:20],
            }
        }
        
        method_results = {}
        for method_name, method_config in methods.items():
            logger.info(f"    Testing: {method_name}")
            method_results[method_name] = self.evaluate_method(
                method_name, method_config, layer_idx, hindi_prompts
            )
        
        # Step 6: Statistical comparison
        comparison = self.compare_methods(method_results)
        
        return {
            "sae_metrics": sae_metrics,
            "language_features": {
                "num_hindi": len(language_features["hindi_features"]),
                "num_english": len(language_features["english_features"]),
                "top_hindi": language_features["hindi_features"][:10],
                "top_english": language_features["english_features"][:10],
                "monolinguality_scores": language_features["monolinguality_scores"][:20],
            },
            "method_results": method_results,
            "comparison": comparison,
        }
    
    def collect_activations(self, collector, data: List) -> Tuple[torch.Tensor, List[str]]:
        """Collect activations for all prompts."""
        activations = []
        prompts = []
        
        for item in data:
            prompt = item.get("prompt", item.get("text", ""))
            prompts.append(prompt)
            act = collector.get_activation(prompt, self.tokenizer)
            activations.append(act)
        
        return torch.stack(activations), prompts
    
    def find_language_specific_features(
        self, 
        sae: nn.Module,
        hindi_acts: torch.Tensor,
        english_acts: torch.Tensor
    ) -> Dict:
        """Find SAE features that are specific to each language."""
        
        with torch.no_grad():
            # Get SAE encodings
            hindi_encoded = sae.encode(hindi_acts)  # [N_hindi, sae_dim]
            english_encoded = sae.encode(english_acts)  # [N_english, sae_dim]
            
            # Compute activation frequency per feature
            hindi_freq = (hindi_encoded > 0).float().mean(dim=0)  # [sae_dim]
            english_freq = (english_encoded > 0).float().mean(dim=0)
            
            # Compute monolinguality score: |hindi_freq - english_freq| / (hindi_freq + english_freq + eps)
            mono_score = torch.abs(hindi_freq - english_freq) / (hindi_freq + english_freq + 1e-6)
            
            # Hindi-specific: high activation on Hindi, low on English
            hindi_specificity = hindi_freq - english_freq
            hindi_features = torch.argsort(hindi_specificity, descending=True).tolist()
            
            # English-specific: high activation on English, low on Hindi
            english_specificity = english_freq - hindi_freq
            english_features = torch.argsort(english_specificity, descending=True).tolist()
            
            # Get monolinguality scores for top features
            mono_scores = mono_score.tolist()
        
        return {
            "hindi_features": hindi_features,
            "english_features": english_features,
            "hindi_freq": hindi_freq.tolist(),
            "english_freq": english_freq.tolist(),
            "monolinguality_scores": sorted(mono_scores, reverse=True),
        }
    
    def evaluate_method(
        self, 
        method_name: str,
        method_config: Dict,
        layer_idx: int,
        prompts: List[str]
    ) -> Dict:
        """Evaluate a steering method on all prompts."""
        
        results = []
        
        for i, prompt in enumerate(prompts):
            if i % 20 == 0:
                logger.info(f"      Prompt {i+1}/{len(prompts)}")
            
            # Baseline
            baseline_output = self.generate_with_method(
                prompt, layer_idx, method_config, coeff=0.0
            )
            baseline_es = self.compute_hindi_score(baseline_output)
            
            # With steering (test multiple coefficients)
            best_delta = -float('inf')
            best_es = baseline_es
            
            for coeff in self.config.steering_coeffs:
                steered_output = self.generate_with_method(
                    prompt, layer_idx, method_config, coeff=coeff
                )
                steered_es = self.compute_hindi_score(steered_output)
                delta = baseline_es - steered_es
                
                if delta > best_delta:
                    best_delta = delta
                    best_es = steered_es
            
            results.append({
                "baseline_es": baseline_es,
                "best_steered_es": best_es,
                "best_delta": best_delta,
                "success": best_delta > 0,
            })
        
        # Compute statistics
        deltas = [r["best_delta"] for r in results]
        success_rate = np.mean([r["success"] for r in results])
        
        return {
            "per_prompt_results": results,
            "statistics": {
                "mean_delta": np.mean(deltas),
                "std_delta": np.std(deltas),  # KEY METRIC: lower = more consistent
                "min_delta": np.min(deltas),
                "max_delta": np.max(deltas),
                "success_rate": success_rate,
                "consistency_score": success_rate / (np.std(deltas) + 0.1),  # Higher = better
            }
        }
    
    def generate_with_method(
        self,
        prompt: str,
        layer_idx: int,
        method_config: Dict,
        coeff: float
    ) -> str:
        """Generate text using specified steering method."""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        hooks = []
        
        if coeff > 0:
            method_type = method_config["type"]
            
            if method_type == "direction":
                # Mean direction steering
                direction = method_config["direction"]
                
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    hidden[:, -1, :] = hidden[:, -1, :] + coeff * direction.to(hidden.device)
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                
            elif method_type == "sae_ablation":
                # SAE feature ablation
                sae = method_config["sae"]
                features = method_config["features_to_ablate"]
                
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    # Encode, zero out features, decode
                    last_token = hidden[:, -1, :]
                    encoded = sae.encode(last_token)
                    # Ablate Hindi-specific features
                    for feat_idx in features:
                        encoded[:, feat_idx] = 0
                    decoded = sae.decode(encoded)
                    # Blend with coefficient
                    hidden[:, -1, :] = (1 - coeff) * last_token + coeff * decoded
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                
            elif method_type == "sae_direction":
                # SAE-based direction (difference between language feature means)
                sae = method_config["sae"]
                hindi_feats = method_config["hindi_features"]
                english_feats = method_config["english_features"]
                
                def hook(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    last_token = hidden[:, -1, :]
                    encoded = sae.encode(last_token)
                    
                    # Reduce Hindi features, boost English features
                    for feat_idx in hindi_feats:
                        encoded[:, feat_idx] *= (1 - coeff * 0.5)
                    for feat_idx in english_feats:
                        encoded[:, feat_idx] *= (1 + coeff * 0.3)
                    
                    decoded = sae.decode(encoded)
                    hidden[:, -1, :] = decoded
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            
            layer = self.model.model.layers[layer_idx]
            hook_handle = layer.register_forward_hook(hook)
            hooks.append(hook_handle)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        finally:
            for h in hooks:
                h.remove()
        
        return generated
    
    def compare_methods(self, method_results: Dict) -> Dict:
        """Statistical comparison of steering methods."""
        
        comparison = {}
        
        # Extract deltas for each method
        method_deltas = {}
        for name, results in method_results.items():
            method_deltas[name] = [r["best_delta"] for r in results["per_prompt_results"]]
        
        # Pairwise comparisons
        method_names = list(method_deltas.keys())
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                t, p = stats.ttest_rel(method_deltas[m1], method_deltas[m2])
                comparison[f"{m1}_vs_{m2}"] = {
                    "t_statistic": t,
                    "p_value": p,
                    "significant": p < 0.05,
                    "better": m1 if np.mean(method_deltas[m1]) > np.mean(method_deltas[m2]) else m2
                }
        
        # Rank methods by consistency (low std) and effectiveness (high mean)
        rankings = []
        for name, results in method_results.items():
            stats_dict = results["statistics"]
            rankings.append({
                "method": name,
                "mean_delta": stats_dict["mean_delta"],
                "std_delta": stats_dict["std_delta"],
                "success_rate": stats_dict["success_rate"],
                "consistency_score": stats_dict["consistency_score"],
            })
        
        rankings.sort(key=lambda x: x["consistency_score"], reverse=True)
        comparison["rankings"] = rankings
        comparison["best_method"] = rankings[0]["method"]
        
        # H2 test: Is SAE better than mean_direction?
        mean_dir_consistency = method_results["mean_direction"]["statistics"]["consistency_score"]
        sae_consistencies = [
            method_results[m]["statistics"]["consistency_score"]
            for m in method_results if m.startswith("sae")
        ]
        
        comparison["H2_supported"] = any(c > mean_dir_consistency for c in sae_consistencies)
        comparison["H2_interpretation"] = (
            f"H2 {'SUPPORTED' if comparison['H2_supported'] else 'NOT SUPPORTED'}: "
            f"Mean direction consistency={mean_dir_consistency:.3f}, "
            f"Best SAE consistency={max(sae_consistencies):.3f}"
        )
        
        return comparison
    
    def aggregate_comparison(self, per_layer_results: Dict) -> Dict:
        """Aggregate comparison across layers."""
        
        aggregate = {
            "methods": defaultdict(lambda: {"mean_deltas": [], "std_deltas": [], "success_rates": []}),
            "H2_support_count": 0,
            "total_layers": len(per_layer_results),
        }
        
        for layer_name, results in per_layer_results.items():
            if "comparison" in results and results["comparison"].get("H2_supported"):
                aggregate["H2_support_count"] += 1
            
            for method_name, method_results in results.get("method_results", {}).items():
                stats = method_results["statistics"]
                aggregate["methods"][method_name]["mean_deltas"].append(stats["mean_delta"])
                aggregate["methods"][method_name]["std_deltas"].append(stats["std_delta"])
                aggregate["methods"][method_name]["success_rates"].append(stats["success_rate"])
        
        # Compute overall statistics
        for method_name, method_data in aggregate["methods"].items():
            aggregate["methods"][method_name] = {
                "overall_mean_delta": np.mean(method_data["mean_deltas"]),
                "overall_std_delta": np.mean(method_data["std_deltas"]),
                "overall_success_rate": np.mean(method_data["success_rates"]),
            }
        
        aggregate["H2_final_verdict"] = (
            f"H2 supported in {aggregate['H2_support_count']}/{aggregate['total_layers']} layers"
        )
        
        return dict(aggregate)
