"""
Experiment 1: Prompt-Level Analysis of Steering Inconsistency
==============================================================

Hypothesis H1: Steering effectiveness correlates with prompt's distance 
from the mean direction.

This experiment:
1. Computes steering effectiveness for each individual prompt
2. Analyzes which prompt characteristics predict success/failure
3. Tests correlation between distance from mean and steering delta
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseExperiment, compute_statistics
from utils.logging_utils import get_logger
from models.model_loader import load_model_and_tokenizer
from models.activation_collector import ActivationCollector
from models.steering import compute_steering_direction, apply_steering

logger = get_logger(__name__)


def run_experiment_1_prompt_analysis(config, results_manager) -> Dict[str, Any]:
    """
    Run Experiment 1: Prompt-Level Analysis
    
    Tests H1: Steering effectiveness correlates with distance from mean direction
    """
    logger.info("=" * 50)
    logger.info("EXPERIMENT 1: Prompt-Level Analysis")
    logger.info("Testing H1: Distance from mean predicts steering success")
    logger.info("=" * 50)
    
    experiment = PromptAnalysisExperiment(config, results_manager)
    return experiment.run()


class PromptAnalysisExperiment(BaseExperiment):
    """Prompt-level steering analysis."""
    
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
            "hypothesis": "H1: Steering effectiveness correlates with distance from mean",
            "per_layer_results": {},
            "aggregate_results": {},
            "prompt_characteristics": {},
        }
        
        # Analyze each layer
        for layer_idx in self.config.layers:
            logger.info(f"\n--- Analyzing Layer {layer_idx} ---")
            layer_results = self.analyze_layer(layer_idx, hindi_data, english_data)
            results["per_layer_results"][f"layer_{layer_idx}"] = layer_results
        
        # Aggregate analysis
        results["aggregate_results"] = self.aggregate_analysis(results["per_layer_results"])
        
        # Generate visualizations
        self.results_manager.save_prompt_analysis_plots(results)
        
        return results
    
    def analyze_layer(self, layer_idx: int, hindi_data: List, english_data: List) -> Dict:
        """Analyze steering at a specific layer."""
        
        collector = ActivationCollector(self.model, layer_idx)
        
        # Step 1: Collect activations for all prompts
        logger.info(f"  Collecting activations for {len(hindi_data)} Hindi prompts...")
        hindi_activations = []
        hindi_prompts = []
        
        for item in hindi_data:
            prompt = item.get("prompt", item.get("text", ""))
            hindi_prompts.append(prompt)
            act = collector.get_activation(prompt, self.tokenizer)
            hindi_activations.append(act)
        
        logger.info(f"  Collecting activations for {len(english_data)} English prompts...")
        english_activations = []
        english_prompts = []
        
        for item in english_data:
            prompt = item.get("prompt", item.get("text", ""))
            english_prompts.append(prompt)
            act = collector.get_activation(prompt, self.tokenizer)
            english_activations.append(act)
        
        # Stack activations
        hindi_acts = torch.stack(hindi_activations)  # [N, hidden_dim]
        english_acts = torch.stack(english_activations)
        
        # Step 2: Compute mean direction
        hindi_mean = hindi_acts.mean(dim=0)
        english_mean = english_acts.mean(dim=0)
        steering_direction = english_mean - hindi_mean  # Push toward English
        steering_direction = F.normalize(steering_direction, dim=0)
        
        # Step 3: Compute distances from means
        logger.info("  Computing prompt distances from means...")
        hindi_distances = self.compute_distances(hindi_acts, hindi_mean, english_mean, steering_direction)
        english_distances = self.compute_distances(english_acts, hindi_mean, english_mean, steering_direction)
        
        # Step 4: Test steering on each prompt individually
        logger.info("  Testing steering on individual prompts...")
        prompt_results = []
        
        for i, (prompt, act, dist) in enumerate(zip(hindi_prompts, hindi_activations, hindi_distances)):
            if i % 20 == 0:
                logger.info(f"    Processing prompt {i+1}/{len(hindi_prompts)}")
            
            result = self.test_single_prompt_steering(
                prompt=prompt,
                activation=act,
                steering_direction=steering_direction,
                layer_idx=layer_idx,
                distances=dist
            )
            result["prompt_idx"] = i
            result["prompt_text"] = prompt[:100]  # Truncate for storage
            prompt_results.append(result)
        
        # Step 5: Analyze correlations
        logger.info("  Computing correlations...")
        correlation_results = self.compute_correlations(prompt_results)
        
        # Step 6: Prompt characteristic analysis
        logger.info("  Analyzing prompt characteristics...")
        characteristic_analysis = self.analyze_prompt_characteristics(
            prompt_results, hindi_prompts
        )
        
        return {
            "num_prompts": len(hindi_data),
            "prompt_results": prompt_results,
            "correlations": correlation_results,
            "characteristic_analysis": characteristic_analysis,
            "summary": {
                "mean_baseline_es": np.mean([r["baseline_es"] for r in prompt_results]),
                "mean_steered_es": np.mean([r["best_steered_es"] for r in prompt_results]),
                "success_rate": np.mean([r["steering_success"] for r in prompt_results]),
                "correlation_distance_delta": correlation_results.get("distance_vs_delta", {}).get("correlation", 0),
            }
        }
    
    def compute_distances(
        self, 
        activations: torch.Tensor,
        hindi_mean: torch.Tensor,
        english_mean: torch.Tensor,
        direction: torch.Tensor
    ) -> List[Dict]:
        """Compute various distance metrics for each activation."""
        distances = []
        
        for act in activations:
            # Distance to Hindi mean
            dist_to_hindi = torch.norm(act - hindi_mean).item()
            
            # Distance to English mean
            dist_to_english = torch.norm(act - english_mean).item()
            
            # Projection onto steering direction
            projection = torch.dot(act, direction).item()
            
            # Cosine similarity with direction
            cos_sim = F.cosine_similarity(act.unsqueeze(0), direction.unsqueeze(0)).item()
            
            distances.append({
                "dist_to_hindi_mean": dist_to_hindi,
                "dist_to_english_mean": dist_to_english,
                "projection_on_direction": projection,
                "cosine_with_direction": cos_sim,
                "hindi_english_ratio": dist_to_hindi / max(dist_to_english, 1e-6),
            })
        
        return distances
    
    def test_single_prompt_steering(
        self,
        prompt: str,
        activation: torch.Tensor,
        steering_direction: torch.Tensor,
        layer_idx: int,
        distances: Dict
    ) -> Dict:
        """Test steering on a single prompt with multiple coefficients."""
        
        # Get baseline generation
        baseline_output = self.generate_text(prompt, layer_idx, steering_direction, coeff=0.0)
        baseline_es = self.compute_hindi_score(baseline_output)
        
        # Test different steering coefficients
        coeff_results = {}
        best_reduction = -float('inf')
        best_coeff = 0
        best_es = baseline_es
        
        for coeff in self.config.steering_coeffs:
            steered_output = self.generate_text(prompt, layer_idx, steering_direction, coeff=coeff)
            steered_es = self.compute_hindi_score(steered_output)
            delta = baseline_es - steered_es  # Positive = reduced Hindi
            
            coeff_results[f"coeff_{coeff}"] = {
                "es": steered_es,
                "delta": delta,
                "output_sample": steered_output[:200]
            }
            
            if delta > best_reduction:
                best_reduction = delta
                best_coeff = coeff
                best_es = steered_es
        
        return {
            "baseline_es": baseline_es,
            "baseline_output": baseline_output[:200],
            "coeff_results": coeff_results,
            "best_coeff": best_coeff,
            "best_steered_es": best_es,
            "best_delta": best_reduction,
            "steering_success": best_reduction > 0,  # Did steering reduce Hindi?
            "distances": distances
        }
    
    def generate_text(
        self, 
        prompt: str, 
        layer_idx: int, 
        direction: torch.Tensor, 
        coeff: float
    ) -> str:
        """Generate text with optional steering."""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        # Apply steering hook if coeff > 0
        hooks = []
        if coeff > 0:
            def steering_hook(module, input, output):
                # output is tuple, first element is hidden states
                hidden = output[0] if isinstance(output, tuple) else output
                # Add steering to last token position
                hidden[:, -1, :] = hidden[:, -1, :] + coeff * direction.to(hidden.device)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(steering_hook)
            hooks.append(hook)
        
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
            for hook in hooks:
                hook.remove()
        
        return generated
    
    def compute_correlations(self, prompt_results: List[Dict]) -> Dict:
        """Compute correlations between distances and steering effectiveness."""
        
        correlations = {}
        
        # Extract arrays
        deltas = [r["best_delta"] for r in prompt_results]
        dist_to_hindi = [r["distances"]["dist_to_hindi_mean"] for r in prompt_results]
        dist_to_english = [r["distances"]["dist_to_english_mean"] for r in prompt_results]
        projections = [r["distances"]["projection_on_direction"] for r in prompt_results]
        cosines = [r["distances"]["cosine_with_direction"] for r in prompt_results]
        ratios = [r["distances"]["hindi_english_ratio"] for r in prompt_results]
        
        # Correlation: distance to Hindi mean vs delta
        r, p = stats.pearsonr(dist_to_hindi, deltas)
        correlations["dist_hindi_vs_delta"] = {"correlation": r, "p_value": p}
        
        # Correlation: distance to English mean vs delta
        r, p = stats.pearsonr(dist_to_english, deltas)
        correlations["dist_english_vs_delta"] = {"correlation": r, "p_value": p}
        
        # Correlation: projection on direction vs delta (KEY TEST FOR H1)
        r, p = stats.pearsonr(projections, deltas)
        correlations["projection_vs_delta"] = {"correlation": r, "p_value": p}
        
        # Correlation: cosine with direction vs delta
        r, p = stats.pearsonr(cosines, deltas)
        correlations["cosine_vs_delta"] = {"correlation": r, "p_value": p}
        
        # Correlation: ratio vs delta
        r, p = stats.pearsonr(ratios, deltas)
        correlations["ratio_vs_delta"] = {"correlation": r, "p_value": p}
        
        # Binary classification: can we predict success from distances?
        X = np.array([dist_to_hindi, dist_to_english, projections, cosines]).T
        y = np.array([1 if r["steering_success"] else 0 for r in prompt_results])
        
        if len(np.unique(y)) > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(random_state=42)
            clf.fit(X_scaled, y)
            accuracy = clf.score(X_scaled, y)
            correlations["prediction_accuracy"] = accuracy
            correlations["feature_importance"] = {
                "dist_hindi": clf.coef_[0][0],
                "dist_english": clf.coef_[0][1],
                "projection": clf.coef_[0][2],
                "cosine": clf.coef_[0][3],
            }
        
        return correlations
    
    def analyze_prompt_characteristics(
        self, 
        prompt_results: List[Dict],
        prompts: List[str]
    ) -> Dict:
        """Analyze which prompt characteristics predict steering success."""
        
        # Group by success/failure
        successful = [r for r in prompt_results if r["steering_success"]]
        failed = [r for r in prompt_results if not r["steering_success"]]
        
        # Compute characteristics
        def get_characteristics(prompt: str) -> Dict:
            tokens = self.tokenizer.encode(prompt)
            return {
                "length_chars": len(prompt),
                "length_tokens": len(tokens),
                "hindi_ratio": self.compute_hindi_score(prompt),
                "num_words": len(prompt.split()),
            }
        
        successful_chars = [get_characteristics(prompts[r["prompt_idx"]]) for r in successful]
        failed_chars = [get_characteristics(prompts[r["prompt_idx"]]) for r in failed]
        
        analysis = {
            "num_successful": len(successful),
            "num_failed": len(failed),
            "success_rate": len(successful) / max(len(prompt_results), 1),
        }
        
        # Compare characteristics
        for key in ["length_chars", "length_tokens", "hindi_ratio", "num_words"]:
            succ_vals = [c[key] for c in successful_chars] if successful_chars else [0]
            fail_vals = [c[key] for c in failed_chars] if failed_chars else [0]
            
            analysis[f"{key}_successful"] = compute_statistics(succ_vals)
            analysis[f"{key}_failed"] = compute_statistics(fail_vals)
            
            # Statistical test
            if len(succ_vals) > 1 and len(fail_vals) > 1:
                t, p = stats.ttest_ind(succ_vals, fail_vals)
                analysis[f"{key}_ttest"] = {"t_statistic": t, "p_value": p}
        
        return analysis
    
    def aggregate_analysis(self, per_layer_results: Dict) -> Dict:
        """Aggregate results across layers."""
        
        all_correlations = defaultdict(list)
        all_success_rates = []
        
        for layer_name, results in per_layer_results.items():
            all_success_rates.append(results["summary"]["success_rate"])
            
            for corr_name, corr_val in results["correlations"].items():
                if isinstance(corr_val, dict) and "correlation" in corr_val:
                    all_correlations[corr_name].append(corr_val["correlation"])
        
        aggregate = {
            "mean_success_rate": np.mean(all_success_rates),
            "std_success_rate": np.std(all_success_rates),
            "best_layer": max(per_layer_results.keys(), 
                            key=lambda k: per_layer_results[k]["summary"]["success_rate"]),
        }
        
        # Average correlations
        for corr_name, values in all_correlations.items():
            aggregate[f"mean_{corr_name}"] = np.mean(values)
        
        # H1 conclusion
        proj_corrs = all_correlations.get("projection_vs_delta", [])
        if proj_corrs:
            mean_corr = np.mean(proj_corrs)
            aggregate["H1_supported"] = abs(mean_corr) > 0.3  # Moderate correlation threshold
            aggregate["H1_correlation"] = mean_corr
            aggregate["H1_interpretation"] = (
                f"H1 {'SUPPORTED' if aggregate['H1_supported'] else 'NOT SUPPORTED'}: "
                f"Mean correlation between projection and delta = {mean_corr:.3f}"
            )
        
        return aggregate
