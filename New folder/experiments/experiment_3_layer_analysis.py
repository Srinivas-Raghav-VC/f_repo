"""
Experiment 3: Layer-wise Feature Localization
==============================================

Analyzes which layers have the most separable language features.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats

from .base import BaseExperiment, compute_statistics
from utils.logging_utils import get_logger
from models.model_loader import load_model_and_tokenizer
from models.activation_collector import ActivationCollector
from models.sae import train_sae

logger = get_logger(__name__)


def run_experiment_3_layer_analysis(config, results_manager) -> Dict[str, Any]:
    """Run layer-wise analysis experiment."""
    logger.info("=" * 50)
    logger.info("EXPERIMENT 3: Layer-wise Feature Localization")
    logger.info("=" * 50)
    
    experiment = LayerAnalysisExperiment(config, results_manager)
    return experiment.run()


class LayerAnalysisExperiment(BaseExperiment):
    """Analyze language feature distribution across layers."""
    
    def run(self) -> Dict[str, Any]:
        self.set_seed()
        
        logger.info("Loading model...")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config.model_name,
            device=self.device,
            dtype=self.config.model_dtype
        )
        
        logger.info("Loading data...")
        hindi_data = self.load_data(self.config.forget_file)[:self.config.num_samples]
        english_data = self.load_data(self.config.retain_file)[:self.config.num_samples]
        
        if not hindi_data or not english_data:
            return {"error": "No data loaded"}
        
        results = {
            "layer_analysis": {},
            "layer_rankings": {},
        }
        
        # Analyze all layers (or subset)
        all_layers = list(range(0, 32, 2))  # Every other layer for speed
        if self.config.quick_test:
            all_layers = [4, 10, 16, 22, 28]
        
        for layer_idx in all_layers:
            logger.info(f"\n  Analyzing layer {layer_idx}...")
            results["layer_analysis"][f"layer_{layer_idx}"] = self.analyze_single_layer(
                layer_idx, hindi_data, english_data
            )
        
        # Rank layers
        results["layer_rankings"] = self.rank_layers(results["layer_analysis"])
        
        # Generate heatmap
        self.results_manager.save_layer_analysis_plots(results)
        
        return results
    
    def analyze_single_layer(self, layer_idx: int, hindi_data: List, english_data: List) -> Dict:
        """Analyze a single layer for language separability."""
        
        collector = ActivationCollector(self.model, layer_idx)
        
        # Collect activations
        hindi_acts = []
        for item in hindi_data:
            prompt = item.get("prompt", item.get("text", ""))
            act = collector.get_activation(prompt, self.tokenizer)
            hindi_acts.append(act)
        
        english_acts = []
        for item in english_data:
            prompt = item.get("prompt", item.get("text", ""))
            act = collector.get_activation(prompt, self.tokenizer)
            english_acts.append(act)
        
        hindi_acts = torch.stack(hindi_acts)
        english_acts = torch.stack(english_acts)
        
        # Metric 1: Linear probe AUC
        X = torch.cat([hindi_acts, english_acts], dim=0).cpu().numpy()
        y = np.array([1] * len(hindi_acts) + [0] * len(english_acts))
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        probs = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        
        # Metric 2: Activation variance (separability)
        hindi_var = torch.var(hindi_acts, dim=0).mean().item()
        english_var = torch.var(english_acts, dim=0).mean().item()
        between_var = torch.var(torch.stack([hindi_acts.mean(0), english_acts.mean(0)]), dim=0).mean().item()
        separability = between_var / (hindi_var + english_var + 1e-6)
        
        # Metric 3: Cosine distance between means
        hindi_mean = hindi_acts.mean(dim=0)
        english_mean = english_acts.mean(dim=0)
        cosine_dist = 1 - F.cosine_similarity(hindi_mean.unsqueeze(0), english_mean.unsqueeze(0)).item()
        
        # Metric 4: Mean direction norm
        direction_norm = torch.norm(english_mean - hindi_mean).item()
        
        return {
            "probe_auc": auc,
            "separability": separability,
            "cosine_distance": cosine_dist,
            "direction_norm": direction_norm,
            "hindi_variance": hindi_var,
            "english_variance": english_var,
            "combined_score": (auc + separability + cosine_dist) / 3,
        }
    
    def rank_layers(self, layer_analysis: Dict) -> Dict:
        """Rank layers by different metrics."""
        
        rankings = {
            "by_auc": [],
            "by_separability": [],
            "by_cosine": [],
            "by_combined": [],
        }
        
        for layer_name, metrics in layer_analysis.items():
            layer_num = int(layer_name.split("_")[1])
            rankings["by_auc"].append((layer_num, metrics["probe_auc"]))
            rankings["by_separability"].append((layer_num, metrics["separability"]))
            rankings["by_cosine"].append((layer_num, metrics["cosine_distance"]))
            rankings["by_combined"].append((layer_num, metrics["combined_score"]))
        
        for key in rankings:
            rankings[key].sort(key=lambda x: x[1], reverse=True)
        
        # Best layers for intervention
        rankings["recommended_layers"] = [
            rankings["by_combined"][i][0] for i in range(min(3, len(rankings["by_combined"])))
        ]
        
        return rankings
