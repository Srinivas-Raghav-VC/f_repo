"""
Results Manager
===============
Handles saving results, generating plots, and final reports.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from .logging_utils import get_logger

logger = get_logger(__name__)


class ResultsManager:
    """Manage experiment results and visualizations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        self.reports_dir = self.output_dir / "reports"
        
        for d in [self.results_dir, self.plots_dir, self.reports_dir]:
            d.mkdir(exist_ok=True)
    
    def save_experiment_results(self, exp_name: str, results: Dict) -> None:
        """Save experiment results to JSON."""
        path = self.results_dir / f"{exp_name}.json"
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {path}")
    
    def save_prompt_analysis_plots(self, results: Dict) -> None:
        """Generate and save plots for Experiment 1."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
            return
        
        # Plot 1: Correlation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Collect correlations across layers
        layers = []
        metrics = ["dist_hindi_vs_delta", "dist_english_vs_delta", 
                   "projection_vs_delta", "cosine_vs_delta", "ratio_vs_delta"]
        corr_matrix = []
        
        for layer_name, layer_results in results.get("per_layer_results", {}).items():
            layers.append(layer_name)
            row = []
            for metric in metrics:
                corr = layer_results.get("correlations", {}).get(metric, {}).get("correlation", 0)
                row.append(corr)
            corr_matrix.append(row)
        
        if corr_matrix:
            sns.heatmap(
                np.array(corr_matrix),
                xticklabels=[m.replace("_vs_delta", "") for m in metrics],
                yticklabels=layers,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
                ax=axes[0]
            )
            axes[0].set_title("Correlation with Steering Delta by Layer")
        
        # Plot 2: Success rate by layer
        success_rates = []
        layer_names = []
        for layer_name, layer_results in results.get("per_layer_results", {}).items():
            layer_names.append(layer_name.replace("layer_", ""))
            success_rates.append(layer_results.get("summary", {}).get("success_rate", 0))
        
        if success_rates:
            axes[1].bar(layer_names, success_rates, color='steelblue')
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Success Rate")
            axes[1].set_title("Steering Success Rate by Layer")
            axes[1].axhline(y=0.5, color='red', linestyle='--', label='Random')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "exp1_prompt_analysis.png", dpi=150)
        plt.close()
        
        logger.info(f"Saved prompt analysis plots to {self.plots_dir}")
    
    def save_sae_comparison_plots(self, results: Dict) -> None:
        """Generate plots for Experiment 2."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Collect method performance across layers
        methods = set()
        layer_data = {}
        
        for layer_name, layer_results in results.get("per_layer_results", {}).items():
            layer_data[layer_name] = {}
            for method, method_results in layer_results.get("method_results", {}).items():
                methods.add(method)
                stats = method_results.get("statistics", {})
                layer_data[layer_name][method] = {
                    "mean_delta": stats.get("mean_delta", 0),
                    "std_delta": stats.get("std_delta", 0),
                    "success_rate": stats.get("success_rate", 0),
                }
        
        if layer_data:
            # Plot 1: Mean delta comparison
            x = list(layer_data.keys())
            for method in methods:
                y = [layer_data[l].get(method, {}).get("mean_delta", 0) for l in x]
                axes[0].plot(x, y, marker='o', label=method)
            
            axes[0].set_xlabel("Layer")
            axes[0].set_ylabel("Mean Delta (ES reduction)")
            axes[0].set_title("Steering Effectiveness by Method")
            axes[0].legend()
            axes[0].axhline(y=0, color='gray', linestyle='--')
            
            # Plot 2: Consistency (std) comparison
            for method in methods:
                y = [layer_data[l].get(method, {}).get("std_delta", 0) for l in x]
                axes[1].plot(x, y, marker='s', label=method)
            
            axes[1].set_xlabel("Layer")
            axes[1].set_ylabel("Std Delta (lower = more consistent)")
            axes[1].set_title("Steering Consistency by Method")
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "exp2_sae_comparison.png", dpi=150)
        plt.close()
        
        logger.info(f"Saved SAE comparison plots to {self.plots_dir}")
    
    def save_layer_analysis_plots(self, results: Dict) -> None:
        """Generate plots for Experiment 3."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        layers = []
        aucs = []
        separabilities = []
        cosines = []
        combined = []
        
        for layer_name, metrics in results.get("layer_analysis", {}).items():
            layers.append(int(layer_name.split("_")[1]))
            aucs.append(metrics.get("probe_auc", 0))
            separabilities.append(metrics.get("separability", 0))
            cosines.append(metrics.get("cosine_distance", 0))
            combined.append(metrics.get("combined_score", 0))
        
        # Sort by layer
        sort_idx = np.argsort(layers)
        layers = [layers[i] for i in sort_idx]
        aucs = [aucs[i] for i in sort_idx]
        separabilities = [separabilities[i] for i in sort_idx]
        cosines = [cosines[i] for i in sort_idx]
        combined = [combined[i] for i in sort_idx]
        
        axes[0, 0].plot(layers, aucs, 'b-o')
        axes[0, 0].set_title("Probe AUC by Layer")
        axes[0, 0].set_xlabel("Layer")
        axes[0, 0].set_ylabel("AUC")
        
        axes[0, 1].plot(layers, separabilities, 'g-o')
        axes[0, 1].set_title("Separability by Layer")
        axes[0, 1].set_xlabel("Layer")
        axes[0, 1].set_ylabel("Between/Within Variance")
        
        axes[1, 0].plot(layers, cosines, 'r-o')
        axes[1, 0].set_title("Cosine Distance by Layer")
        axes[1, 0].set_xlabel("Layer")
        axes[1, 0].set_ylabel("1 - Cosine Similarity")
        
        axes[1, 1].plot(layers, combined, 'm-o')
        axes[1, 1].set_title("Combined Score by Layer")
        axes[1, 1].set_xlabel("Layer")
        axes[1, 1].set_ylabel("Score")
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "exp3_layer_analysis.png", dpi=150)
        plt.close()
    
    def generate_final_report(self, all_results: Dict, config) -> None:
        """Generate final summary report."""
        
        report = []
        report.append("=" * 70)
        report.append("MMIE RESEARCH: FINAL REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")
        
        # Hypothesis summary
        report.append("HYPOTHESIS RESULTS")
        report.append("-" * 40)
        
        # H1
        exp1 = all_results.get("prompt_analysis", {})
        if "aggregate_results" in exp1:
            agg = exp1["aggregate_results"]
            report.append(f"H1 (Distance predicts steering): {agg.get('H1_interpretation', 'Not tested')}")
        
        # H2
        exp2 = all_results.get("sae_vs_direction", {})
        if "aggregate_results" in exp2:
            agg = exp2["aggregate_results"]
            report.append(f"H2 (SAE more consistent): {agg.get('H2_final_verdict', 'Not tested')}")
        
        # H3
        exp6 = all_results.get("clustering", {})
        if "H3_results" in exp6:
            h3 = exp6["H3_results"]
            report.append(f"H3 (Clusters need different directions): {h3.get('H3_interpretation', 'Not tested')}")
        
        # H4
        exp4 = all_results.get("tokenization", {})
        if "H4_results" in exp4:
            h4 = exp4["H4_results"]
            report.append(f"H4 (Tokenization bias confounds): {h4.get('H4_interpretation', 'Not tested')}")
        
        report.append("")
        
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-" * 40)
        
        # Best layer
        exp3 = all_results.get("layer_analysis", {})
        if "layer_rankings" in exp3:
            recommended = exp3["layer_rankings"].get("recommended_layers", [])
            report.append(f"Recommended layers for intervention: {recommended}")
        
        # Adversarial
        exp5 = all_results.get("adversarial", {})
        if "overall_vulnerability" in exp5:
            report.append(f"Adversarial vulnerability: {exp5.get('vulnerability_assessment', 'Unknown')}")
            report.append(f"  Extraction rate: {exp5['overall_vulnerability']:.1%}")
        
        report.append("")
        
        # Configuration
        report.append("CONFIGURATION")
        report.append("-" * 40)
        report.append(f"Model: {config.model_name}")
        report.append(f"Layers analyzed: {config.layers}")
        report.append(f"SAE type: {config.sae_type}")
        report.append(f"Samples per language: {config.num_samples}")
        report.append(f"Steering coefficients: {config.steering_coeffs}")
        
        report.append("")
        report.append("=" * 70)
        
        # Save report
        report_text = "\n".join(report)
        report_path = self.reports_dir / "final_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"\nFinal report saved to {report_path}")
        logger.info("\n" + report_text)
        
        # Also save as JSON
        json_report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": config.model_name,
                "layers": config.layers,
                "sae_type": config.sae_type,
                "num_samples": config.num_samples,
            },
            "hypothesis_results": {},
            "key_findings": {},
        }
        
        # Extract hypothesis results
        if exp1.get("aggregate_results"):
            json_report["hypothesis_results"]["H1"] = exp1["aggregate_results"].get("H1_supported", None)
        if exp2.get("aggregate_results"):
            json_report["hypothesis_results"]["H2"] = "SAE" in exp2["aggregate_results"].get("H2_final_verdict", "")
        if exp6.get("H3_results"):
            json_report["hypothesis_results"]["H3"] = exp6["H3_results"].get("H3_supported", None)
        if exp4.get("H4_results"):
            json_report["hypothesis_results"]["H4"] = exp4["H4_results"].get("H4_supported", None)
        
        with open(self.reports_dir / "final_report.json", 'w') as f:
            json.dump(json_report, f, indent=2)
