#!/usr/bin/env python3
"""
Multilingual Mechanistic Interpretability Experiments (MMIE)
============================================================
Complete Research Codebase for:
"The Limits of Linear Language Steering: Why Mean-Difference Vectors 
 Fail for Multilingual Control in LLMs"

Author: Raghav
Institution: IIIT Kottayam
Date: November 2025

Hypotheses:
-----------
H0 (Main): Linear steering fails because Hindi-English forms non-linear distribution
H1: Steering effectiveness correlates with distance from mean direction
H2: SAE features provide more consistent steering than mean-difference
H3: Prompts cluster into groups requiring different steering directions
H4: Tokenization bias confounds activation collection

Usage:
------
    python main.py --experiment all --output_dir results/
    python main.py --experiment prompt_analysis --layers 7,13,19
    python main.py --experiment sae_vs_direction --sae_type jumprelu
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from experiments import (
    run_experiment_1_prompt_analysis,
    run_experiment_2_sae_vs_direction,
    run_experiment_3_layer_analysis,
    run_experiment_4_tokenization,
    run_experiment_5_adversarial,
    run_experiment_6_clustering
)
from utils.logging_utils import setup_logging, get_logger
from utils.results_manager import ResultsManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMIE Research: Multilingual Steering Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python main.py --experiment all
  
  # Run specific experiment
  python main.py --experiment prompt_analysis --layers 7,13,19
  
  # Quick test mode
  python main.py --experiment all --quick_test
        """
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="all",
        choices=["all", "prompt_analysis", "sae_vs_direction", "layer_analysis",
                 "tokenization", "adversarial", "clustering"],
        help="Which experiment to run"
    )
    
    # Data paths
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="/mnt/user-data/uploads",
        help="Directory containing JSONL data files"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name"
    )
    
    parser.add_argument(
        "--layers",
        type=str,
        default="7,13,19,22",
        help="Comma-separated layer indices to analyze"
    )
    
    # SAE settings
    parser.add_argument(
        "--sae_type",
        type=str,
        default="jumprelu",
        choices=["jumprelu", "topk", "both"],
        help="SAE architecture type"
    )
    
    parser.add_argument(
        "--sae_dim",
        type=int,
        default=4096,
        help="SAE hidden dimension"
    )
    
    # Steering settings
    parser.add_argument(
        "--steering_coeffs",
        type=str,
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated steering coefficients to test"
    )
    
    # Computation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples per language for analysis"
    )
    
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run quick test with reduced samples"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir / "experiment.log", verbose=args.verbose)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("MMIE Research: Multilingual Steering Analysis")
    logger.info("=" * 60)
    
    # Create config
    config = Config(
        data_dir=args.data_dir,
        output_dir=str(output_dir),
        model_name=args.model_name,
        layers=[int(x) for x in args.layers.split(",")],
        sae_type=args.sae_type,
        sae_dim=args.sae_dim,
        steering_coeffs=[float(x) for x in args.steering_coeffs.split(",")],
        batch_size=args.batch_size,
        num_samples=args.num_samples if not args.quick_test else 20,
        seed=args.seed,
        device=args.device,
        quick_test=args.quick_test
    )
    
    # Save config
    config.save(output_dir / "config.json")
    logger.info(f"Config saved to {output_dir / 'config.json'}")
    
    # Initialize results manager
    results_manager = ResultsManager(output_dir)
    
    # Run experiments
    experiments_to_run = []
    if args.experiment == "all":
        experiments_to_run = [
            ("prompt_analysis", run_experiment_1_prompt_analysis),
            ("sae_vs_direction", run_experiment_2_sae_vs_direction),
            ("layer_analysis", run_experiment_3_layer_analysis),
            ("tokenization", run_experiment_4_tokenization),
            ("adversarial", run_experiment_5_adversarial),
            ("clustering", run_experiment_6_clustering),
        ]
    else:
        exp_map = {
            "prompt_analysis": run_experiment_1_prompt_analysis,
            "sae_vs_direction": run_experiment_2_sae_vs_direction,
            "layer_analysis": run_experiment_3_layer_analysis,
            "tokenization": run_experiment_4_tokenization,
            "adversarial": run_experiment_5_adversarial,
            "clustering": run_experiment_6_clustering,
        }
        experiments_to_run = [(args.experiment, exp_map[args.experiment])]
    
    all_results = {}
    
    for exp_name, exp_func in experiments_to_run:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running Experiment: {exp_name}")
        logger.info(f"{'=' * 60}")
        
        try:
            results = exp_func(config, results_manager)
            all_results[exp_name] = results
            results_manager.save_experiment_results(exp_name, results)
            logger.info(f"✓ {exp_name} completed successfully")
        except Exception as e:
            logger.error(f"✗ {exp_name} failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            all_results[exp_name] = {"error": str(e)}
    
    # Generate final report
    logger.info(f"\n{'=' * 60}")
    logger.info("Generating Final Report")
    logger.info(f"{'=' * 60}")
    
    results_manager.generate_final_report(all_results, config)
    
    logger.info(f"\n✓ All results saved to: {output_dir}")
    logger.info("=" * 60)
    
    return all_results


if __name__ == "__main__":
    main()
