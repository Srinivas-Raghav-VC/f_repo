#!/usr/bin/env python3
"""
MMIE Complete Research Suite
============================

Multilingual Mechanistic Interpretability Experiments

This codebase tests:
1. Semantic subspace hypothesis (do languages share meaning space?)
2. Causal language control (can we change output language?)
3. Goldilocks zone (what coefficient range works?)
4. Cross-language leaks (does Hindi steering affect Urdu/Bengali?)
5. SAE feature disentanglement (can we separate language from content?)

Author: Raghav, IIIT Kottayam
Advisor: Dr. Krishnendendu S P
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Local imports
from config import Config, setup_environment
from data_loader import load_all_data
from model_manager import ModelManager
from experiments import (
    run_semantic_subspace_analysis,
    run_steering_grid_search,
    run_causality_test,
    run_coherence_test,
    run_cross_language_test,
    run_sae_analysis,
    run_adversarial_test,
)
from reporting import generate_final_report, save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMIE Complete Research Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--experiment", "-e", type=str, default="all",
        choices=["all", "semantic", "steering", "causality", "coherence", 
                 "crosslang", "sae", "adversarial"],
        help="Which experiment to run")
    
    parser.add_argument("--data_dir", "-d", type=str, default=".",
        help="Directory with JSONL data files")
    
    parser.add_argument("--output_dir", "-o", type=str, default="./results",
        help="Output directory")
    
    parser.add_argument("--model", "-m", type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name")
    
    parser.add_argument("--quick", action="store_true",
        help="Quick mode with fewer samples")
    
    parser.add_argument("--layers", type=str, default="7,10,13,16,19,22",
        help="Layers to test")
    
    parser.add_argument("--coefficients", type=str, default="1,2,5,10,20",
        help="Steering coefficients to test")
    
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MMIE COMPLETE RESEARCH SUITE")
    print("=" * 70)
    print(f"Experiment: {args.experiment}")
    print(f"Output: {output_dir}")
    print(f"Quick mode: {args.quick}")
    print("=" * 70)
    
    # Config
    config = Config(
        data_dir=args.data_dir,
        output_dir=str(output_dir),
        model_name=args.model,
        layers=[int(x) for x in args.layers.split(",")],
        coefficients=[float(x) for x in args.coefficients.split(",")],
        quick_mode=args.quick,
        seed=args.seed,
    )
    config.save(output_dir / "config.json")
    
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Load model
    print("\n[1] Loading model...")
    model_manager = ModelManager(config)
    model_manager.load()
    
    # Load data
    print("\n[2] Loading data...")
    data = load_all_data(config)
    
    # Run experiments
    all_results = {}
    
    experiments_to_run = {
        "semantic": ("Semantic Subspace Analysis", run_semantic_subspace_analysis),
        "steering": ("Steering Grid Search", run_steering_grid_search),
        "causality": ("Causality Test", run_causality_test),
        "coherence": ("Coherence Test", run_coherence_test),
        "crosslang": ("Cross-Language Leak Test", run_cross_language_test),
        "sae": ("SAE Feature Analysis", run_sae_analysis),
        "adversarial": ("Adversarial Robustness", run_adversarial_test),
    }
    
    if args.experiment == "all":
        to_run = list(experiments_to_run.keys())
    else:
        to_run = [args.experiment]
    
    for exp_key in to_run:
        exp_name, exp_func = experiments_to_run[exp_key]
        print(f"\n{'=' * 70}")
        print(f"[{to_run.index(exp_key) + 3}] Running: {exp_name}")
        print("=" * 70)
        
        try:
            results = exp_func(model_manager, data, config)
            all_results[exp_key] = results
            save_results(results, output_dir / f"{exp_key}_results.json")
            print(f"  ✓ {exp_name} completed")
        except Exception as e:
            print(f"  ✗ {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_key] = {"error": str(e)}
    
    # Generate report
    print(f"\n{'=' * 70}")
    print("Generating Final Report")
    print("=" * 70)
    
    generate_final_report(all_results, config, output_dir)
    
    print(f"\n✓ All results saved to: {output_dir}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
