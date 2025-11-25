# MMIE Research: Multilingual Mechanistic Interpretability Experiments

**Paper Title:** "The Limits of Linear Language Steering: Why Mean-Difference Vectors Fail for Multilingual Control in LLMs"

## Overview

This codebase contains all experiments for investigating why linear steering methods produce inconsistent results for Hindi-English language control in LLMs.

### Hypotheses Tested

| ID | Hypothesis | Experiment |
|----|------------|------------|
| **H0** | Linear steering fails because Hindi-English forms non-linear distribution | All experiments |
| **H1** | Steering effectiveness correlates with distance from mean direction | Exp 1 |
| **H2** | SAE features provide more consistent steering than mean-difference | Exp 2 |
| **H3** | Prompts cluster into groups requiring different steering directions | Exp 6 |
| **H4** | Tokenization bias (5.1x) confounds activation collection | Exp 4 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (reduced samples)
./run.sh quick

# Run all experiments
./run.sh all

# Run specific experiment
./run.sh exp1  # Prompt analysis
./run.sh exp2  # SAE vs direction
./run.sh exp3  # Layer analysis
./run.sh exp4  # Tokenization bias
./run.sh exp5  # Adversarial robustness
./run.sh exp6  # Clustering
```

## Required Data Files

Place these JSONL files in your data directory (default: `/mnt/user-data/uploads/`):

- `forget_hindi.jsonl` - Hindi prompts (forget set)
- `retain_english.jsonl` - English prompts (retain set)
- `mixed_hinglish.jsonl` - Code-mixed prompts (optional)
- `adversarial.jsonl` - Adversarial prompts (optional)
- `urdu_test.jsonl` - Related language test (optional)
- `punjabi_test.jsonl` - Related language test (optional)
- `bengali_test.jsonl` - Related language test (optional)

### Data Format

```json
{"prompt": "Your Hindi or English prompt here", "expected_language": "hi"}
```

## Experiments

### Experiment 1: Prompt-Level Analysis

**Tests H1:** Does distance from mean direction predict steering success?

**Method:**
1. Collect activations for each prompt
2. Compute distances to Hindi/English means
3. Test steering at multiple coefficients
4. Correlate distance metrics with steering effectiveness

**Key Outputs:**
- Correlation heatmap (distance metrics vs steering delta)
- Feature importance for predicting success
- Per-prompt steering results

### Experiment 2: SAE vs Direction Comparison

**Tests H2:** Are SAE features more consistent than mean-difference?

**Methods Compared:**
1. `mean_direction`: Standard mean(English) - mean(Hindi)
2. `sae_top10_ablation`: Ablate top 10 Hindi-specific SAE features
3. `sae_top20_ablation`: Ablate top 20 Hindi-specific SAE features
4. `sae_direction`: Scale Hindi/English features in opposite directions

**Key Outputs:**
- Consistency scores (success_rate / std_delta)
- Method rankings by layer
- Statistical comparisons (paired t-tests)

### Experiment 3: Layer-wise Analysis

**Finds optimal layers for intervention**

**Metrics:**
- Probe AUC (linear separability)
- Separability (between/within variance ratio)
- Cosine distance between language means
- Combined score

**Key Outputs:**
- Layer ranking by each metric
- Recommended intervention layers

### Experiment 4: Tokenization Bias

**Tests H4:** Does tokenization bias affect steering?

**Method:**
1. Compute tokens/character for each language
2. Create length-matched subsets
3. Compare steering on matched vs unmatched data

**Key Outputs:**
- Bias ratio (typically ~5x for Hindi vs English)
- Improvement from length matching

### Experiment 5: Adversarial Robustness

**Tests vulnerability to extraction attacks**

**Attack Types:**
- Translation prompts ("Translate to Hindi...")
- Role-play prompts ("You are a Hindi teacher...")
- Instruction prompts ("Respond in Hindi...")
- Indirect prompts ("What's the Hindi word for...")

**Key Outputs:**
- Extraction rate per attack type
- Overall vulnerability assessment

### Experiment 6: Multi-Cluster Steering

**Tests H3:** Do prompt clusters need different directions?

**Method:**
1. Cluster Hindi prompts by activation patterns
2. Compute cluster-specific steering directions
3. Compare global vs cluster-specific steering

**Key Outputs:**
- Optimal cluster count (silhouette score)
- Improvement from cluster-aware steering
- Statistical significance (paired t-test)

## Output Structure

```
results/
├── run_YYYYMMDD_HHMMSS/
│   ├── config.json
│   ├── experiment.log
│   ├── results/
│   │   ├── prompt_analysis.json
│   │   ├── sae_vs_direction.json
│   │   ├── layer_analysis.json
│   │   ├── tokenization.json
│   │   ├── adversarial.json
│   │   └── clustering.json
│   ├── plots/
│   │   ├── exp1_prompt_analysis.png
│   │   ├── exp2_sae_comparison.png
│   │   └── exp3_layer_analysis.png
│   └── reports/
│       ├── final_report.txt
│       └── final_report.json
```

## Configuration

Key parameters in `config.py`:

```python
# Model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_dtype = "bfloat16"

# Layers to analyze
layers = [7, 13, 19, 22]

# SAE settings
sae_type = "jumprelu"  # or "topk"
sae_dim = 16384
sae_threshold = 0.05  # for JumpReLU
sae_k = 64  # for TopK

# Steering
steering_coeffs = [0.5, 1.0, 2.0, 4.0]

# Data
num_samples = 100  # per language
```

## Extending

### Adding New Experiments

1. Create `experiments/experiment_N_name.py`
2. Implement `run_experiment_N_name(config, results_manager)` function
3. Add to `experiments/__init__.py`
4. Add case to `run.sh`

### Adding New Steering Methods

1. Add method config to `experiment_2_sae_vs_direction.py`
2. Implement hook logic in `generate_with_method()`

### Adding New SAE Architectures

1. Add class to `models/sae.py`
2. Add case to `train_sae()` function

## Citation

If you use this code, please cite:

```bibtex
@article{raghav2025steering,
  title={The Limits of Linear Language Steering: Why Mean-Difference Vectors 
         Fail for Multilingual Control in LLMs},
  author={Raghav},
  institution={IIIT Kottayam},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- Dr. Krishnendendu S P (Advisor)
- IIIT Kottayam
