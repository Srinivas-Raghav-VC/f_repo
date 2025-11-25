#!/bin/bash
# MMIE Research Run Script
# ========================
# 
# Usage:
#   ./run.sh                    # Run all experiments
#   ./run.sh quick              # Quick test mode
#   ./run.sh exp1               # Run experiment 1 only
#   ./run.sh exp2               # Run experiment 2 only
#
# Environment:
#   Set DATA_DIR to your data directory (default: /mnt/user-data/uploads)
#   Set OUTPUT_DIR for results (default: ./results)

set -e

# Configuration
DATA_DIR="${DATA_DIR:-/mnt/user-data/uploads}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LAYERS="${LAYERS:-7,13,19,22}"

# Parse arguments
MODE="${1:-all}"

echo "========================================"
echo "MMIE Research: Multilingual Steering"
echo "========================================"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "Mode: $MODE"
echo "========================================"

# Check for data files
echo "Checking data files..."
for f in forget_hindi.jsonl retain_english.jsonl; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✓ $f found"
    else
        echo "  ✗ $f NOT FOUND"
    fi
done

# Run based on mode
case $MODE in
    quick)
        echo "Running quick test..."
        python main.py --experiment all --quick_test \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL" \
            --layers "$LAYERS"
        ;;
    exp1)
        echo "Running Experiment 1: Prompt Analysis..."
        python main.py --experiment prompt_analysis \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL" \
            --layers "$LAYERS"
        ;;
    exp2)
        echo "Running Experiment 2: SAE vs Direction..."
        python main.py --experiment sae_vs_direction \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL" \
            --layers "$LAYERS"
        ;;
    exp3)
        echo "Running Experiment 3: Layer Analysis..."
        python main.py --experiment layer_analysis \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL"
        ;;
    exp4)
        echo "Running Experiment 4: Tokenization..."
        python main.py --experiment tokenization \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL" \
            --layers "$LAYERS"
        ;;
    exp5)
        echo "Running Experiment 5: Adversarial..."
        python main.py --experiment adversarial \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL" \
            --layers "$LAYERS"
        ;;
    exp6)
        echo "Running Experiment 6: Clustering..."
        python main.py --experiment clustering \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL" \
            --layers "$LAYERS"
        ;;
    all)
        echo "Running all experiments..."
        python main.py --experiment all \
            --data_dir "$DATA_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --model_name "$MODEL" \
            --layers "$LAYERS" \
            --verbose
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: quick, exp1, exp2, exp3, exp4, exp5, exp6, all"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Done! Results saved to: $OUTPUT_DIR"
echo "========================================"
