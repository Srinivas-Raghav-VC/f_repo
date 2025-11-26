#!/bin/bash
# MMIE Complete Research Suite
# =============================
#
# Usage:
#   ./run.sh              # Run all experiments (full)
#   ./run.sh quick        # Quick mode (fewer samples)
#   ./run.sh semantic     # Run only semantic analysis
#   ./run.sh steering     # Run only steering grid search
#   ./run.sh causality    # Run only causality test
#   ./run.sh coherence    # Run only coherence test
#   ./run.sh crosslang    # Run only cross-language test
#   ./run.sh sae          # Run only SAE analysis
#   ./run.sh adversarial  # Run only adversarial test

set -e

# Configuration
DATA_DIR="${DATA_DIR:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

# Parse arguments
MODE="${1:-all}"
QUICK_FLAG=""

if [ "$MODE" = "quick" ]; then
    QUICK_FLAG="--quick"
    MODE="all"
fi

if [ "$2" = "quick" ]; then
    QUICK_FLAG="--quick"
fi

echo "========================================"
echo "MMIE Complete Research Suite"
echo "========================================"
echo "Mode: $MODE"
echo "Data dir: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Quick mode: ${QUICK_FLAG:-no}"
echo "========================================"

# Check data files
echo ""
echo "Checking data files..."
for f in forget_hindi.jsonl retain_english.jsonl; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f (missing)"
    fi
done

echo ""

# Run
python main.py \
    --experiment "$MODE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    $QUICK_FLAG

echo ""
echo "========================================"
echo "Done! Results in: $OUTPUT_DIR"
echo "========================================"
