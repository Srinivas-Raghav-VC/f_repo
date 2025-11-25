#!/bin/bash
# ============================================================================
# MMIE COMPREHENSIVE VERIFICATION PIPELINE
# ============================================================================
# Run these commands in order to fully validate the methodology
#
# Based on:
# - OpenAI "Scaling and evaluating sparse autoencoders" (2024)
# - Anthropic "Scaling Monosemanticity" (2024)
# - RMU "Representation Misdirection for Unlearning" (AAAI 2025)
# - TOFU benchmark methodology
# ============================================================================

echo "=============================================="
echo "MMIE VERIFICATION PIPELINE"
echo "=============================================="

# ============================================================================
# STEP 0: SETUP
# ============================================================================
echo ""
echo "[STEP 0] Setup and Data Check"
echo "=============================================="

# Check data files exist
echo "Checking data files..."
ls -la data/*.jsonl 2>/dev/null || echo "WARNING: No data files in data/"

# Check GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ============================================================================
# STEP 1: QUICK SMOKE TEST (10-15 minutes)
# ============================================================================
echo ""
echo "[STEP 1] Quick Smoke Test"
echo "=============================================="
echo "This tests:"
echo "  - Tokenization bias"
echo "  - SAE architectures (TopK vs JumpReLU)"
echo "  - Layer importance (3 methods)"
echo "  - Intervention methods"
echo "  - Complete evaluation on all data"
echo ""

python mmie_smoke_test.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data_dir data \
    --out smoke_test_quick.json \
    --quick

echo ""
echo "Results saved to: smoke_test_quick.json"
echo ""

# ============================================================================
# STEP 2: LAYER COMPARISON (20-30 minutes)
# ============================================================================
echo ""
echo "[STEP 2] Layer Region Comparison"
echo "=============================================="
echo "Testing early vs mid vs late layers"
echo ""

# Early layers (4-8)
echo "[2a] Testing EARLY layers (4-8)..."
python mmie_v8_fixed.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data_dir data \
    --layers 4 5 6 7 8 \
    --sae_type jumprelu \
    --out results_early_layers.json \
    --plots_dir plots_early

# Mid layers (13-17)
echo "[2b] Testing MID layers (13-17)..."
python mmie_v8_fixed.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data_dir data \
    --layers 13 14 15 16 17 \
    --sae_type jumprelu \
    --out results_mid_layers.json \
    --plots_dir plots_mid

# Late layers (21-25)
echo "[2c] Testing LATE layers (21-25)..."
python mmie_v8_fixed.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data_dir data \
    --layers 21 22 23 24 25 \
    --sae_type jumprelu \
    --out results_late_layers.json \
    --plots_dir plots_late

# ============================================================================
# STEP 3: COMPARE LAYER RESULTS
# ============================================================================
echo ""
echo "[STEP 3] Comparing Layer Results"
echo "=============================================="

python3 << 'EOF'
import json

def load_results(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

early = load_results("results_early_layers.json")
mid = load_results("results_mid_layers.json")
late = load_results("results_late_layers.json")

print("\n" + "="*60)
print("LAYER REGION COMPARISON")
print("="*60)

for name, data in [("EARLY (4-8)", early), ("MID (13-17)", mid), ("LATE (21-25)", late)]:
    if not data:
        print(f"\n{name}: No results")
        continue
    
    print(f"\n{name}:")
    
    # Get layer results
    layers = data.get("layer_selection", {}).get("layers", {})
    if layers:
        # Find best layer
        best_layer = max(layers.keys(), key=lambda l: layers[l].get("es_reduction", -999))
        best_reduction = layers[best_layer].get("es_reduction", 0)
        best_ppl = layers[best_layer].get("ppl_increase", 0)
        
        print(f"  Best layer: {best_layer}")
        print(f"  ES reduction: {best_reduction:.3f} ({'✓ REDUCES' if best_reduction > 0 else '✗ INCREASES'})")
        print(f"  PPL increase: {best_ppl*100:.1f}%")
        
        # Count effective layers
        effective = [l for l, d in layers.items() if d.get("es_reduction", 0) > 0]
        print(f"  Layers that reduce Hindi: {len(effective)}/{len(layers)}")

# Recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

best_region = None
best_score = -999

for name, data in [("early", early), ("mid", mid), ("late", late)]:
    if not data:
        continue
    layers = data.get("layer_selection", {}).get("layers", {})
    if layers:
        # Score = avg ES reduction - avg PPL increase
        reductions = [d.get("es_reduction", 0) for d in layers.values()]
        ppl_increases = [d.get("ppl_increase", 0) for d in layers.values()]
        score = sum(reductions)/len(reductions) - sum(ppl_increases)/len(ppl_increases)
        
        if score > best_score:
            best_score = score
            best_region = name

if best_region:
    print(f"\n→ Use {best_region.upper()} layers (best ES/PPL tradeoff)")
else:
    print("\n→ Could not determine best region")

EOF

# ============================================================================
# STEP 4: FULL SMOKE TEST (30-45 minutes)
# ============================================================================
echo ""
echo "[STEP 4] Full Smoke Test (Optional - more thorough)"
echo "=============================================="
echo "Run this for comprehensive validation:"
echo ""
echo "python mmie_smoke_test.py \\"
echo "    --model Qwen/Qwen2.5-1.5B-Instruct \\"
echo "    --data_dir data \\"
echo "    --out smoke_test_full.json"
echo ""

# ============================================================================
# STEP 5: RESULTS SUMMARY
# ============================================================================
echo ""
echo "[STEP 5] Results Summary"
echo "=============================================="

python3 << 'EOF'
import json
import os

print("\nFILES GENERATED:")
for f in ["smoke_test_quick.json", "results_early_layers.json", 
          "results_mid_layers.json", "results_late_layers.json"]:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  ✓ {f} ({size} bytes)")
    else:
        print(f"  ✗ {f} (not found)")

# Load smoke test results
if os.path.exists("smoke_test_quick.json"):
    with open("smoke_test_quick.json") as f:
        results = json.load(f)
    
    print("\nSMOKE TEST FINDINGS:")
    
    # Tokenization
    tok = results.get("test1_tokenization", {})
    if tok.get("bias_ratio"):
        print(f"  Tokenization bias: {tok['bias_ratio']:.1f}x (Hindi vs English)")
    
    # SAE
    sae = results.get("test2_sae", {})
    if sae.get("best"):
        print(f"  Best SAE: {sae['best']}")
    if sae.get("topk_status") == "FAILING":
        print(f"  ⚠️ TopK SAE is FAILING - use JumpReLU")
    
    # Layers
    layers = results.get("test3_layers", {})
    if layers.get("layers_that_reduce_hindi"):
        print(f"  Effective layers: {layers['layers_that_reduce_hindi']}")
    else:
        print(f"  ⚠️ No layers reduce Hindi with direction method")
    
    # Interventions
    ints = results.get("test4_interventions", {})
    if ints.get("best_method"):
        print(f"  Best intervention: {ints['best_method']}")
    
    # Issues
    if results.get("issues"):
        print(f"\n⚠️ ISSUES TO ADDRESS:")
        for issue in results["issues"]:
            print(f"    - {issue}")
    
    # Recommendations
    if results.get("recommendations"):
        rec = results["recommendations"]
        print(f"\nRECOMMENDED CONFIGURATION:")
        print(f"  SAE: {rec.get('sae', 'jumprelu')}")
        print(f"  Layers: {rec.get('layers', 'mid-layers')}")
        print(f"  Method: {rec.get('method', 'direction_toward_english')}")

EOF

echo ""
echo "=============================================="
echo "VERIFICATION COMPLETE"
echo "=============================================="
