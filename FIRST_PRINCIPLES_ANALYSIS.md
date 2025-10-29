# First-Principles Analysis: Will mmie.py Work for Research?

## Executive Summary

**Short Answer:** YES, but with critical caveats. The code will work for single-GPU research experiments with 1-2 seeds, but will fail with:
- Multiple seeds (memory leak)
- Multi-GPU setups (device assumption bugs)
- Invalid inputs (bootstrap bounds)
- Weight mismatches (silent failures)

**Algorithm Correctness:** ✅ Sound (SAE, LoRA, ReFT are standard)
**Implementation Quality:** ⚠️ Has bugs but workable for research with fixes

---

## 1. Dependency Analysis

### ✅ Core Dependencies (WILL WORK)
- `torch>=2.1` ✅
- `transformers>=4.41` ✅
- `peft>=0.11` ✅
- `scikit-learn>=1.3` ✅
- `numpy>=1.24` ✅

### ⚠️ Optional Dependencies (GRACEFUL DEGRADATION)
- `lid_ensemble.py` - ✅ EXISTS (verified)
- `backends/sae_lens_loader.py` - ✅ EXISTS (verified)
- `transliteration_utils.py` - ✅ EXISTS (verified)
- `langid` - ✅ Pure Python, no native deps
- `google-genai` - Optional (only for judge)
- `fasttext` - Optional (only if FASTTEXT_LID_PATH set)

**Verdict:** Dependencies are fine. Code handles missing optional deps gracefully.

---

## 2. Execution Flow Analysis

### Flow Logic (Will It Execute?)

```
1. Load data → ✅ read_jsonl() is robust
2. Load base model → ✅ Uses transformers API correctly
3. Select layers → ⚠️ Can fail if model structure unexpected
4. Train/load SAEs → ✅ Standard PyTorch training
5. Train/load LoRA → ⚠️ PEFT API usage correct but strict=False hides errors
6. Train/load ReFT → ✅ Custom implementation, looks correct
7. Evaluate → ✅ Standard metrics computation
```

**Critical Path Issues:**

1. **Layer Resolution** (Line 193): If model doesn't match expected structure, raises `AttributeError`. No fallback.
   - **Impact:** Will crash on unexpected model architectures
   - **Likelihood:** Low (most models follow standard patterns)

2. **Device Handling** (Lines 117-131): Assumes single device
   ```python
   dev = next(model.parameters()).device  # FAILS if model split across GPUs
   ```
   - **Impact:** Will crash on multi-GPU with `device_map="auto"`
   - **Likelihood:** Medium (only if using multi-GPU)

3. **Memory Management** (Lines 1777-1875): Models never deleted
   - **Impact:** OOM after 3-5 seeds
   - **Likelihood:** High (will happen every time)

---

## 3. Algorithm Correctness Analysis

### ✅ SAE Training (Lines 640-655)
- **MSE reconstruction loss** ✅ Standard
- **L1 sparsity penalty** ✅ Standard (`aux_coeff=1/32`)
- **TopK activation** ✅ Correct implementation
- **Training loop** ✅ Standard AdamW optimizer

**Verdict:** Algorithmically sound.

### ✅ Layer Selection (Lines 368-440)
- **CKA similarity** ✅ Debias centering correct
- **Procrustes similarity** ✅ SVD-based, correct
- **ANC (Average Neuron Correlation)** ✅ Pearson correlation per dim, correct
- **Semantic probes** ✅ Logistic regression AUC, standard

**Verdict:** Sound metrics, correctly implemented.

### ✅ LoRA Training (Lines 917-960)
- **PEFT integration** ✅ Uses official API
- **Gradient ascent on forget** ✅ Correct for unlearning
- **KL divergence to base on retain** ✅ Standard retain objective
- **State dict saving** ✅ Uses `get_peft_model_state_dict()`

**Verdict:** Correct implementation.

### ✅ ReFT Training (Lines 979-1009)
- **Low-rank adapter** ✅ Correct: `h + B(A(h))`
- **Hook-based injection** ✅ Correct forward hook usage
- **Training loop** ✅ Same objectives as LoRA

**Verdict:** Correct implementation.

### ✅ Evaluation Metrics
- **Extraction Strength** ✅ LID-based, reasonable proxy
- **Perplexity** ✅ Standard NLL computation
- **MIA** ✅ Correct: loss difference between base/edited
- **Probes** ✅ Logistic regression AUC

**Verdict:** Metrics are appropriate for the task.

---

## 4. Critical Bugs That Will Cause Failures

### 🔴 CRITICAL: Memory Leak (Lines 1777-1875)

**Problem:**
```python
for seed in args.seeds:  # Default: [42,43,44]
    lora = load_causal_lm(...)  # ~2-8GB GPU memory
    reft = load_causal_lm(...)   # ~2-8GB GPU memory
    # ... evaluation ...
    # NO CLEANUP!
```

**Impact:**
- Seed 1: ~4-16GB used ✅
- Seed 2: ~8-32GB used ⚠️
- Seed 3: ~12-48GB used ❌ OOM

**Fix Required:** Add cleanup after each seed iteration.

**Will It Work?** Yes for 1-2 seeds, no for 3+ seeds.

### 🔴 CRITICAL: Device Mismatch (Lines 496-498, 117-131)

**Problem:**
```python
# Line 496-498: Hook tries to move SAE to layer device
if sae_module.E.weight.device != h.device:
    sae_module.to(device=h.device)  # Moves module but doesn't update reference

# Line 117-131: Assumes single device
dev = next(model.parameters()).device  # FAILS if model split across GPUs
```

**Impact:**
- Single GPU: ✅ Works
- Multi-GPU with `device_map="auto"`: ❌ Crashes

**Fix Required:** Handle multi-device models properly.

**Will It Work?** Yes on single GPU, no on multi-GPU.

### 🟡 MODERATE: Bootstrap Bounds (Lines 96-97)

**Problem:**
```python
lo = boots[int(alpha/2*n_boot)]  # No bounds checking
hi = boots[int((1-alpha/2)*n_boot)-1]  # Can exceed array length
```

**Impact:**
- Normal usage (alpha=0.05): ✅ Works
- Invalid input (alpha>1): ❌ IndexError

**Fix Required:** Add input validation.

**Will It Work?** Yes with default args, no with invalid inputs.

### 🟡 MODERATE: Silent State Dict Errors (Multiple locations)

**Problem:**
```python
model.load_state_dict(sd, strict=False)  # Hides mismatches
```

**Impact:**
- Correct weights: ✅ Works
- Wrong rank/config: ⚠️ Silent failure, wrong results

**Fix Required:** Log warnings for missing keys.

**Will It Work?** Yes if weights match, silently wrong if they don't.

---

## 5. Will It Work for Your Research?

### Scenario 1: Single GPU, 1-2 Seeds, Standard Model
**Verdict:** ✅ **YES, WILL WORK**
- Memory leak won't cause immediate issues
- Device handling is correct for single GPU
- Algorithms are sound

### Scenario 2: Single GPU, 3+ Seeds
**Verdict:** ⚠️ **MAY WORK, BUT RISKY**
- Memory leak will cause OOM
- Need to add cleanup or reduce model size

### Scenario 3: Multi-GPU Setup
**Verdict:** ❌ **WILL FAIL**
- Device assumption bugs will crash
- Need to fix device handling

### Scenario 4: Custom Model Architecture
**Verdict:** ⚠️ **MAY FAIL**
- Layer resolution might fail
- Need to verify model structure matches expected patterns

---

## 6. Quick Fixes for Research Use

### Fix 1: Add Memory Cleanup (REQUIRED for multiple seeds)
```python
for seed in args.seeds:
    lora = load_causal_lm(...)
    reft = load_causal_lm(...)
    try:
        # ... evaluation ...
    finally:
        if gate: gate.remove()
        if scrub: scrub.remove()
        del lora, reft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### Fix 2: Add Bootstrap Validation (RECOMMENDED)
```python
def bootstrap_ci(values, alpha=0.05, n_boot=2000, seed=0):
    if not values or alpha <= 0 or alpha >= 1:
        return (float('nan'), (float('nan'), float('nan')))
    # ... rest of code with bounds checking ...
```

### Fix 3: Fix Device Handling (REQUIRED for multi-GPU)
```python
def _get_model_device(model):
    # Handle multi-device models
    devices = {p.device for p in model.parameters()}
    if len(devices) == 1:
        return next(iter(devices))
    # For multi-device, return primary device or raise error
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
```

---

## 7. Research Workflow Recommendations

### ✅ Safe Research Workflow
1. **Start with 1 seed** - Verify everything works
2. **Use single GPU** - Avoid device bugs
3. **Monitor memory** - Add `torch.cuda.memory_allocated()` logging
4. **Check outputs** - Verify results make sense before running multiple seeds
5. **Apply fixes** - Add memory cleanup before scaling up

### ⚠️ Experimental Workflow
1. Test with TinyLlama first (small model)
2. Verify SAE quality metrics
3. Check that selected layers make sense
4. Verify extraction strength decreases as expected
5. Scale up gradually

---

## 8. Final Verdict

**Will it work for research?**

**YES, with conditions:**
- ✅ Single GPU setup
- ✅ 1-2 seeds maximum (or apply memory fix)
- ✅ Standard model architectures
- ✅ Valid input parameters

**NO, if:**
- ❌ Multi-GPU setup (without device fixes)
- ❌ Many seeds (without memory cleanup)
- ❌ Custom architectures (may need layer resolution fixes)
- ❌ Invalid inputs (need validation)

**Recommendation:**
The code is **sufficient for research experiments** but needs the memory cleanup fix for multi-seed runs. The algorithms are sound, the implementation is mostly correct, but has known bugs that are fixable. For a research prototype, this is acceptable - but plan to fix the memory leak before running production-scale experiments.

---

## 9. Testing Checklist

Before running your experiment:

- [ ] Verify all dependencies installed (`pip install -r requirements.txt`)
- [ ] Test with 1 seed first
- [ ] Check GPU memory usage after each seed
- [ ] Verify model loads correctly
- [ ] Check that layer selection produces reasonable results
- [ ] Verify SAE training completes without errors
- [ ] Check that evaluation metrics are computed
- [ ] If using multiple seeds, apply memory cleanup fix

---

## 10. Expected Outcomes

If everything works correctly, you should get:
- ✅ JSON report with ES, PPL, MIA metrics
- ✅ Layer selection results
- ✅ SAE quality metrics (if enabled)
- ✅ Comparison between LoRA and ReFT+SAE approaches
- ✅ Gate decisions (PROCEED/STOP)

If things fail:
- ❌ OOM error → Memory leak (needs cleanup)
- ❌ Device mismatch → Multi-GPU issue (needs device fix)
- ❌ AttributeError on layer resolution → Unsupported model (needs adapter)
- ❌ IndexError in bootstrap → Invalid alpha (needs validation)

