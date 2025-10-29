# Critical Code Review Summary: mmie.py

## 🔴 Must Fix Immediately

### 1. Memory Leak in Main Loop (Lines 1777-1875)
**Problem:** Models loaded per seed iteration are never deleted, causing GPU memory to fill up.

```python
# Current problematic code:
for seed in args.seeds:
    lora = load_causal_lm(...)  # Loaded but never deleted
    reft = load_causal_lm(...)  # Loaded but never deleted
    # ... evaluation ...
    # No cleanup!
```

**Fix:**
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

**Impact:** Can cause OOM errors after 2-3 seeds on GPU.

---

### 2. Bootstrap Confidence Interval Index Error (Lines 96-97)
**Problem:** Array indexing can go out of bounds.

```python
# Current code:
lo = boots[int(alpha/2*n_boot)]  # Can be negative or out of bounds
hi = boots[int((1-alpha/2)*n_boot)-1]  # Can exceed array length
```

**Fix:**
```python
if not values or alpha <= 0 or alpha >= 1:
    return (float('nan'), (float('nan'), float('nan')))
# ... existing code ...
lo_idx = max(0, int(np.floor(alpha/2*n_boot)))
hi_idx = min(len(boots)-1, int(np.ceil((1-alpha/2)*n_boot))-1)
lo = boots[lo_idx]
hi = boots[hi_idx]
```

---

### 3. Silent Exception Swallowing
**Problem:** Critical errors are hidden by `except Exception: pass`.

**Examples:**
- Lines 31-32, 38-39: Import failures hidden
- Lines 1542-1543: Judge refinement failures ignored
- Lines 1622-1623: Judge assist failures ignored

**Impact:** Debugging is impossible, metrics may be wrong.

**Fix:** Replace with logging:
```python
except Exception as e:
    if args.log_verbose:
        print(f"[error] Judge refinement failed: {e}")
    # Optionally continue with fallback
```

---

### 4. Device Mismatch in Hooks
**Problem:** SAE hooks try to move modules to device but don't update references.

**Location:** Lines 496-498 (`SAEGate._attach`)

**Fix:**
```python
if sae_module.E.weight.device != h.device:
    sae_module = sae_module.to(device=h.device)
    self.sae[li] = sae_module  # Update reference
```

---

## 🟡 Should Fix Soon

### 5. Dynamic Gating State Mutation
**Problem:** `DynamicGatingLogitsProcessor` modifies shared gate state per sequence.

**Location:** Lines 1063-1082

**Issue:** If batches processed concurrently, alpha values conflict.

**Fix:** Use per-sequence alpha tracking or ensure sequential processing.

---

### 6. Non-Reproducible Random Seeds
**Problem:** Seeds set in try/except blocks that can fail silently.

**Location:** Lines 1423-1438, 1499-1504

**Fix:** Use context manager:
```python
@contextmanager
def set_seed_context(seed):
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
```

---

### 7. Inefficient Activation Collection
**Problem:** Same activations computed multiple times.

**Location:** `select_layers()` calls `collect_layer_means()` twice for same data.

**Fix:** Cache activations per (layer, text_set) combination.

---

## 🔍 Code Understanding Questions

1. **Why is `base` model loaded once but never deleted?** (Line 1469)
   - Should be deleted after use or explicitly kept in memory?

2. **What happens if `_resolve_blocks()` fails?** (Line 193)
   - Only raises error, no fallback strategy.

3. **Why are SAE modules stored in dict but never cleaned up?** (Line 1749)
   - Could accumulate memory.

4. **What's the purpose of `strict=False` in state_dict loading?** (Multiple locations)
   - Should validate shapes even if strict=False.

---

## 📝 Quick Wins

1. **Add docstrings** to public functions
2. **Extract magic numbers** to constants
3. **Split `main()` function** into smaller pieces
4. **Add type hints** to function signatures
5. **Validate configuration** before use

---

## 🎯 Testing Recommendations

1. **Test with 1 seed** - Verify no memory leaks
2. **Test with 10+ seeds** - Catch memory accumulation
3. **Test with different devices** - Verify device handling
4. **Test with invalid inputs** - Verify error handling
5. **Test bootstrap CI** - Verify boundary conditions

---

## 📊 Risk Assessment

| Issue | Severity | Likelihood | Priority |
|-------|----------|------------|----------|
| Memory leak | Critical | High | P0 |
| Bootstrap bounds | Critical | Medium | P0 |
| Silent exceptions | High | High | P1 |
| Device mismatch | High | Medium | P1 |
| Seed reproducibility | Medium | High | P2 |
| Code organization | Low | N/A | P3 |

---

## Next Steps

1. ✅ Fix memory leaks (P0)
2. ✅ Fix bootstrap bounds (P0)
3. ✅ Add exception logging (P1)
4. ✅ Refactor main() function (P2)
5. ✅ Add comprehensive tests (P2)

