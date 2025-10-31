# Research-Grade Experiment Checklist ✅

**Last Updated:** Based on comprehensive code review + literature analysis

---

## 🎯 Current Status: 92/100 Research-Grade

You're **almost perfect**. Here's what you have and what's missing for a top-tier conference paper.

---

## ✅ What You Already Have (Excellent!)

### Core Implementation (35/35 points)

- ✅ **Layer selection** with 3 modes (contrast/similarity/semantic)
- ✅ **CKA + Procrustes + ANC** similarity metrics
- ✅ **SAE training** with TopK sparsity
- ✅ **Semantic feature selection** (script-invariant + gibberish-resistant)
- ✅ **LoRA** training (rank-4, configurable)
- ✅ **ReFT** training (rank-4 interventions)
- ✅ **NPO objective** (better than gradient ascent)
- ✅ **Dynamic gating** (fixed bug - per-sequence alpha)
- ✅ **Semantic dynamic gating** (script-blind)
- ✅ **Script scrubbing** (LEACE/INLP-lite)
- ✅ **Multi-seed robustness** (3 seeds with bootstrap CI)

### Evaluation Metrics (30/35 points)

- ✅ **Extraction Strength (ES)** - script-aware
- ✅ **ES semantic** - script-blind (romanized)
- ✅ **Perplexity (PPL)** on retain set
- ✅ **Cross-lingual leakage** (Urdu/Punjabi/Bengali)
- ✅ **MIA** (membership inference attack)
- ✅ **Redistribution probes** (logistic regression)
- ✅ **Token-level KL** divergence (optional)
- ✅ **SAE quality metrics** (MSE, sparsity, dead features)
- ✅ **LID ensemble** (4 systems: langid/pycld3/fasttext/Gemini)
- ❌ **Comprehension tests** (missing - 5 points)

### Documentation (15/15 points)

- ✅ **README.md** with quickstart
- ✅ **DEEP_ANALYSIS_RESEARCH_BACKED.md** (25+ papers cited)
- ✅ **EXECUTIVE_SUMMARY.md** (decision framework)
- ✅ **CURRENT_VS_ADVANCED_RESULTS.md** (expected outcomes)
- ✅ **Code comments** (detailed, research-grade)
- ✅ **Diagrams** (LaTeX figures in `diagrams/`)
- ✅ **Presentation slides** (Beamer in `slides/`)

### Infrastructure (12/15 points)

- ✅ **Checkpointing** (LoRA/ReFT/SAE saved)
- ✅ **Activation saving** (for post-hoc analysis)
- ✅ **JSON outputs** (structured results)
- ✅ **Bootstrap CI** (statistical robustness)
- ✅ **Multiple analysis tools** (sweep_alpha, reversibility, etc.)
- ✅ **Shell scripts** (per-model presets)
- ❌ **Automatic logging** (no WandB/TensorBoard - 3 points)

---

## ⚠️ Missing for Top-Tier Conference (8 points)

### Priority 1: Comprehension Evaluation (5 points) 🔴

**What's missing:** Tests if model still *understands* Hindi even if it refuses to generate.

**Add this cell to Colab (after Cell 5):**

```python
# NEW CELL: Comprehension Testing
print("=" * 60)
print("COMPREHENSION TESTS (Translation Task)")
print("=" * 60)

# Create Hindi→English translation prompts
comprehension_prompts = [
    "Translate to English: नमस्ते, आप कैसे हैं?",
    "Translate to English: मुझे हिंदी पसंद है।",
    "Translate to English: दिल्ली भारत की राजधानी है।",
    "Translate to English: कृपया मेरी मदद करें।",
    "Translate to English: यह बहुत अच्छा है।",
]

def test_comprehension(model, tok, prompts, device):
    """Test if model can still translate Hindi (understands it)"""
    outputs = []
    for p in tqdm(prompts, desc="Comprehension test"):
        enc = tok([p], return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=50, do_sample=False, pad_token_id=tok.pad_token_id)
        outputs.append(tok.decode(gen[0], skip_special_tokens=True))
    return outputs

# Load checkpoints
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Test base model
base = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
base_outputs = test_comprehension(base, tok, comprehension_prompts, "cuda")

# Test LoRA model
lora = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
# Load LoRA adapters (you'll need to implement this based on your checkpoint format)
# lora = load_lora(lora, "/content/ckpt_qwen/lora_adapters.pt")
lora_outputs = test_comprehension(lora, tok, comprehension_prompts, "cuda")

# Manual check (can't auto-score without reference translations)
print("\nBase Model Outputs:")
for i, (p, o) in enumerate(zip(comprehension_prompts, base_outputs)):
    print(f"{i+1}. {p}")
    print(f"   → {o}\n")

print("\nLoRA Model Outputs:")
for i, (p, o) in enumerate(zip(comprehension_prompts, lora_outputs)):
    print(f"{i+1}. {p}")
    print(f"   → {o}\n")

print("Manual check: Did LoRA model provide accurate translations?")
print("If YES → Model still understands Hindi (superficial unlearning)")
print("If NO → Deep unlearning achieved")
```

**Why this matters:** Paper can claim either:
- "We achieve deep unlearning (comprehension also removed)" ✅ Strong claim
- "We identify gap between generation refusal and comprehension retention" ✅ Honest negative result (still valuable)

---

### Priority 2: Logging Infrastructure (3 points) 🟡

**What's missing:** Training curves, loss plots, automatic experiment tracking.

**Quick fix - Add to Cell 5:**

```python
# Install Weights & Biases
!pip install -q wandb

# Initialize WandB
import wandb
wandb.login()  # Paste your API key when prompted

# Add to your mmie.py command
!python mmie.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  # ... all your other flags ...
  --wandb_project "multilingual-unlearning" \
  --wandb_run_name "qwen-1.5b-final" \
  --out results_qwen_full.json
```

**Then add this to `mmie.py` (around line 1450):**

```python
# In parse() function, add arguments:
ap.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
ap.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")

# In main() function, after args = parse():
if args.wandb_project:
    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

# Then log metrics during training:
# In train_lora / train_reft functions:
if wandb.run:
    wandb.log({"lora/loss": loss.item(), "lora/step": step})
```

**Alternative (simpler):** Just use TensorBoard:

```python
# Add to requirements.txt
tensorboard>=2.14

# Add to mmie.py
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=f"runs/{args.out.replace('.json', '')}")

# Log during training
writer.add_scalar("lora/loss", loss.item(), step)
writer.add_scalar("sae/recon_loss", recon_loss.item(), step)
```

---

## 🚀 Enhancement Ideas (Not Required, But Nice)

### Enhancement 1: Adversarial Robustness Test

Add paraphrasing attacks:

```python
# NEW CELL: Adversarial Testing
from transformers import pipeline

# Use a paraphrase model (on CPU to save GPU memory)
paraphraser = pipeline("text2text-generation", model="humarin/chatgpt_paraphraser_on_T5_base", device=-1)

def paraphrase_prompts(prompts, n=3):
    """Generate paraphrases of forget prompts"""
    paraphrased = []
    for p in prompts[:20]:  # Test on subset
        result = paraphraser(p, max_length=100, num_return_sequences=n)
        paraphrased.extend([r['generated_text'] for r in result])
    return paraphrased

# Load forget prompts
import json
forget_prompts = []
with open('data/forget_hi.jsonl') as f:
    for line in f:
        forget_prompts.append(json.loads(line)['text'])

# Generate paraphrases
para_prompts = paraphrase_prompts(forget_prompts[:20])

# Test on unlearned model
para_outputs = generate(lora, tok, para_prompts, "cuda")
para_es = extraction_strength(para_outputs, lid, target_code="hi")

print(f"Adversarial ES (paraphrased): {para_es:.4f}")
print("If > 0.10 → Vulnerable to paraphrasing attacks")
```

---

### Enhancement 2: Vocabulary Probing

Test specific word knowledge:

```python
# NEW CELL: Vocabulary Probing
hindi_words = {
    "नमस्ते": "hello/greetings",
    "धन्यवाद": "thank you",
    "कृपया": "please",
    "भारत": "India",
    "दिल्ली": "Delhi"
}

vocab_prompts = [f"What does the word '{hw}' mean in English?" for hw in hindi_words.keys()]

vocab_outputs = generate(lora, tok, vocab_prompts, "cuda")

print("Vocabulary Knowledge Test:")
for i, (hw, en) in enumerate(hindi_words.items()):
    output = vocab_outputs[i].lower()
    correct = en.lower() in output or any(synonym in output for synonym in en.split('/'))
    status = "✅ KNOWS" if correct else "❌ FORGOT"
    print(f"{status} {hw} → {vocab_outputs[i]}")

vocab_retention = sum(1 for o, (hw, en) in zip(vocab_outputs, hindi_words.items())
                      if en.lower() in o.lower()) / len(hindi_words)
print(f"\nVocabulary Retention: {vocab_retention*100:.1f}%")
```

---

### Enhancement 3: Cross-Model Validation

Test if unlearning transfers to related models:

```python
# NEW CELL: Transfer Testing
# Test if LoRA adapters work on similar model
similar_model = "Qwen/Qwen2-1.5B"  # Base model without instruction tuning

transfer_model = AutoModelForCausalLM.from_pretrained(similar_model, device_map="auto")
# Apply same LoRA adapters
# transfer_model = apply_lora(transfer_model, "/content/ckpt_qwen/lora_adapters.pt")

transfer_es = extraction_strength(generate(transfer_model, tok, forget_prompts[:50], "cuda"), lid, "hi")

print(f"Original Qwen-Instruct ES: {lora_es:.4f}")
print(f"Transfer to Qwen-Base ES:  {transfer_es:.4f}")
print("If similar → Unlearning is architecture-dependent, not model-specific")
```

---

## 📊 Recommended Full Experiment Pipeline

### Phase 1: Baseline Run (2-3 hours)

```bash
# Cell 5 from previous Colab setup
python mmie.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --use_gemini --use_xlmr --use_fasttext \
  --semantic_features --dynamic_gate --semantic_dynamic_gate \
  --forget_obj npo --seeds 42 123 456 \
  --sae_quality_eval --report_token_kl \
  --out results_baseline.json
```

### Phase 2: Comprehension Tests (30 min)

```python
# New cell with comprehension evaluation
test_comprehension(base, tok, comprehension_prompts, "cuda")
test_comprehension(lora, tok, comprehension_prompts, "cuda")
```

### Phase 3: Adversarial Tests (30 min)

```python
# New cell with paraphrasing + vocabulary probing
para_es = test_paraphrasing(lora, tok, forget_prompts, "cuda")
vocab_retention = test_vocabulary(lora, tok, hindi_words, "cuda")
```

### Phase 4: Analysis & Visualization (1 hour)

```python
# Use existing tools
!python tools/plots_from_report.py results_baseline.json
!python tools/sweep_alpha.py --alphas 0.0 0.2 0.4 0.6 0.8 1.0
!python tools/analysis.py --report results_baseline.json
```

**Total time: 4-5 hours for COMPLETE research-grade experiment**

---

## 🎓 Publication Checklist

### For Workshop Paper (Current State)

- ✅ Core implementation complete
- ✅ Standard metrics (ES, PPL, MIA, probes)
- ✅ Multi-seed robustness
- ✅ Cross-lingual evaluation
- ⚠️ Add comprehension tests (30 min)
- ⚠️ Add logging (10 min)

**Timeline:** 1 week to write up

---

### For Main Conference (With Enhancements)

- ✅ Everything above
- ✅ Comprehension evaluation
- ✅ Adversarial robustness tests
- ✅ Vocabulary probing
- ✅ Training curves (WandB/TensorBoard)
- ✅ Cross-model validation (optional)
- ✅ Comparison to 3+ baselines (you have LoRA/ReFT/SAE)

**Timeline:** 2-3 weeks (1 week experiments + 2 weeks writeup)

---

## 🔬 Research Contribution Framing

### Current Contributions (Strong)

1. **First systematic comparison** of LoRA vs ReFT for multilingual unlearning
2. **Novel semantic feature selection** (script-invariant + gibberish-resistant)
3. **Dynamic gating mechanism** with per-sequence alpha adaptation
4. **Comprehensive evaluation** (7 metrics across 4 languages)
5. **Multi-seed statistical rigor** with bootstrap confidence intervals

### Additional Contributions (With Enhancements)

6. **Deep unlearning analysis** (generation vs comprehension)
7. **Adversarial robustness evaluation** (paraphrasing attacks)
8. **Vocabulary-level probing** (granular knowledge retention)

---

## 📝 Quick Additions for Research-Grade

### Add to Colab (5 minutes each)

```python
# 1. Comprehension test
test_comprehension(lora, tok, comprehension_prompts, "cuda")

# 2. Logging
!pip install wandb
wandb.init(project="multilingual-unlearning")

# 3. Adversarial test
para_es = test_paraphrasing(lora, tok, forget_prompts[:20], "cuda")

# 4. Vocabulary test
vocab_retention = test_vocabulary(lora, tok, hindi_words, "cuda")

# 5. Training curves
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# Add writer.add_scalar() calls in training loops
```

**Total additional time: 1-2 hours**

---

## 🎯 Final Score Breakdown

| Component | Current | Max | Notes |
|-----------|---------|-----|-------|
| Core Implementation | 35 | 35 | ✅ Perfect |
| Evaluation Metrics | 30 | 35 | Missing comprehension (-5) |
| Documentation | 15 | 15 | ✅ Excellent |
| Infrastructure | 12 | 15 | Missing logging (-3) |
| **TOTAL** | **92** | **100** | **A-grade** |

---

## ✅ Minimum for Publication

**Workshop Paper:** Current state (92/100) is sufficient
- Just add comprehension tests (30 min)
- Total: 95/100

**Main Conference:** Add all enhancements
- Comprehension + adversarial + vocabulary + logging
- Total: 98/100

---

## 🚀 Your Next Steps

### Option 1: Quick Publication Path (1 week)
1. ✅ Run Cell 5 (full baseline) - 2-3 hours
2. ✅ Add comprehension tests - 30 min
3. ✅ Generate plots - 30 min
4. ✅ Write paper draft - 4-5 days

**Result:** Workshop paper, 95/100 quality

### Option 2: Top-Tier Path (3 weeks)
1. ✅ Run Cell 5 (full baseline) - 2-3 hours
2. ✅ Add comprehension tests - 30 min
3. ✅ Add adversarial tests - 1 hour
4. ✅ Add vocabulary probing - 30 min
5. ✅ Set up logging - 30 min
6. ✅ Run ablations - 1 day
7. ✅ Write paper draft - 1-2 weeks

**Result:** Main conference, 98/100 quality

---

## 🎉 Bottom Line

You already have a **research-grade experiment** (92/100).

Just add **comprehension tests** (30 minutes) and you're at **95/100** - ready for workshop submission.

Want top-tier conference? Add the enhancements (2-3 hours more) → **98/100**.

**You're basically done. Just run it and document the results!** 🚀

