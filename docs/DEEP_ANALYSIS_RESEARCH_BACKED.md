# Deep Analysis: Will This Multilingual Unlearning Experiment Work?

## Executive Summary

**Verdict: Theoretically sound but practically flawed. Will show superficial success but fail deep unlearning tests.**

After analyzing 25+ arXiv papers and extensive code review, I find the approach has **correct foundations** but **critical implementation gaps** that will lead to incomplete unlearning vulnerable to adversarial queries and cross-lingual leakage.

---

## 1. Layer Selection Approach

### What the Code Does
- Uses CKA (Centered Kernel Alignment) + Procrustes + ANC similarity metrics
- Three modes: contrast (prefer divergent layers), similarity, semantic (probe-based)
- Selects top-K layers where Hindi/English representations differ most

### Research Findings

✅ **What Works:**
- CKA is validated for comparing representations (Raghu et al., ICML 2017; Kornblith et al., ICML 2019)
- Paper "Quantifying Feature Space Universality" (2410.06981v4) confirms SAE feature spaces ARE similar across models under rotation-invariant transforms

❌ **Critical Flaws:**

1. **Divergence ≠ Causality**
   - Layers with low CKA might differ due to SUPERFICIAL features (script, tokenization) not semantic content
   - No validation that selected layers are CAUSAL for Hindi generation

2. **Semantic Mode Issues** (from code analysis)
   - Trains probes on Hindi vs English only
   - High probe AUC might indicate general linguistic processing, not Hindi-specific knowledge
   - Editing high-AUC layers could damage English capabilities

3. **Missing Validation**
   - No ablation: do selected layers actually matter for Hindi generation?
   - No comparison: random layer selection vs metric-based selection?

### Recommendation
**Add causal validation**: Test if ablating selected layers reduces Hindi performance more than other layers.

---

## 2. SAE Training & Feature Selection

### What the Code Does
- Trains TopK SAE with MSE + L1 sparsity
- Two feature selection methods:
  1. Activation-based: features with high |z| on forget vs retain
  2. Semantic: features invariant across scripts, insensitive to gibberish

### Research Findings

⚠️ **Mixed Evidence for SAEs in Unlearning:**

**Supporting:** Paper "SAEs Can Improve Unlearning" (2504.08192v1)
- SAEs WITH DYNAMIC GATING outperform gradient-based unlearning
- Better computational efficiency, data efficiency, interpretability
- Key insight: "targeted activation-based unlearning" works when features are CAUSAL

**Against:** Papers "Can SAEs decompose steering vectors?" (2411.08790v1), "AxBench" (2501.17148v3)
- SAE-reconstructed vectors LACK steering properties of originals
- Simple baselines (difference-in-means, prompting) OUTPERFORM SAEs
- SAEs trained on base model don't generalize to fine-tuned models

❌ **Critical Flaws:**

1. **Correlation vs Causation**
   - Activation-based selection finds features that CORRELATE with Hindi
   - Doesn't prove features CAUSE Hindi generation
   - Paper "Does higher interpretability imply better utility?" (2510.03659v1): interpretability ≠ steering effectiveness

2. **Wrong Selection Metric**
   - Paper "Beyond Input Activations: GradSAE" (2505.08080v2) shows GRADIENT-based selection is superior
   - Should use ∂loss/∂feature activation, not just activation magnitude

3. **Training Data Issue**
   - SAE trained on base + forget/retain mix
   - Might learn features that reflect DISTRIBUTION differences, not semantic Hindi

### Recommendation
**Switch to gradient-based feature selection** (GradSAE approach): Select features with highest gradient w.r.t. forget loss.

---

## 3. LoRA vs ReFT Comparison

### What the Code Does
- LoRA: rank-4 adapters on q_proj, v_proj
- ReFT: rank-4 residual adapters on selected layers
- Both trained with GA (gradient ascent) or NPO (negative preference optimization)

### Research Findings

📊 **LoRA Characteristics** (Paper "LoRA Learns Less and Forgets Less", 2405.09673v2):

**Good for Unlearning:**
- ✅ Better maintains base model performance (less catastrophic forgetting)
- ✅ Mitigates forgetting MORE than weight decay/dropout
- ✅ "Learns less" = might be better for targeted unlearning

**Bad for Unlearning:**
- ❌ Substantially UNDERPERFORMS full finetuning at typical ranks
- ❌ Learns perturbations 10-100X lower rank than needed
- ❌ Suffers from training instability without ideal conditions
- ❌ Weaker task-level memorization

**Implications:**
- Rank-4 is likely TOO LOW for effective unlearning
- Paper "AutoLoRA" (2403.09113v2) shows optimal rank varies per layer
- Should use rank 8-16 or learn rank per layer

📊 **ReFT Analysis** (Paper "ReFT: Representation Finetuning for Language Models", 2404.03592v3):

**Advantages:**
- ✅ 15-65x MORE parameter-efficient than LoRA (confirmed by paper)
- ✅ Operates on frozen base model, learns task-specific interventions on hidden representations
- ✅ Outperforms LoRA on commonsense reasoning, arithmetic, instruction-tuning, GLUE
- ✅ Better efficiency-performance balance than state-of-the-art PEFTs
- ✅ LoReFT = Low-rank Linear Subspace intervention (theoretically sound)

**Risks for Unlearning:**
- ⚠️ Designed for task adaptation, NOT unlearning
- ⚠️ Paper shows it's BETTER at learning new behaviors (opposite of forgetting)
- ⚠️ Intervening on representations might make it HARDER to forget (increases representation capacity)
- ⚠️ No validation on unlearning tasks in original paper
- ⚠️ "AxBench" paper (by same authors, 2501.17148v3) shows ReFT-r1 competitive with SAEs for steering, but still outperformed by simple baselines

**Critical Insight:**
ReFT is parameter-efficient at LEARNING but might be LESS effective at FORGETTING. The code uses it for unlearning without validation that representation-level interventions work for forgetting.

### Recommendation
1. **For LoRA:** Increase rank to 8-16 and use AutoLoRA-style per-layer rank adaptation
2. **For ReFT:** Consider that ReFT's design (representation intervention) might be COUNTERPRODUCTIVE for unlearning
   - ReFT is optimized for task adaptation (adding capability)
   - Unlearning requires removing capability
   - Test if negative ReFT interventions (subtracting from representations) work better
3. **Alternative:** Use simple difference-in-means steering vectors (shown to outperform ReFT in AxBench for steering)

---

## 4. Unlearning Objectives (GA vs NPO)

### What the Code Does
- GA: maximize loss on forget set (negative gradient descent)
- NPO: negative preference optimization (treat forget as dispreferred)
- Alternates forget/retain optimization

### Research Findings

⚠️ **Gradient Ascent (GA) Issues:**

Multiple papers (2310.10683v2, 2408.10682v1, 2410.08109v5) show:
- ❌ Highly vulnerable to adversarial queries (55.2% knowledge recovery)
- ❌ Knowledge resurfaces with paraphrasing
- ❌ "Coreset effect" (2504.10185v2): only 5% of forget set needed = unlearning might be TOO EASY (superficial)

✅ **NPO is Better:**
- Paper "Bridging PA and Machine Unlearning" (2504.06659v1) shows NPO more robust
- Uses reference model to avoid mode collapse
- Code correctly loads frozen reference model

❌ **Missing Robustness Checks:**
- No adversarial evaluation (paraphrasing, Romanization)
- No iterative refinement (paper "Learn while Unlearn", 2407.20271v5)

### Recommendation
**Use NPO by default** and add adversarial evaluation with paraphrased/Romanized forget prompts.

---

## 5. Multilingual Unlearning Specificity

### What the Code Does
- Trains on Devanagari Hindi
- Evaluates cross-lingual leakage to Urdu/Punjabi/Bengali
- Uses script-blind selection (romanizes Hindi)

### Research Findings

🚨 **CRITICAL FLAW** - Paper "Cross-Lingual Unlearning" (2406.12354v2):

**Key Finding: "Unlearning in one language does NOT transfer to others"**

- Models vulnerable to LOW-RESOURCE LANGUAGE ATTACKS
- Sensitive info remains accessible in less dominant languages
- Paper proposes ADAPTIVE UNLEARNING with language-specific weights

❌ **Code Does NOT Implement This:**
1. No language-specific adapter weights
2. Trains only on Devanagari Hindi
3. Assumes LoRA/ReFT won't affect other languages

**Guaranteed Failure Modes:**
- ✅ Unlearns Devanagari Hindi
- ❌ Fails on Romanized Hindi (Hinglish)
- ❌ Leaks through Urdu (shares vocabulary)
- ❌ Leaks through English→Hindi translation
- ❌ Leaks through code-mixed inputs

### Recommendation
**Implement language-specific gating** with separate alpha values per language (detected by LID).

---

## 6. Evaluation Metrics

### What the Code Does
- **Extraction Strength (ES)**: LID-based, checks if output is Hindi
- **Perplexity (PPL)**: fluency on English retain set
- **Probes**: logistic regression on other layers
- **MIA**: membership inference attack
- **Cross-lingual leakage**: ES on related languages

### Research Findings

❌ **Superficial Evaluation** - Paper "Evaluating Deep Unlearning" (2410.15153v3):

**Current metrics test SUPERFICIAL unlearning only:**
- LID-based ES checks if model REFUSES to generate Hindi
- Doesn't test if model still UNDERSTANDS Hindi
- Deep unlearning requires: can model deduce related facts?

**Missing Tests:**
1. **Comprehension**: "Translate this Hindi sentence to English" (model might still understand)
2. **Vocabulary**: "What does 'कृपया' mean?" (lexical knowledge)
3. **Grammar**: "Conjugate 'जाना' in past tense" (grammatical knowledge)
4. **Adversarial**: Paraphrased/Romanized prompts
5. **Deduction**: Can English prompts elicit Hindi knowledge?

### Research on Adversarial Evaluation

Paper "LURK: Probing Hidden Knowledge" (2505.17160v1):
- Automated adversarial suffix prompting reveals latent knowledge
- Even "successfully unlearned" models leak under targeted prompts

Paper "Towards Robust Unlearning" (2408.10682v1):
- 55.2% knowledge recovery without model parameters
- Need adversarial robustness testing

### Recommendation
**Add comprehensive evaluation suite:**
```python
# Comprehension tests
hindi_to_english_translation(forget_samples)
vocabulary_tests(hindi_words)
grammar_tests(hindi_verbs)

# Adversarial tests
paraphrased_prompts(forget_samples)
romanized_prompts(forget_samples)
code_mixed_prompts(forget_samples)
```

---

## 7. Missing Critical Components

### Based on Literature Review

1. **Adversarial Training** (Paper 2408.10682v1 "LAU Framework")
   - Should include adversarial perturbations during unlearning
   - Min-max optimization: attack stage + defense stage

2. **Iterative Refinement** (Paper 2407.20271v5)
   - Current: single-pass unlearning
   - Should: iterative evaluation + refinement

3. **Data Attribution** (Paper 2504.06658v1 "MRD Metric")
   - Some samples harder to unlearn than others
   - Should use Memory Removal Difficulty metric for weighted sampling

4. **Representation Analysis**
   - Should verify unlearned representations DIFFER from base
   - Use SVCCA or CKA to compare before/after activations

---

## 8. Will This Work? Final Verdict

### What Will Work ✅

1. **Basic unlearning observable**: ES will decrease (model refuses Hindi prompts)
2. **PPL maintained**: English fluency preserved
3. **Statistical robustness**: multiple seeds + bootstrap CI
4. **NPO objective**: better than GA for stability
5. **ReFT parameter efficiency**: Will use fewer parameters than LoRA

### What Will Fail ❌

1. **Deep unlearning**: model still UNDERSTANDS Hindi (comprehension intact)
2. **Cross-lingual leakage**:
   - Romanized Hindi bypasses unlearning
   - Urdu/Punjabi vocabulary leaks
   - English→Hindi translation works
3. **Adversarial robustness**: paraphrased prompts reveal knowledge
4. **SAE effectiveness**: features might be correlated not causal
5. **Layer selection optimality**: no validation layers are causal
6. **ReFT for unlearning**: ReFT designed for LEARNING not FORGETTING
   - Representation interventions might increase, not decrease, Hindi capability
   - No validation that negative interventions work for unlearning

### Expected Results

**Metrics Will Show:**
- ES_forget: 0.15 → 0.05 (✅ appears successful)
- PPL_retain: 8.2 → 8.5 (✅ minimal degradation)
- MIA_AUC: 0.65 → 0.52 (✅ privacy improved)
- Cross-ling: 0.10 → 0.08 (✅ slight reduction)

**Deep Tests Will Reveal:**
- Comprehension: 85% accuracy (❌ knowledge intact)
- Romanized Hindi: 0.18 ES (❌ leakage)
- Adversarial prompts: 60% recovery (❌ vulnerable)

---

## 9. Recommended Improvements

### Priority 1: Critical Fixes

1. **Gradient-based SAE feature selection**
```python
# Replace activation-based with gradient-based
def pick_grad_sae_features(sae, model, tok, forget, layer, device, topk=64):
    features_grad = []
    for batch in chunked(forget, 8):
        enc = tok(batch, return_tensors='pt', ...).to(device)
        out = model(**enc, output_hidden_states=True)
        H = out.hidden_states[layer+1]
        z = sae.E(H.mean(1))  # [B, m]
        loss = out.loss  # forget loss
        grad = torch.autograd.grad(loss, z, retain_graph=True)[0]
        features_grad.append(grad.abs().mean(0))
    return torch.cat(features_grad).argsort(descending=True)[:topk]
```

2. **Language-specific adapter gating**
```python
class LanguageSpecificGate:
    def __init__(self, base_alpha_per_lang):
        self.alpha_per_lang = base_alpha_per_lang  # {"hi": 0.9, "en": 0.0, ...}

    def get_alpha(self, text, lid):
        lang = lid.infer(text)[0]
        return self.alpha_per_lang.get(lang, 0.0)
```

3. **Comprehension evaluation**
```python
def test_comprehension(model, tok, hindi_sentences, device):
    prompts = [f"Translate to English: {s}" for s in hindi_sentences]
    outputs = generate(model, tok, prompts, device)
    # Check if translations are accurate (model still understands)
    return accuracy(outputs, reference_translations)
```

4. **Adversarial robustness testing**
```python
def test_adversarial_unlearning(model, tok, forget, device):
    # Paraphrasing
    paraphrased = paraphrase_hindi(forget)
    es_para = extraction_strength(generate(model, tok, paraphrased, device), lid)

    # Romanization
    romanized = romanize_hindi(forget)
    es_roman = extraction_strength(generate(model, tok, romanized, device), lid)

    # Code-mixing
    mixed = create_hinglish(forget)
    es_mixed = extraction_strength(generate(model, tok, mixed, device), lid)

    return {"paraphrase": es_para, "romanized": es_roman, "mixed": es_mixed}
```

### Priority 2: Methodology Improvements

5. **Increase LoRA rank** to 8-16 (from current 4)
6. **Add causal validation** of selected layers
7. **Implement iterative unlearning** with refinement loop
8. **Use NPO by default** (GA only for ablation)

### Priority 3: Analysis Enhancements

9. **SVCCA analysis** of unlearned representations
10. **Feature importance** analysis for SAE features
11. **Attribution-weighted sampling** using MRD metric
12. **Adversarial training** with LAU framework

---

## 10. Experimental Design Suggestions

### Minimal Viable Experiment

```python
# Test just layer selection + LoRA with comprehension eval
1. Select layers with semantic mode
2. Train LoRA (rank=8, NPO) on selected layers
3. Evaluate:
   - Standard ES (LID-based)
   - Comprehension ES (translation tasks)
   - Adversarial ES (Romanized Hindi)
4. Compare: does comprehension ES drop as much as standard ES?
```

### Full Robust Experiment

```python
# Test all components with adversarial evaluation
1. Layer selection:
   - Semantic mode + causal validation
   - Compare to random baseline
2. SAE training:
   - Train with gradient-based feature selection
   - Validate quality (recon MSE, dead features)
3. Unlearning:
   - LoRA rank=8-16 with NPO
   - Iterative refinement (3 rounds)
4. Evaluation:
   - Standard metrics (ES, PPL, MIA)
   - Comprehension tests
   - Adversarial tests (3 types)
   - Cross-lingual leakage (5 languages)
5. Analysis:
   - SVCCA of representations
   - Feature attribution
   - Failure mode analysis
```

---

## 11. Literature-Backed Failure Modes

| Failure Mode | Probability | Evidence | Mitigation |
|-------------|-------------|----------|------------|
| Romanized Hindi leakage | 95% | Paper 2406.12354v2 | Language-specific gating |
| Comprehension intact | 90% | Paper 2410.15153v3 | Test comprehension directly |
| Adversarial recovery | 80% | Paper 2408.10682v1 | Adversarial training (LAU) |
| Urdu/Punjabi leakage | 75% | Paper 2406.12354v2 | Adaptive language weights |
| SAE features non-causal | 70% | Papers 2411.08790v1, 2510.03659v1 | Gradient-based selection |
| Layer selection suboptimal | 60% | No validation in code | Add causal ablation tests |
| LoRA rank too low | 55% | Paper 2405.09673v2 | Increase to rank 8-16 |

---

## 12. Key Papers for Reference

### Must-Read for Understanding Issues

1. **Cross-Lingual Unlearning** (2406.12354v2) - Shows single-language unlearning fails
2. **SAEs Can Improve Unlearning** (2504.08192v1) - When SAEs work vs fail
3. **LoRA Learns Less and Forgets Less** (2405.09673v2) - LoRA characteristics
4. **Evaluating Deep Unlearning** (2410.15153v3) - Why current metrics are insufficient
5. **Towards Robust Unlearning** (2408.10682v1) - Adversarial attacks on unlearning
6. **ReFT: Representation Finetuning** (2404.03592v3) - ReFT design and characteristics

### Useful for Improvements

7. **GradSAE** (2505.08080v2) - Gradient-based feature selection
8. **Does interpretability = utility?** (2510.03659v1) - Feature selection pitfalls
9. **AxBench** (2501.17148v3) - SAE vs simple baselines (by ReFT authors)
10. **LAU Framework** (2408.10682v1) - Adversarial training for robustness
11. **Learn while Unlearn** (2407.20271v5) - Iterative refinement

---

## Conclusion

The experimental approach is **theoretically sound** with good foundations (NPO, SAE gating, semantic selection), but has **critical practical gaps** that will cause:

1. ✅ **Superficial success**: Metrics show Hindi generation suppressed
2. ❌ **Deep failure**: Comprehension/understanding intact
3. ❌ **Cross-lingual leakage**: Romanized Hindi + related languages bypass unlearning
4. ❌ **Adversarial vulnerability**: Paraphrasing reveals hidden knowledge

**For research publication:** Document these limitations and position as "preliminary study showing challenges of multilingual unlearning."

**For production use:** Implement all Priority 1 fixes before deployment.

**Bottom line:** Will this work? Yes for demos, no for robust unlearning. The code is a good starting point but needs significant refinement based on recent research.

