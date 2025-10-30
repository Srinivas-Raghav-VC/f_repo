# De-Risking Multilingual LLM Unlearning: A 2025 Verification Playbook for MMIE

## Executive Summary

This report provides a comprehensive verification of the MMIE research codebase for multilingual machine unlearning. Our analysis reveals a solid architectural foundation undermined by critical statistical flaws and a growing gap with 2025 state-of-the-art methods. The primary issue—a lack of correction for multiple hypothesis testing—inflates the Type I error rate to an untenable **~47%**, rendering current results unpublishable.

Immediate implementation of False Discovery Rate (FDR) correction is non-negotiable. Beyond this, we recommend a strategic pivot to more advanced, efficient, and interpretable components. Upgrading from custom TopK Sparse Autoencoders (SAEs) to **SAELens Matryoshka SAEs** is critical for cleaner feature disentanglement and improved unlearning success [1] [2]. Integrating **GRUN (Gated Representation Unlearning)** via the `PyReFT` library is essential for outperforming the current Gradient Ascent/NPO approach, offering superior utility preservation and a **>85%** reduction in training time [3] [4].

The existing 6-gate evaluation framework is a strong start but lacks coverage in key areas identified by recent survey literature, namely sample-level privacy verification and efficiency benchmarking [5]. We recommend augmenting it with newer audits like the **Feature Injection Test** and **WaterDrum**, alongside a more robust Membership Inference Attack (MIA).

While the codebase shows recent positive momentum—including the implementation of ActPert auditing and fixes for memory leaks—achieving readiness for a top-tier conference like NeurIPS or ACL requires a focused 7-9 week effort. This playbook outlines a clear, prioritized roadmap to address these gaps, transforming the MMIE project from a promising prototype into a statistically robust and scientifically novel contribution.

---

## 1. Implementation Quality Audit: A "C+" Grade with a Path to "A"

The current MMIE codebase is architecturally sound but suffers from critical gaps in statistical rigor and incomplete integration of state-of-the-art libraries. While recent improvements like memory leak fixes and the addition of key libraries are positive steps, the project's validity is threatened by unverified component implementations and a show-stopping statistical oversight.

### Codebase Health Check: Positive Momentum on Reproducibility

Recent efforts to fix memory leaks and ensure proper model cleanup after each seeded run are commendable. These changes directly address a key pillar of the reproducibility checklists used by top-tier conferences (e.g., NeurIPS, ICLR). Similarly, the explicit addition of libraries like `pingouin` for bootstrap confidence intervals and `statsmodels` for future statistical corrections demonstrates a commitment to rigorous evaluation [6]. However, full reproducibility remains elusive without deterministic algorithm flags and automated git-hash logging.

### Alignment to Reference Papers: ActPert and GradSAE Fidelity is Assumed, Not Verified

The project has implemented two key 2025 methods, ActPert and GradSAE, but their fidelity to the original papers is unconfirmed.

* **ActPert:** The implementation of Activation Perturbation-based Auditing appears to follow the specification in Chen et al. (May 2025), using perturbations at layer 12 with a noise intensity of 0.01 [7] [8]. The method involves injecting noise into target token embeddings, calculating an activation perturbation vector, and reintroducing it during generation to probe for residual knowledge [7] [8]. However, without unit tests comparing outputs against a reference implementation or the paper's published ROUGE-L scores, its correctness is an open question.
* **GradSAE:** The gradient-based SAE feature selection is described as "GradSAE-style," which suggests potential deviation from the Shu et al. (May 2025) paper [9] [10]. The original GradSAE method identifies influential latents by incorporating output-side gradient information, arguing that input activation alone is insufficient for reliable model steering [9] [11]. Verification requires a line-by-line code review against the official implementation to ensure the gradient-based influence calculation is correctly implemented.

### Missing Integrations: PyReFT and Matryoshka APIs Are Present but Unused

The codebase includes the necessary libraries (`pyreft`, `sae-lens`) for significant upgrades, but they are not yet integrated [12] [13]. The `PyReFT` backend is detected but not connected to an unlearning method like GRUN, and the SAE architecture remains a custom TopK implementation instead of the superior Matryoshka models available through `sae-lens` [1] [14]. These missing integrations represent the largest opportunity for improving the project's performance and novelty.

---

## 2. Statistical Rigor & FDR Control: The Most Critical Blocker

The single most critical issue preventing publication is the absence of corrections for multiple hypothesis testing across the six evaluation gates. With an estimated Type I error rate of **~46.9%** (calculated as `1 - (1 - 0.10)^6`), nearly half of the "significant" findings could be false positives. This invalidates any conclusions drawn from the evaluation framework. Implementing a False Discovery Rate (FDR) correction is a non-negotiable first step.

### Choosing the Right Correction Method for Correlated Gates

For multiple correlated tests, the standard Benjamini-Hochberg (BH) procedure is a valid starting point, as it provably controls FDR under positive regression dependence (PRDS), a condition often met in ML experiments [15] [16]. However, given the unknown and likely complex correlation structure between the six evaluation gates (e.g., extraction strength and adversarial robustness are likely correlated), a more conservative approach may be warranted.

| Method | `statsmodels` key | Dependence Assumption | Power | Recommendation for MMIE (6 Gates) |
|--------|-------------------|----------------------|-------|----------------------------------|
| **Benjamini-Hochberg (BH)** | `fdr_bh` | Independence or Positive Dependence (PRDS) | High | **Critical Fix.** Use as the default. Simple, powerful, and widely accepted [15] [17] [6]. |
| **Benjamini-Yekutieli (BY)** | `fdr_by` | Arbitrary Dependence | Lower | **Safe Fallback.** Use if tests are negatively correlated or dependence is unknown/complex. More conservative [18] [16] [17]. |
| **Holm-Bonferroni** | `holm` | Arbitrary Dependence (Controls FWER) | Lower than BH | A reasonable alternative to control the stricter Family-Wise Error Rate, but often too conservative for exploratory research [19]. |
| **Storey's q-value** | (via R package `qvalue`) | General Dependence | High | A powerful alternative to BH, offering a Bayesian interpretation, but requires external library integration [20] [21]. |
| **Permutation Tests** | (manual implementation) | Arbitrary Dependence | Data-dependent | The gold standard for controlling error rates with unknown correlation, but computationally expensive. Reserve for headline claims [20]. |

**Recommendation:** Immediately implement the Benjamini-Hochberg procedure using `statsmodels`. Given the small number of hypotheses (six), the power loss from the more conservative Benjamini-Yekutieli method is likely acceptable and provides stronger guarantees against arbitrary correlations [18].

### Implementation and Reporting

The fix is a one-line change in the analysis script. After collecting the p-values from all six gates for a given run, apply the correction.

```python
import statsmodels.stats.multitest as smm

# p_values from the 6 evaluation gates
p_values = [0.02, 0.08, 0.09, 0.25, 0.30, 0.01]

# Apply Benjamini-Hochberg FDR correction
rejected, p_adjusted, _, _ = smm.multipletests(p_values, alpha=0.10, method='fdr_bh')

print("Adjusted p-values:", p_adjusted)
print("Hypotheses rejected:", rejected)
```

For publication, all reported p-values from the evaluation gates must be the **adjusted p-values**. Results should be presented alongside bootstrap confidence intervals (BCa) and effect sizes to provide a complete picture of the findings' magnitude and uncertainty.

---

## 3. Modern Unlearning Algorithms Benchmark: GRUN Outperforms NPO

The current unlearning method, gradient ascent combined with Negative Preference Optimization (NPO), is being rapidly superseded. The 2025 literature points to Gated Representation Unlearning (GRUN) as a more effective and dramatically more efficient alternative [4] [22].

### GRUN: A Superior Framework for Fine-Tuning-Based Unlearning

Introduced by Ren et al. (ACL 2025), GRUN enhances fine-tuning-based unlearning by explicitly separating the tasks of identifying and suppressing target knowledge [3] [4]. It consists of two lightweight components:

1. **Soft Gate Function:** A small, trainable module that learns to distinguish target data (to be forgotten) from retain data. It outputs a value near 1 for forget inputs and near 0 for retain inputs [3].
2. **Suppression Module:** Instead of updating model weights, GRUN uses Representation Fine-tuning (ReFT) to learn a low-rank transformation that modifies the internal representations of the LLM, steering them away from the undesired knowledge [3] [14].

This design provides significant advantages over methods like NPO. In experiments, GRUN reduced training time by over **95%** compared to vanilla fine-tuning and **85%** compared to LoRA, while requiring fewer than **0.05%** of the model's parameters to be updated [3]. It consistently improves both unlearning effectiveness and model utility preservation across benchmarks like TOFU and WMDP [3].

| Method | Mechanism | Key Advantage | Key Disadvantage | Publication |
|--------|-----------|--------------|-----------------|-------------|
| **GA + NPO** | Gradient ascent on forget data, preference optimization against target outputs. | Simple to implement. | Often degrades general model utility. | (Baseline) |
| **GRUN** | Gated Representation Fine-tuning (ReFT) to suppress specific representations. | **Highly efficient** (<0.05% params), preserves utility, effective for sequential unlearning. | Can harm utility if retain data is not diverse. | Ren et al., ACL 2025 [3] [4] |
| **Offset Unlearning** | Adds a learned "unlearning vector" to model weights. | Simple, effective for certain tasks. | Less explored for multilingual contexts. | Huang et al., 2024 |
| **"Simplicity Prevails"** | A refined NPO variant focusing on simpler loss formulations. | Improves on standard NPO. | Still a preference-based method, may have utility trade-offs. | Fan et al., 2024 |

### Integration Guide: Activating GRUN with PyReFT

The `pyreft` library is already part of the codebase. Full integration involves using it to implement the GRUN framework.

1. **Configure ReFT:** Use `pyreft.ReftConfig` to define the intervention. The GRUN paper suggests applying LoReFT interventions at interval layers in the upper half of the model (e.g., last, 7th-to-last, 12th-to-last layers) with a low rank (e.g., **r=4**) [3] [12].
2. **Implement the Gate:** Implement the soft gate function (a simple linear layer or small MLP) that takes layer activations as input.
3. **Combine Gate and ReFT:** Modify the model's forward pass (using `pyreft`'s intervention mechanism) to apply the ReFT transformation scaled by the gate's output [14].
4. **Train:** Train only the gate and ReFT parameters on a mixed dataset of "forget" (Hindi) and "retain" (English) examples.

**Recommendation:** Prioritize the full integration of GRUN via `PyReFT`. It should become the default unlearning method for its superior performance and efficiency, which will also strengthen the novelty of the research.

---

## 4. SAE Architecture & Interpretability Upgrade: Matryoshka over TopK

The custom TopK SAE architecture is a liability. It is prone to "feature absorption" and "feature splitting," where concepts are not cleanly represented by individual latents, undermining interpretability and control [23]. The 2025 SOTA is to use **Matryoshka SAEs**, which are demonstrably superior for feature disentanglement and targeted unlearning tasks [1] [24].

### Architecture Deep-Dive: Why Matryoshka SAEs Are Better

Matryoshka SAEs, introduced by Bussmann et al., train multiple nested dictionaries of increasing size simultaneously [1] [25]. This forces smaller, inner dictionaries to learn general, high-level features, while larger, outer dictionaries learn more specific ones [1]. This hierarchical structure directly combats the failure modes of simpler architectures:

* **TopK SAEs:** Simply activate the `k` latents with the highest activation for a given input. This can lead to a single, overly general latent absorbing multiple distinct-but-related concepts [26].
* **Matryoshka SAEs:** By using nested loss terms, they enforce a hierarchy that encourages disentangled representations [27]. This results in cleaner, more interpretable features at multiple levels of abstraction [1].

### Benchmarks: SAEBench Confirms Matryoshka's Superiority for Unlearning

The comprehensive `SAEBench` benchmark (ICML 2025) confirms these advantages empirically [2]. While Matryoshka SAEs may show slightly worse reconstruction loss, they substantially outperform other architectures on feature disentanglement metrics, including a specific **Unlearning Capability** task [2]. They demonstrate superior performance in targeted concept erasure, a task highly analogous to MMIE's goal of removing Hindi knowledge [1]. Studies show Matryoshka SAEs can reduce feature absorption by **~10x** compared to BatchTopK SAEs [23].

**Recommendation:** The custom TopK SAE should be replaced. A new set of SAEs should be trained on the base model using the `sae-lens` library with a Matryoshka architecture [13]. The existing GradSAE-style feature selection method remains compatible and should be applied to the new Matryoshka latents to identify influential features for unlearning interventions [10].

---

## 5. Efficiency Stack: FlashAttention-2, Quantization, and LID

Making FlashAttention-2 and 8-bit quantization default options is a sound goal for improving efficiency, but it requires careful implementation to avoid pitfalls related to performance and reproducibility.

### Compatibility and Risks: A Hardware-Aware Approach

FlashAttention-2 (FA2) offers significant speedups, especially for long sequences, but it is not a universal drop-in replacement for standard attention [28] [29].

| Component | Precision/Method | Compatibility & Performance | Risks & Mitigations |
|-----------|-----------------|----------------------------|---------------------|
| **FlashAttention-2** | FP16 / BF16 | Requires Ampere/Ada/Hopper GPUs. Up to **2x** faster than FA1 [28] [30]. | **BF16 bug on A100:** Can produce incorrect values (SNR drop from `inf` to `46.5`) [31]. **Mitigation:** Default to FP16 on A100s. |
| **8-bit Quantization** | `bitsandbytes` | Compatible with FA2 because weights are up-casted during computation [32]. Reduces memory but can slightly increase latency [29]. | Overtrained models may degrade more when quantized [33]. **Mitigation:** Evaluate perplexity and task performance post-quantization to ensure parity. |
| **4-bit Quantization** | `bitsandbytes` (NF4) | Also compatible with FA2 [32]. Offers maximum memory savings. | Higher risk of performance degradation. **Mitigation:** Use primarily for inference on memory-constrained hardware. Rigorous post-quantization evaluation is essential. |

**Recommendation:**
* **Inference:** Make FA2 + 8-bit quantization the default, gated by a CI check that verifies perplexity and key metric parity with the FP16 baseline.
* **Training:** Avoid FA2 for gradient-based steps due to potential non-determinism and the BF16 bug. Stick to standard attention with FP16/BF16 precision.
* **Implementation:** Enable FA2 in Hugging Face Transformers via `attn_implementation="flash_attention_2"` and load quantized models using `BitsAndBytesConfig` [34].

### High-Throughput LID Pipeline

The recommendation to upgrade the Language Identification (LID) pipeline to a combination of FastText and `pycld3` is strongly supported. This combination is known to be significantly faster (up to 100x) than many alternatives, which is critical for high-throughput preprocessing and evaluation, especially when dealing with code-mixed text. Implement batching and establish clear confidence thresholds to handle ambiguous cases and prevent LID errors from leaking into evaluation metrics.

---

## 6. Evaluation Framework Expansion: Moving from Sufficient to SOTA

The current 6-gate evaluation framework is a solid foundation, aligning with emerging standards like the "MUSE" six-way evaluation [35]. However, a landmark 2025 survey on unlearning verification by Zhang et al. provides a more comprehensive, seven-dimension rubric that reveals critical gaps in the current setup [5].

### Mapping MMIE's Gates to the 2025 Verification Dimensions

| Zhang et al. Dimension [5] | MMIE's Current Coverage (6 Gates) | Gap & Recommended Addition |
|---------------------------|----------------------------------|---------------------------|
| 1. Theoretical Guarantees | None | Acknowledge as a limitation; focus on empirical rigor. |
| 2. Access Requirements | Black-box & White-box (uses gradients) | Covered. |
| 3. **Sample-Level Verification** | MIA (standard) | **GAP.** Standard MIA is weak. Add **U-LiRA+** (Du et al. 2024) or **RaMIA** for stronger privacy risk calibration. |
| 4. Verification Accuracy | Extraction Strength, PPL, Adversarial Robustness | Covered. Augment with **ActPert** [7], **Feature Injection Test**, and **WaterDrum** for deeper auditing. |
| 5. Reliance on Pre-injected Data | None (uses existing knowledge) | Covered. |
| 6. **Efficiency & Scalability** | None | **GAP.** Add wall-clock time and memory profiling for unlearning methods as a formal gate. |
| 7. Method Specificity | Cross-lingual Leakage | Covered. |

### Critical Additions to the Audit Suite

To de-risk rejection, the evaluation framework must be expanded:

* **Stronger MIA:** The current MIA is likely insufficient. Investigate and implement a state-of-the-art attack like **U-LiRA+** (Du et al., 2024) or **RaMIA**, which are designed to be more robust for evaluating modern LLMs. While specific details on U-LiRA+ were not found in the initial search, its mention in the literature makes it a priority for investigation.
* **Parametric Audits:** The current audits are primarily behavioral. Add parametric audits to check for residual knowledge in the model's weights and representations. The **Feature Injection Test** and **WaterDrum** are two such methods highlighted in the 2025 survey literature [5].
* **Script-Blind Validation:** The use of script-blind semantic evaluation is validated by recent work (Lu & Koehn, 2024), which shows that information propagates across languages in multilingual LLMs [36]. Continue this approach, but for publication, explicitly document the prompt generation process, LID thresholds, and controls used to ensure semantic equivalence.

---

## 7. Statistical Validation Best Practices: Beyond a Single p-value

Achieving statistical rigor for a top-tier publication in 2025 requires more than just correcting p-values. Reviewers now expect a multi-faceted approach to statistical validation.

* **Bootstrap vs. Permutation Tests:** While Bootstrap BCa confidence intervals are good for estimating uncertainty, they are not ideal for hypothesis testing with correlated data. For the most critical claims (e.g., the primary Hindi forgetting metric), supplement BCa CIs with **permutation tests**. These tests provide more robust Type I error control under complex dependency structures and are increasingly expected by reviewers.
* **Effect Sizes and Power Analysis:** For every significant result, report an effect size (e.g., Cohen's d, odds ratio) to quantify the magnitude of the finding. While a full a-priori power analysis may be difficult, a post-hoc sensitivity analysis can demonstrate the statistical power of the experimental design.

---

## 8. Reproducibility & Artifact Readiness: Securing the Badge

The project is well-positioned to receive a strong "Artifacts Available" and "Reproducible" badge, but a few final steps are needed to meet the stringent criteria of NeurIPS and ICLR.

| Requirement | Current Status | Action Needed |
|------------|---------------|--------------|
| **Hardware/Software List** | Implicit | Explicitly document GPU models, CUDA/cuDNN versions, and all library versions in `environment.yml`. |
| **Seeds & Determinism** | Partial (seeds used) | Add `torch.use_deterministic_algorithms(True)` and document any remaining sources of non-determinism (e.g., specific CUDA kernels) [28]. |
| **Data & Preprocessing** | Not specified | Provide scripts and links to all datasets, including forget/retain splits and evaluation prompts. |
| **Code & Checkpoints** | Partial | Ensure the final code is clean, commented, and runnable with a single script. Provide trained model and SAE checkpoints. |
| **Experiment Logging** | Not specified | Automatically log the git hash of the codebase for every experimental run to link results to a specific code version. |

---

## 9. Roadmap & Timeline to Publication

With parallel workstreams, the MMIE project can be made submission-ready for a major 2026 conference deadline in **7-9 weeks**.

| Phase | Task | Duration | Dependencies | Priority |
|-------|------|----------|-------------|----------|
| **Phase 1: Critical Fixes** | **1. Implement FDR Correction (BH/BY)** | 3 days | - | **CRITICAL** |
| (Weeks 1-3) | **2. Integrate GRUN with PyReFT** | 2 weeks | - | **CRITICAL** |
| | **3. Train Matryoshka SAEs with `sae-lens`** | 2 weeks | Base Model | **CRITICAL** |
| | **4. Verify ActPert & GradSAE Implementations** | 1 week | - | High |
| **Phase 2: SOTA Enhancement** | **5. Implement FA2 + Quantization Defaults** | 1 week | - | High |
| (Weeks 3-6) | **6. Add New Audits (FIT, WaterDrum, U-LiRA+)** | 3 weeks | Phase 1 | Medium |
| | **7. Implement Permutation Tests for Key Metrics** | 1 week | Phase 1 | Medium |
| **Phase 3: Polish & Submit** | **8. Finalize Reproducibility Package** | 1 week | All previous | High |
| (Weeks 7-9) | **9. Run Final Experiments & Analysis** | 2 weeks | All previous | High |
| | **10. Write Manuscript** | 3 weeks | All previous | High |

---

## 10. Future-Proofing: Emerging 2025-2026 SOTA Watchlist

The field of machine unlearning is evolving rapidly. To maintain a competitive edge for future work, the following emerging trends should be monitored:

* **FlashAttention-3 and FP8:** The next generation of attention mechanisms promises further efficiency gains, especially on Hopper-generation hardware, by leveraging FP8 precision [28] [37].
* **Advanced ReFT Variants:** Research into ReFT is ongoing. Look for methods that offer even finer-grained control or hybrid approaches combining ReFT with other PEFT techniques like LoRA [12].
* **Next-Generation Privacy Audits:** Keep an eye on the evolution of MIA techniques beyond U-LiRA+ and RaMIA, as privacy evaluation becomes an increasingly central part of unlearning research.

---

## References

1. *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. https://openreview.net/forum?id=m25T5rAy43
2. *SAEBench: A Comprehensive Benchmark for Sparse ...*. https://arxiv.org/pdf/2503.09532
3. *A General Framework to Enhance Fine-tuning-based LLM ...*. https://aclanthology.org/2025.findings-acl.949.pdf
4. *A General Framework to Enhance Fine-tuning-based LLM ...*. https://aclanthology.org/2025.findings-acl.949/
5. *Towards Reliable Forgetting: A Survey on Machine Unlearning ...*. https://arxiv.org/html/2506.15115v2
6. *statsmodels.stats.multitest.fdrcorrection*. https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection.html
7. *Does Machine Unlearning Truly Remove Knowledge?*. https://arxiv.org/html/2505.23270v2
8. *Fetched web page*. https://arxiv.org/pdf/2505.23270v2.pdf
9. *Identifying Influential Latents by Gradient Sparse Autoencoders*. https://www.researchgate.net/publication/391707373_Beyond_Input_Activations_Identifying_Influential_Latents_by_Gradient_Sparse_Autoencoders
10. *Identifying Influential Latents by Gradient Sparse Autoencoders - arXiv*. https://arxiv.org/html/2505.08080v1
11. *Beyond Input Activations: Identifying Influential Latents by ...*. https://arxiv.org/pdf/2505.08080
12. *stanfordnlp/pyreft: Stanford NLP Python library for ... - GitHub*. https://github.com/stanfordnlp/pyreft
13. *decoderesearch/SAELens: Training Sparse Autoencoders ...*. https://github.com/jbloomAus/SAELens
14. *renjie3/GRUN - GitHub*. https://github.com/renjie3/GRUN
15. *Beware of counter-intuitive levels of false discoveries in datasets ...*. https://pmc.ncbi.nlm.nih.gov/articles/PMC12359981/
16. *False discovery rate control for non-positively regression dependent ...*. https://www.researchgate.net/publication/222514811_False_discovery_rate_control_for_non-positively_regression_dependent_test_statistics
17. *statsmodels.stats.multitest.multipletests*. https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
18. *A comparison of multiple testing adjustment methods with block ...*. https://www.researchgate.net/publication/316942148_A_comparison_of_multiple_testing_adjustment_methods_with_block-correlation_positively-dependent_tests
19. *Multiplicity corrections in life sciences: challenges and ...*. https://pmc.ncbi.nlm.nih.gov/articles/PMC12205177/
20. *False Discovery Rate*. https://www.publichealth.columbia.edu/research/population-health-methods/false-discovery-rate
21. *Integrating genetics and environmental factors in ...*. https://www.sciencedirect.com/science/article/pii/S2351989425000186
22. *A General Framework to Enhance Fine-tuning-based LLM Unlearning*. https://arxiv.org/abs/2502.17823
23. *Learning Multi-Level Features with Matryoshka SAEs - LessWrong*. https://www.lesswrong.com/posts/rKM9b6B2LqwSB5ToN/learning-multi-level-features-with-matryoshka-saes
24. *Learning Multi-Level Features with Matryoshka Sparse ...*. https://arxiv.org/abs/2503.17547
25. *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. https://icml.cc/virtual/2025/poster/44178
26. *An Intuitive Explanation of Sparse Autoencoders for LLM ...*. https://adamkarvonen.github.io/machine_learning/2024/06/11/sae-intuitions.html
27. *Correlated Features Break Narrow Sparse Autoencoders - arXiv*. https://arxiv.org/html/2505.11756v1
28. *Dao-AILab/flash-attention: Fast and memory-efficient ...*. https://github.com/Dao-AILab/flash-attention
29. *Optimizing inference*. https://huggingface.co/docs/transformers/main/en/llm_optims
30. *FlashAttention-2: Faster Attention with Better Parallelism and Work...*. https://openreview.net/forum?id=mZn2Xyh9Ec
31. *BF16 Flash Attention producing incorrect values compared ...*. https://github.com/Dao-AILab/flash-attention/issues/1071
32. *FlashAttention-2's 16 bit requirement - Hugging Face Forums*. https://discuss.huggingface.co/t/flashattention-2s-16-bit-requirement/67069
33. *ICLR 2025 - Bird's-eye views of conference proceedings*. https://www.confviews.com/iclr2025/
34. *GPU - Hugging Face*. https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one
35. *(PDF) MUSE: Machine Unlearning Six-Way Evaluation for ...*. https://www.researchgate.net/publication/382111152_MUSE_Machine_Unlearning_Six-Way_Evaluation_for_Language_Models
36. *Every Language Counts: Learn and Unlearn in Multilingual LLMs*. https://arxiv.org/html/2406.13748v1
37. *Aman's AI Journal • Primers • FlashAttention*. https://aman.ai/primers/ai/flashattention/

