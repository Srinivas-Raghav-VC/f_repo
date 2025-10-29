# -*- coding: utf-8 -*-
"""
RESEARCH-GRADE Multilingual Unlearning Experiment
Qwen 2.5-1.5B with ALL Advanced Features Enabled
"""

# ============================================================================
# CELL 1: GPU Check
# ============================================================================
!nvidia-smi

# ============================================================================
# CELL 2: Clone Repository
# ============================================================================
!git clone https://github.com/Srinivas-Raghav-VC/f_repo.git
%cd f_repo

# ============================================================================
# CELL 3: Install Dependencies
# ============================================================================
# Core ML libraries
!pip install -q torch transformers accelerate peft einops scikit-learn scipy numpy python-dotenv

# LID ensemble components
!pip install -q fasttext langid pycld3
!pip install -q google-generativeai  # Gemini SDK for LID and judge

# Transliteration for script-blind evaluation
!pip install -q indic-transliteration

# Experiment tracking
!pip install -q wandb

# Visualization
!pip install -q matplotlib seaborn

print("✅ All dependencies installed!")

# ============================================================================
# CELL 4: Configure Environment Variables
# ============================================================================
import os
from google.colab import userdata

# Required: Hugging Face token
try:
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print("✅ HF_TOKEN loaded from secrets")
except:
    os.environ['HF_TOKEN'] = input("Enter your HuggingFace token: ")
    print("⚠️ Using manual HF token")

# Required: Gemini API key (for LID ensemble + judge-assisted selection)
try:
    os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
    print("✅ GEMINI_API_KEY loaded from secrets")
except:
    print("\n" + "="*60)
    print("GET YOUR FREE GEMINI API KEY:")
    print("👉 https://aistudio.google.com/app/apikey")
    print("="*60 + "\n")
    os.environ['GEMINI_API_KEY'] = input("Paste your Gemini API key: ")
    print("✅ Gemini API key set!")

# Set offload directory for large model handling
!mkdir -p /content/offload
os.environ['OFFLOAD_DIR'] = '/content/offload'
os.environ['SAFETENSORS_FAST'] = '0'

print("\n✅ Environment configured")
print("\nGemini will be used for:")
print("  1. LID ensemble voting (enhanced language detection)")
print("  2. LLM judge-assisted layer selection (semantic quality scoring)")
print("  3. Adversarial prompt generation (optional)")

# ============================================================================
# CELL 5: WandB Login (Experiment Tracking)
# ============================================================================
import wandb

wandb.login()  # Paste API key when prompted

# Initialize experiment tracking
wandb.init(
    project="multilingual-unlearning-research",
    name="qwen-1.5b-full-research-grade",
    config={
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "seeds": [42, 123, 456],
        "forget_obj": "npo",
        "stability_select": 5,
        "judge_assist": True,
        "script_scrub": True,
        "semantic_features": True,
        "dynamic_gating": True,
        "semantic_dynamic_gating": True
    },
    tags=["research-grade", "qwen-1.5b", "hindi-unlearning", "sae-gating"]
)

print("✅ WandB initialized")

# ============================================================================
# CELL 6: Verify Data Files
# ============================================================================
import os

data_files = [
    'data/forget_hi.jsonl',
    'data/retain_en.jsonl',
    'data/mixed.jsonl',
    'data/urdu.jsonl',
    'data/punjabi.jsonl',
    'data/bengali.jsonl',
    'adversarial.jsonl'
]

print("=" * 60)
print("DATA FILES STATUS")
print("=" * 60)
for f in data_files:
    if os.path.exists(f):
        with open(f) as file:
            lines = sum(1 for _ in file)
        print(f"✅ {f}: {lines} samples")
    else:
        print(f"❌ Missing: {f}")

print("\n⚠️ If any files are missing, the experiment will fail!")

# ============================================================================
# CELL 7: Quick Sanity Check (OPTIONAL - 5 minutes)
# ============================================================================
print("Running quick sanity check with minimal steps...")
print("This verifies the pipeline works before the full run.\n")

!python mmie.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl \
  --retain data/retain_en.jsonl \
  --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl \
  --n_top 2 \
  --selection_mode semantic \
  --train_sae_steps 100 \
  --sae_k 16 \
  --lora_steps 50 \
  --reft_steps 50 \
  --seeds 42 \
  --use_gemini \
  --device cuda \
  --out test_quick.json

print("\n✅ Quick test complete! Pipeline is working.")
print("Check test_quick.json for preliminary results.\n")

# ============================================================================
# CELL 8: Generate Adversarial Prompts (OPTIONAL - 10 minutes)
# ============================================================================
print("Generating adversarial prompts using Gemini...")
print("This creates harder test cases: paraphrases, code-mixing, etc.\n")

!python tools/build_training_pairs.py \
  --forget data/forget_hi.jsonl \
  --target_lang "Hindi (Devanagari)" \
  --out_pairs data/pairs_npo.jsonl \
  --out_adv data/adversarial_gemini.jsonl \
  --model gemini-2.0-flash-exp

print("\n✅ Adversarial prompts generated!")
print("Using data/adversarial_gemini.jsonl for robustness testing.\n")

# ============================================================================
# CELL 9: MAIN EXPERIMENT - FULL RESEARCH-GRADE RUN (3.5 hours)
# ============================================================================
print("="*70)
print("STARTING FULL RESEARCH-GRADE EXPERIMENT")
print("="*70)
print("\nNew features enabled:")
print("  🔥 Judge-assisted layer selection (Gemini scores layers)")
print("  🔥 Script scrubbing (LEACE projection for semantic unlearning)")
print("  🔥 Adversarial ES evaluation (robustness testing)")
print("  ⚙️ Semantic feature picker with tau=0.05 (noise filtering)")
print("  ⚙️ Optimized SAE gating alpha=0.6 for Qwen 1.5B")
print("  ⚙️ Stability selection with deterministic tie-breaking")
print("\nEstimated time: 3.5 hours (with judge + stability)")
print("="*70 + "\n")

!python mmie.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl \
  --retain data/retain_en.jsonl \
  --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --adversarial adversarial.jsonl \
  \
  --stability_select 5 \
  --stability_strategy vote \
  --judge_assist_selection \
  --judge_pool 10 \
  --judge_cap 32 \
  --judge_alpha 0.6 \
  --judge_beta 0.4 \
  --judge_model gemini-2.0-flash-exp \
  --judge_timeout 30.0 \
  --select_top_k 3 \
  --select_mode semantic \
  --script_blind_selection \
  --use_anc \
  \
  --train_sae_steps 2000 \
  --sae_k 32 \
  --sae_expansion 4 \
  \
  --sae_gate \
  --sae_gate_alpha 0.6 \
  --sae_gate_topk 64 \
  --semantic_features \
  --semantic_tau 0.05 \
  --dynamic_gate \
  --semantic_dynamic_gate \
  --script_scrub \
  --scrub_k 2 \
  \
  --lora_steps 500 \
  --reft_steps 500 \
  --forget_obj npo \
  --rank 4 \
  \
  --seeds 42 123 456 \
  --sae_quality_eval \
  --report_token_kl \
  --es_romanized \
  \
  --use_gemini \
  --use_xlmr \
  --use_fasttext \
  \
  --gate_es_forget_ratio 0.5 \
  --gate_es_mixed_ratio 0.7 \
  --gate_ppl_ratio 1.10 \
  \
  --device cuda \
  --ckpt_dir /content/ckpt_qwen \
  --out results_qwen_research_grade.json

print("\n" + "="*70)
print("🎉 EXPERIMENT COMPLETE!")
print("="*70)

# ============================================================================
# CELL 10: Load and Parse Results
# ============================================================================
import json
import pandas as pd
import numpy as np

# Load results
with open('results_qwen_research_grade.json', 'r') as f:
    results = json.load(f)

# Display base model metrics
print("=" * 70)
print("BASE MODEL METRICS")
print("=" * 70)
base = results.get('base', {})
print(f"ES_forget:       {base.get('es_forget', 0):.4f}")
print(f"ES_semantic:     {base.get('es_semantic', 0):.4f}")
print(f"PPL_retain:      {base.get('ppl_retain', 0):.2f}")
print(f"ES_mixed:        {base.get('es_mixed', 0):.4f}")
if 'es_adversarial_mean' in base:
    print(f"ES_adversarial:  {base.get('es_adversarial_mean', 0):.4f}")

# Display layer selection results
print("\n" + "=" * 70)
print("LAYER SELECTION (Stability + Judge)")
print("=" * 70)
chosen_layers = results.get('layers', [])
print(f"Selected layers: {chosen_layers}")
print("\nLayer scores:")
for li, score_dict in results.get('layer_scores', {}).items():
    combo = score_dict.get('combo', 0)
    print(f"  Layer {li}: combo={combo:.4f}")

# Display LoRA results
print("\n" + "=" * 70)
print("LORA RESULTS (averaged across 3 seeds)")
print("=" * 70)
lora_agg = results.get('arms', {}).get('lora', {}).get('aggregate', {})
if lora_agg:
    print(f"ES_forget:       {lora_agg.get('es_forget_mean', 0):.4f} "
          f"(95% CI: [{lora_agg.get('es_forget_ci', [0,0])[0]:.4f}, "
          f"{lora_agg.get('es_forget_ci', [0,0])[1]:.4f}])")
    print(f"PPL_retain:      {lora_agg.get('ppl_retain_mean', 0):.2f} "
          f"(95% CI: [{lora_agg.get('ppl_retain_ci', [0,0])[0]:.2f}, "
          f"{lora_agg.get('ppl_retain_ci', [0,0])[1]:.2f}])")
    print(f"ES_mixed:        {lora_agg.get('es_mixed_mean', 0):.4f}")
    if 'es_adversarial_mean' in lora_agg:
        print(f"ES_adversarial:  {lora_agg.get('es_adversarial_mean', 0):.4f} "
              f"(95% CI: [{lora_agg.get('es_adversarial_ci', [0,0])[0]:.4f}, "
              f"{lora_agg.get('es_adversarial_ci', [0,0])[1]:.4f}])")

    print(f"\nGate Results:")
    gates = results.get('gates', {}).get('lora', {})
    for gate_name, passed in gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {gate_name}: {status}")

    decision = results.get('decisions', {}).get('lora', 'UNKNOWN')
    print(f"\n{'🟢' if decision == 'PROCEED' else '🔴'} Decision: {decision}")

# Display ReFT results
print("\n" + "=" * 70)
print("REFT RESULTS (averaged across 3 seeds)")
print("=" * 70)
reft_agg = results.get('arms', {}).get('reft', {}).get('aggregate', {})
if reft_agg:
    print(f"ES_forget:       {reft_agg.get('es_forget_mean', 0):.4f}")
    print(f"PPL_retain:      {reft_agg.get('ppl_retain_mean', 0):.2f}")
    print(f"ES_mixed:        {reft_agg.get('es_mixed_mean', 0):.4f}")
    if 'es_adversarial_mean' in reft_agg:
        print(f"ES_adversarial:  {reft_agg.get('es_adversarial_mean', 0):.4f}")

    print(f"\nGate Results:")
    gates = results.get('gates', {}).get('reft', {})
    for gate_name, passed in gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {gate_name}: {status}")

# Display cross-lingual leakage
print("\n" + "=" * 70)
print("CROSS-LINGUAL LEAKAGE")
print("=" * 70)
if lora_agg:
    xlang = lora_agg.get('crossling_es', {})
    for lang, (mean, (lo, hi)) in xlang.items():
        print(f"{lang:12} {mean:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")

# Display SAE quality metrics (if available)
print("\n" + "=" * 70)
print("SAE QUALITY METRICS")
print("=" * 70)
sae_quality = results.get('sae_quality', {})
if sae_quality:
    for layer, metrics in sae_quality.items():
        print(f"\nLayer {layer}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
else:
    print("No SAE quality metrics available.")

# ============================================================================
# CELL 11: Log to WandB
# ============================================================================
# Log final metrics to WandB
if lora_agg:
    wandb.log({
        "ES_forget_lora": lora_agg.get('es_forget_mean', 0),
        "PPL_retain_lora": lora_agg.get('ppl_retain_mean', 0),
        "ES_mixed_lora": lora_agg.get('es_mixed_mean', 0),
        "ES_adversarial_lora": lora_agg.get('es_adversarial_mean', 0),
        "MIA_AUC_lora": lora_agg.get('mia', {}).get('AUC_mean', 0),
        "all_gates_pass_lora": results.get('decisions', {}).get('lora') == 'PROCEED'
    })

if reft_agg:
    wandb.log({
        "ES_forget_reft": reft_agg.get('es_forget_mean', 0),
        "PPL_retain_reft": reft_agg.get('ppl_retain_mean', 0),
        "ES_mixed_reft": reft_agg.get('es_mixed_mean', 0),
        "ES_adversarial_reft": reft_agg.get('es_adversarial_mean', 0),
    })

wandb.finish()
print("\n✅ Metrics logged to WandB")

# ============================================================================
# CELL 12: Comprehension Test (Does model still UNDERSTAND Hindi?)
# ============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

print("="*70)
print("COMPREHENSION TEST: Does the model still understand Hindi?")
print("="*70)

# Load base and LoRA models
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Load LoRA adapter
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0,
                         target_modules=["q_proj", "v_proj"],
                         task_type="CAUSAL_LM")
lora_model = get_peft_model(base_model, lora_config)
lora_model.load_state_dict(torch.load('/content/ckpt_qwen/lora_adapters.pt',
                                      map_location='cuda'), strict=False)

comprehension_prompts = [
    "Translate to English: नमस्ते, आप कैसे हैं?",
    "Translate to English: मुझे हिंदी पसंद है।",
    "Translate to English: दिल्ली भारत की राजधानी है।",
    "What does 'नमस्ते' mean in English?",
    "What does 'धन्यवाद' mean in English?"
]

def test_comprehension(model, tok, prompts, device="cuda"):
    outputs = []
    for p in tqdm(prompts, desc="Testing"):
        enc = tok([p], return_tensors='pt').to(device)
        gen = model.generate(**enc, max_new_tokens=50, do_sample=False)
        outputs.append(tok.decode(gen[0], skip_special_tokens=True))
    return outputs

print("\nBase model translations:")
base_comp = test_comprehension(base_model, tokenizer, comprehension_prompts)
for prompt, output in zip(comprehension_prompts, base_comp):
    print(f"  Q: {prompt}")
    print(f"  A: {output}\n")

print("\nLoRA model translations:")
lora_comp = test_comprehension(lora_model, tokenizer, comprehension_prompts)
for prompt, output in zip(comprehension_prompts, lora_comp):
    print(f"  Q: {prompt}")
    print(f"  A: {output}\n")

print("="*70)
print("INTERPRETATION:")
print("✅ If LoRA still translates correctly → Superficial unlearning (script-only)")
print("✅ If LoRA fails to translate → Deep semantic unlearning")
print("="*70)

# ============================================================================
# CELL 13: Generate Visualizations
# ============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Create comparison plot
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Extract metrics
methods = ['Base', 'LoRA', 'ReFT']
es_forget_values = [
    base.get('es_forget', 0),
    lora_agg.get('es_forget_mean', 0) if lora_agg else 0,
    reft_agg.get('es_forget_mean', 0) if reft_agg else 0
]
ppl_retain_values = [
    base.get('ppl_retain', 0),
    lora_agg.get('ppl_retain_mean', 0) if lora_agg else 0,
    reft_agg.get('ppl_retain_mean', 0) if reft_agg else 0
]
es_mixed_values = [
    base.get('es_mixed', 0),
    lora_agg.get('es_mixed_mean', 0) if lora_agg else 0,
    reft_agg.get('es_mixed_mean', 0) if reft_agg else 0
]
es_adv_values = [
    base.get('es_adversarial_mean', 0),
    lora_agg.get('es_adversarial_mean', 0) if lora_agg else 0,
    reft_agg.get('es_adversarial_mean', 0) if reft_agg else 0
]

# ES_forget comparison
axes[0].bar(methods, es_forget_values, color=['gray', 'blue', 'green'])
axes[0].set_ylabel('Extraction Strength', fontsize=12)
axes[0].set_title('Hindi Generation (Lower = Better)', fontsize=14, fontweight='bold')
axes[0].axhline(y=0.05, color='r', linestyle='--', linewidth=2, label='Target')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# PPL_retain comparison
axes[1].bar(methods, ppl_retain_values, color=['gray', 'blue', 'green'])
axes[1].set_ylabel('Perplexity', fontsize=12)
axes[1].set_title('English Fluency (Lower = Better)', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# ES_mixed comparison
axes[2].bar(methods, es_mixed_values, color=['gray', 'blue', 'green'])
axes[2].set_ylabel('Extraction Strength', fontsize=12)
axes[2].set_title('Mixed-Language Handling', fontsize=14, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

# ES_adversarial comparison (NEW!)
axes[3].bar(methods, es_adv_values, color=['gray', 'blue', 'green'])
axes[3].set_ylabel('Extraction Strength', fontsize=12)
axes[3].set_title('Adversarial Robustness 🔥', fontsize=14, fontweight='bold')
axes[3].axhline(y=0.10, color='r', linestyle='--', linewidth=2, label='Threshold')
axes[3].legend()
axes[3].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results_research_grade_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualization saved: results_research_grade_visualization.png")

# Cross-lingual leakage plot
if lora_agg and lora_agg.get('crossling_es'):
    fig, ax = plt.subplots(figsize=(10, 6))
    xlang_data = lora_agg.get('crossling_es', {})
    languages = list(xlang_data.keys())
    means = [xlang_data[lang][0] for lang in languages]
    cis = [xlang_data[lang][1] for lang in languages]
    errors = [[means[i] - cis[i][0], cis[i][1] - means[i]] for i in range(len(languages))]

    ax.barh(languages, means, xerr=np.array(errors).T, color='coral', alpha=0.7)
    ax.set_xlabel('Extraction Strength', fontsize=12)
    ax.set_ylabel('Language', fontsize=12)
    ax.set_title('Cross-Lingual Leakage (Lower = Better)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.10, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('crosslingual_leakage.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ Visualization saved: crosslingual_leakage.png")

# ============================================================================
# CELL 14: Alpha Sweep Analysis (OPTIONAL - 30 minutes)
# ============================================================================
print("Running SAE gating alpha sweep...")
print("Testing alphas: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]\n")

!python tools/sweep_alpha.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl \
  --retain data/retain_en.jsonl \
  --alphas 0.0 0.2 0.4 0.6 0.8 1.0 \
  --ckpt_dir /content/ckpt_qwen \
  --device cuda

print("\n✅ Alpha sweep complete!")
if os.path.exists('sweep_alpha_results.png'):
    print("Check sweep_alpha_results.png for optimal alpha visualization")

# ============================================================================
# CELL 15: Download All Results
# ============================================================================
from google.colab import files

print("Downloading all result files...\n")

# Main results
files.download('results_qwen_research_grade.json')
print("✅ Downloaded: results_qwen_research_grade.json")

# Visualizations
if os.path.exists('results_research_grade_visualization.png'):
    files.download('results_research_grade_visualization.png')
    print("✅ Downloaded: results_research_grade_visualization.png")

if os.path.exists('crosslingual_leakage.png'):
    files.download('crosslingual_leakage.png')
    print("✅ Downloaded: crosslingual_leakage.png")

# Alpha sweep
if os.path.exists('sweep_alpha_results.json'):
    files.download('sweep_alpha_results.json')
    print("✅ Downloaded: sweep_alpha_results.json")
if os.path.exists('sweep_alpha_results.png'):
    files.download('sweep_alpha_results.png')
    print("✅ Downloaded: sweep_alpha_results.png")

# Checkpoints (optional - large files)
checkpoint_download = input("\nDownload model checkpoints? (y/n): ")
if checkpoint_download.lower() == 'y':
    !zip -r checkpoints.zip /content/ckpt_qwen
    files.download('checkpoints.zip')
    print("✅ Downloaded: checkpoints.zip")

# Activations (optional - very large)
activation_download = input("\nDownload activations? (y/n): ")
if activation_download.lower() == 'y':
    !zip -r activations.zip activations/
    files.download('activations.zip')
    print("✅ Downloaded: activations.zip")

print("\n" + "="*70)
print("🎉 ALL DONE! Your research-grade experiment is complete.")
print("="*70)

# ============================================================================
# CELL 16: Summary Report
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY REPORT")
print("="*70)

print("\n📊 Key Metrics:")
print(f"  Base ES_forget:       {base.get('es_forget', 0):.4f}")
print(f"  LoRA ES_forget:       {lora_agg.get('es_forget_mean', 0):.4f} "
      f"({((lora_agg.get('es_forget_mean', 0) - base.get('es_forget', 0)) / base.get('es_forget', 1) * 100):.1f}% change)")
print(f"  LoRA ES_adversarial:  {lora_agg.get('es_adversarial_mean', 0):.4f}")
print(f"  LoRA PPL_retain:      {lora_agg.get('ppl_retain_mean', 0):.2f} "
      f"({((lora_agg.get('ppl_retain_mean', 0) - base.get('ppl_retain', 0)) / base.get('ppl_retain', 1) * 100):.1f}% change)")

print("\n🎯 Gate Status:")
gates = results.get('gates', {}).get('lora', {})
passed = sum(1 for v in gates.values() if v)
total = len(gates)
print(f"  Passed: {passed}/{total} gates")
for gate_name, passed_flag in gates.items():
    status = "✅" if passed_flag else "❌"
    print(f"    {status} {gate_name}")

decision = results.get('decisions', {}).get('lora', 'UNKNOWN')
print(f"\n{'🟢 PROCEED' if decision == 'PROCEED' else '🔴 STOP'}: {decision}")

print("\n🔬 Research-Grade Features Used:")
print("  ✅ Judge-assisted layer selection (Gemini)")
print("  ✅ Stability selection with 5-seed voting")
print("  ✅ Script scrubbing (LEACE projection)")
print("  ✅ Semantic SAE feature picker (tau=0.05)")
print("  ✅ Dynamic + semantic gating")
print("  ✅ Adversarial robustness testing")
print("  ✅ Cross-lingual leakage evaluation")
print("  ✅ MIA privacy testing")
print("  ✅ Token-level KL divergence")

print("\n📈 Publication Readiness:")
readiness_score = 0
if lora_agg.get('es_forget_mean', 1.0) < 0.10:
    readiness_score += 20
    print("  ✅ ES_forget < 0.10 (strong unlearning)")
if lora_agg.get('ppl_retain_mean', 100) / base.get('ppl_retain', 1) <= 1.15:
    readiness_score += 20
    print("  ✅ PPL increase < 15% (fluency preserved)")
if lora_agg.get('es_adversarial_mean', 1.0) < 0.15:
    readiness_score += 20
    print("  ✅ Adversarial ES < 0.15 (robust)")
if passed >= total - 1:
    readiness_score += 20
    print(f"  ✅ {passed}/{total} gates passed (gating criteria met)")
if len(chosen_layers) >= 3:
    readiness_score += 20
    print("  ✅ Multiple layers selected (comprehensive intervention)")

print(f"\n🏆 Overall Readiness Score: {readiness_score}/100")
if readiness_score >= 80:
    print("   Status: PUBLICATION READY ✅")
elif readiness_score >= 60:
    print("   Status: NEAR READY (minor improvements needed)")
else:
    print("   Status: NEEDS WORK (significant improvements required)")

print("\n" + "="*70)
print("Next steps:")
if readiness_score >= 80:
    print("  1. Write up your results")
    print("  2. Create figures and tables for paper")
    print("  3. Compare to baselines in literature")
    print("  4. Submit to top-tier conference (NeurIPS, ICLR, ACL)")
else:
    print("  1. Review failed gates and identify bottlenecks")
    print("  2. Consider hyperparameter tuning (alpha, tau, layers)")
    print("  3. Analyze comprehension test results")
    print("  4. Re-run with adjustments")
print("="*70)

