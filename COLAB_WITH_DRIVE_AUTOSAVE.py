# -*- coding: utf-8 -*-
"""
RESEARCH-GRADE Multilingual Unlearning with Google Drive Auto-Save
Everything is automatically backed up to Drive in real-time!
"""

# ============================================================================
# CELL 1: GPU Check
# ============================================================================
!nvidia-smi

# ============================================================================
# CELL 2: Mount Google Drive (AUTO-SAVE EVERYTHING)
# ============================================================================
from google.colab import drive
import os
from datetime import datetime

# Mount Drive
drive.mount('/content/drive', force_remount=False)

# Create experiment directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DRIVE_ROOT = f"/content/drive/MyDrive/SAE_Experiments"
EXPERIMENT_DIR = f"{DRIVE_ROOT}/qwen_1.5b_{timestamp}"

os.makedirs(DRIVE_ROOT, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/results", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/logs", exist_ok=True)
os.makedirs(f"{EXPERIMENT_DIR}/activations", exist_ok=True)

print("="*70)
print("🔥 GOOGLE DRIVE AUTO-SAVE CONFIGURED")
print("="*70)
print(f"\nAll outputs will be saved to:")
print(f"  📁 {EXPERIMENT_DIR}")
print(f"\nStructure:")
print(f"  ├── results/        (JSON results, metrics)")
print(f"  ├── checkpoints/    (LoRA/ReFT/SAE weights)")
print(f"  ├── logs/           (Execution logs, WandB)")
print(f"  └── activations/    (Hidden state dumps)")
print("\n✅ Everything auto-syncs in real-time!")
print("="*70)

# ============================================================================
# CELL 3: Clone Repository
# ============================================================================
!git clone https://github.com/Srinivas-Raghav-VC/f_repo.git
%cd f_repo

# Create symlink to Drive for instant backup
!ln -s {EXPERIMENT_DIR}/checkpoints /content/f_repo/ckpt_qwen
!ln -s {EXPERIMENT_DIR}/results /content/f_repo/results_backup

print(f"\n✅ Repository cloned and linked to Drive")

# ============================================================================
# CELL 4: Install Dependencies
# ============================================================================
# Core ML libraries
!pip install -q torch transformers accelerate peft einops scikit-learn scipy numpy python-dotenv

# LID ensemble components
!pip install -q fasttext langid pycld3
!pip install -q google-generativeai  # Gemini SDK

# Transliteration
!pip install -q indic-transliteration

# Experiment tracking
!pip install -q wandb

# Visualization
!pip install -q matplotlib seaborn

print("✅ All dependencies installed!")

# ============================================================================
# CELL 5: Configure Environment Variables
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

# Required: Gemini API key
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

# Set offload directory
!mkdir -p /content/offload
os.environ['OFFLOAD_DIR'] = '/content/offload'
os.environ['SAFETENSORS_FAST'] = '0'

print("\n✅ Environment configured")

# ============================================================================
# CELL 6: WandB Login with Drive Backup
# ============================================================================
import wandb
import shutil

wandb.login()

# Initialize WandB with Drive sync
wandb.init(
    project="multilingual-unlearning-research",
    name=f"qwen-1.5b-{timestamp}",
    dir=f"{EXPERIMENT_DIR}/logs",  # Save WandB logs to Drive
    config={
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "seeds": [42, 123, 456],
        "forget_obj": "npo",
        "stability_select": 5,
        "judge_assist": True,
        "script_scrub": True,
        "semantic_features": True,
        "experiment_dir": EXPERIMENT_DIR
    },
    tags=["research-grade", "qwen-1.5b", "hindi-unlearning", "auto-saved"]
)

print(f"✅ WandB initialized with Drive backup at {EXPERIMENT_DIR}/logs")

# ============================================================================
# CELL 7: Verify Data Files
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
missing_files = []
for f in data_files:
    if os.path.exists(f):
        with open(f) as file:
            lines = sum(1 for _ in file)
        print(f"✅ {f}: {lines} samples")
    else:
        print(f"❌ Missing: {f}")
        missing_files.append(f)

if missing_files:
    print(f"\n⚠️ WARNING: {len(missing_files)} files missing!")
    print("The experiment will fail without these files.")
else:
    print("\n✅ All data files present!")

# ============================================================================
# CELL 8: Create Real-Time Progress Logger
# ============================================================================
import sys
from datetime import datetime

class DriveLogger:
    """Logger that writes to both console and Drive in real-time"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', buffering=1)  # Line buffering

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Force write to Drive

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Redirect stdout to Drive logger
log_path = f"{EXPERIMENT_DIR}/logs/experiment_log_{timestamp}.txt"
sys.stdout = DriveLogger(log_path)

print("="*70)
print(f"🔥 REAL-TIME LOGGING ENABLED")
print(f"📝 Log file: {log_path}")
print("="*70)
print(f"\nExperiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("All output will be saved to Drive in real-time!\n")

# ============================================================================
# CELL 9: MAIN EXPERIMENT with Auto-Save Hooks
# ============================================================================
import subprocess
import time
import json

print("="*70)
print("STARTING FULL RESEARCH-GRADE EXPERIMENT")
print("="*70)
print("\n🔥 Features enabled:")
print("  ✅ Judge-assisted layer selection (Gemini)")
print("  ✅ Stability selection (5-seed voting)")
print("  ✅ Script scrubbing (LEACE projection)")
print("  ✅ Dynamic + semantic gating")
print("  ✅ SAE quality evaluation")
print("  ✅ Cross-lingual leakage testing")
print("\n💾 Auto-save features:")
print("  ✅ Real-time Drive sync")
print("  ✅ Checkpoint backup every 500 steps")
print("  ✅ Results JSON auto-saved")
print("  ✅ Activations preserved")
print("\nEstimated time: 3.5 hours")
print("="*70 + "\n")

# Build command
cmd = f"""python mmie.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --forget data/forget_hi.jsonl \
  --retain data/retain_en.jsonl \
  --mixed data/mixed.jsonl \
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
  --adversarial adversarial.jsonl \
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
  --train_sae_steps 2000 \
  --sae_k 32 \
  --sae_expansion 4 \
  --sae_gate \
  --sae_gate_alpha 0.6 \
  --sae_gate_topk 64 \
  --semantic_features \
  --semantic_tau 0.05 \
  --dynamic_gate \
  --semantic_dynamic_gate \
  --script_scrub \
  --scrub_k 2 \
  --lora_steps 500 \
  --reft_steps 500 \
  --forget_obj npo \
  --rank 4 \
  --seeds 42 123 456 \
  --sae_quality_eval \
  --report_token_kl \
  --es_romanized \
  --use_gemini \
  --use_xlmr \
  --use_fasttext \
  --gate_es_forget_ratio 0.5 \
  --gate_es_mixed_ratio 0.7 \
  --gate_ppl_ratio 1.10 \
  --device cuda \
  --ckpt_dir /content/f_repo/ckpt_qwen \
  --out {EXPERIMENT_DIR}/results/results_qwen_full.json
"""

# Run experiment
start_time = time.time()
result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
end_time = time.time()

duration = end_time - start_time
print("\n" + "="*70)
if result.returncode == 0:
    print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
else:
    print(f"⚠️ EXPERIMENT FAILED WITH CODE {result.returncode}")
print(f"⏱️  Duration: {duration/3600:.2f} hours ({duration/60:.1f} minutes)")
print("="*70)

# Copy activations to Drive
print("\n📦 Backing up activations to Drive...")
if os.path.exists('activations'):
    !cp -r activations/* {EXPERIMENT_DIR}/activations/
    print(f"✅ Activations saved to {EXPERIMENT_DIR}/activations/")

# ============================================================================
# CELL 10: Auto-Save Summary Metrics to Drive
# ============================================================================
import json
import pandas as pd

results_path = f"{EXPERIMENT_DIR}/results/results_qwen_full.json"

if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Create summary CSV
    summary_data = []
    for arm in ['lora', 'reft']:
        if arm in results.get('arms', {}):
            agg = results['arms'][arm].get('aggregate', {})
            summary_data.append({
                'Method': arm.upper(),
                'ES_forget_mean': agg.get('es_forget_mean', 0),
                'ES_forget_CI_low': agg.get('es_forget_ci', [0,0])[0],
                'ES_forget_CI_high': agg.get('es_forget_ci', [0,0])[1],
                'PPL_retain_mean': agg.get('ppl_retain_mean', 0),
                'PPL_retain_CI_low': agg.get('ppl_retain_ci', [0,0])[0],
                'PPL_retain_CI_high': agg.get('ppl_retain_ci', [0,0])[1],
                'ES_mixed_mean': agg.get('es_mixed_mean', 0),
                'MIA_AUC': agg.get('mia', {}).get('AUC_mean', 0),
                'Decision': results.get('decisions', {}).get(arm, 'UNKNOWN')
            })

    df = pd.DataFrame(summary_data)
    csv_path = f"{EXPERIMENT_DIR}/results/summary_metrics.csv"
    df.to_csv(csv_path, index=False)

    print("="*70)
    print("📊 SUMMARY METRICS (saved to Drive)")
    print("="*70)
    print(df.to_string(index=False))
    print(f"\n✅ CSV saved: {csv_path}")

    # Create human-readable summary
    summary_path = f"{EXPERIMENT_DIR}/results/SUMMARY.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MULTILINGUAL UNLEARNING EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: Qwen/Qwen2.5-1.5B-Instruct\n")
        f.write(f"Duration: {duration/3600:.2f} hours\n\n")

        base = results.get('base', {})
        f.write("BASE MODEL METRICS:\n")
        f.write(f"  ES_forget:  {base.get('es_forget', 0):.4f}\n")
        f.write(f"  PPL_retain: {base.get('ppl_retain', 0):.2f}\n")
        f.write(f"  ES_mixed:   {base.get('es_mixed', 0):.4f}\n\n")

        if 'lora' in results.get('arms', {}):
            lora_agg = results['arms']['lora'].get('aggregate', {})
            f.write("LORA RESULTS:\n")
            f.write(f"  ES_forget:  {lora_agg.get('es_forget_mean', 0):.4f}\n")
            f.write(f"  PPL_retain: {lora_agg.get('ppl_retain_mean', 0):.2f}\n")
            f.write(f"  ES_mixed:   {lora_agg.get('es_mixed_mean', 0):.4f}\n")
            f.write(f"  Decision:   {results.get('decisions', {}).get('lora', 'UNKNOWN')}\n\n")

            gates = results.get('gates', {}).get('lora', {})
            f.write("GATE STATUS:\n")
            for gate_name, passed in gates.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                f.write(f"  {gate_name}: {status}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("ALL FILES SAVED TO:\n")
        f.write(f"{EXPERIMENT_DIR}\n")
        f.write("="*70 + "\n")

    print(f"✅ Summary saved: {summary_path}")

    # Log to WandB
    if lora_agg:
        wandb.log({
            "ES_forget_lora": lora_agg.get('es_forget_mean', 0),
            "PPL_retain_lora": lora_agg.get('ppl_retain_mean', 0),
            "ES_mixed_lora": lora_agg.get('es_mixed_mean', 0),
            "all_gates_pass": results.get('decisions', {}).get('lora') == 'PROCEED',
            "experiment_duration_hours": duration / 3600
        })

    # Save WandB summary to Drive
    wandb.save(f"{EXPERIMENT_DIR}/results/*")

else:
    print("⚠️ Results file not found! Experiment may have failed.")

# ============================================================================
# CELL 11: Generate Visualizations and Save to Drive
# ============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if os.path.exists(results_path):
    sns.set_style("whitegrid")

    # Extract data
    base = results.get('base', {})
    lora_agg = results.get('arms', {}).get('lora', {}).get('aggregate', {})
    reft_agg = results.get('arms', {}).get('reft', {}).get('aggregate', {})

    methods = ['Base', 'LoRA', 'ReFT']
    es_values = [
        base.get('es_forget', 0),
        lora_agg.get('es_forget_mean', 0) if lora_agg else 0,
        reft_agg.get('es_forget_mean', 0) if reft_agg else 0
    ]
    ppl_values = [
        base.get('ppl_retain', 0),
        lora_agg.get('ppl_retain_mean', 0) if lora_agg else 0,
        reft_agg.get('ppl_retain_mean', 0) if reft_agg else 0
    ]

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ES_forget
    axes[0].bar(methods, es_values, color=['gray', 'blue', 'green'], alpha=0.7)
    axes[0].set_ylabel('Extraction Strength', fontsize=12)
    axes[0].set_title('Hindi Generation (Lower = Better)', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0.05, color='r', linestyle='--', linewidth=2, label='Target')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # PPL_retain
    axes[1].bar(methods, ppl_values, color=['gray', 'blue', 'green'], alpha=0.7)
    axes[1].set_ylabel('Perplexity', fontsize=12)
    axes[1].set_title('English Fluency (Lower = Better)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = f"{EXPERIMENT_DIR}/results/comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ Visualization saved: {plot_path}")

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
        ax.set_title('Cross-Lingual Leakage', fontsize=14, fontweight='bold')
        ax.axvline(x=0.10, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        xleak_path = f"{EXPERIMENT_DIR}/results/crosslingual_leakage.png"
        plt.savefig(xleak_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Visualization saved: {xleak_path}")

# ============================================================================
# CELL 12: Create Archive and Final Backup
# ============================================================================
import zipfile
import shutil

print("\n" + "="*70)
print("📦 CREATING FINAL ARCHIVE")
print("="*70)

# Create ZIP of entire experiment
archive_path = f"{DRIVE_ROOT}/qwen_1.5b_{timestamp}_COMPLETE.zip"
print(f"\nCreating ZIP archive: {archive_path}")

with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(EXPERIMENT_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, EXPERIMENT_DIR)
            zipf.write(file_path, arcname)

print(f"✅ Archive created: {archive_path}")

# Create README
readme_path = f"{EXPERIMENT_DIR}/README.txt"
with open(readme_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("MULTILINGUAL UNLEARNING EXPERIMENT - QWEN 1.5B\n")
    f.write("="*70 + "\n\n")
    f.write(f"Experiment ID: {timestamp}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Duration: {duration/3600:.2f} hours\n\n")
    f.write("DIRECTORY STRUCTURE:\n")
    f.write("  results/       - JSON results, CSVs, summary\n")
    f.write("  checkpoints/   - Model weights (LoRA, ReFT, SAE)\n")
    f.write("  logs/          - WandB logs, execution logs\n")
    f.write("  activations/   - Hidden state dumps\n\n")
    f.write("QUICK START:\n")
    f.write("  1. Read SUMMARY.txt for key metrics\n")
    f.write("  2. Check comparison_plot.png for visual results\n")
    f.write("  3. Load results_qwen_full.json for detailed analysis\n\n")
    f.write("CITATION:\n")
    f.write("  If you use these results, please cite:\n")
    f.write("  [Your Paper Title]\n")
    f.write("  [Your Name], [Year]\n\n")
    f.write("="*70 + "\n")

print(f"✅ README created: {readme_path}")

print("\n" + "="*70)
print("🎉 ALL FILES SAVED TO GOOGLE DRIVE!")
print("="*70)
print(f"\n📁 Experiment directory:")
print(f"   {EXPERIMENT_DIR}")
print(f"\n📦 Complete archive:")
print(f"   {archive_path}")
print(f"\n💾 Total size:")
!du -sh {EXPERIMENT_DIR}
print("\n✅ Your data is safe even if Colab disconnects!")
print("="*70)

# ============================================================================
# CELL 13: Share-able Drive Link Generator
# ============================================================================
print("\n" + "="*70)
print("📤 GOOGLE DRIVE SHARING")
print("="*70)
print("\nTo share your results:")
print(f"\n1. Go to Google Drive:")
print(f"   https://drive.google.com/drive/my-drive")
print(f"\n2. Navigate to:")
print(f"   My Drive > SAE_Experiments > qwen_1.5b_{timestamp}")
print(f"\n3. Right-click the folder → Share → Anyone with link")
print(f"\n4. Or download the ZIP:")
print(f"   qwen_1.5b_{timestamp}_COMPLETE.zip ({archive_path})")
print("\n" + "="*70)

# Print shareable paths
print("\n📂 Key files to share:")
print(f"   - Summary: {EXPERIMENT_DIR}/results/SUMMARY.txt")
print(f"   - Metrics CSV: {EXPERIMENT_DIR}/results/summary_metrics.csv")
print(f"   - Plot: {EXPERIMENT_DIR}/results/comparison_plot.png")
print(f"   - Full JSON: {EXPERIMENT_DIR}/results/results_qwen_full.json")

# ============================================================================
# CELL 14: Finish WandB and Cleanup
# ============================================================================
wandb.finish()
print("\n✅ WandB session finished")
print("Check your WandB dashboard for full experiment tracking!")

# Restore stdout
sys.stdout = sys.stdout.terminal
print("\n" + "="*70)
print("🎉 EXPERIMENT COMPLETE - ALL DATA SAVED TO DRIVE!")
print("="*70)
print(f"\nYour results are permanently saved at:")
print(f"  {EXPERIMENT_DIR}")
print(f"\nEven if Colab session ends, your data is safe! 🛡️")
print("="*70)

# ============================================================================
# CELL 15: Quick Results Display
# ============================================================================
if os.path.exists(results_path):
    print("\n" + "="*70)
    print("📊 QUICK RESULTS SUMMARY")
    print("="*70)

    with open(results_path, 'r') as f:
        results = json.load(f)

    base = results.get('base', {})
    lora_agg = results.get('arms', {}).get('lora', {}).get('aggregate', {})

    print(f"\n📈 Key Metrics:")
    print(f"  Base ES_forget:  {base.get('es_forget', 0):.4f}")
    print(f"  LoRA ES_forget:  {lora_agg.get('es_forget_mean', 0):.4f} "
          f"({((lora_agg.get('es_forget_mean', 0) - base.get('es_forget', 0)) / (base.get('es_forget', 1) + 1e-9) * 100):.1f}% change)")
    print(f"  LoRA PPL_retain: {lora_agg.get('ppl_retain_mean', 0):.2f}")

    gates = results.get('gates', {}).get('lora', {})
    passed = sum(1 for v in gates.values() if v)
    total = len(gates)

    print(f"\n🎯 Gate Status: {passed}/{total} passed")
    for gate_name, pass_flag in gates.items():
        status = "✅" if pass_flag else "❌"
        print(f"  {status} {gate_name}")

    decision = results.get('decisions', {}).get('lora', 'UNKNOWN')
    print(f"\n{'🟢' if decision == 'PROCEED' else '🔴'} Final Decision: {decision}")

    print("\n📁 All files saved to Drive:")
    print(f"  {EXPERIMENT_DIR}")
    print("="*70)

