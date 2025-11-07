#!/bin/bash
#SBATCH -J mmie_qwen15b
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

# Optional: load modules here (site-specific)
# module load cuda/12.4

export SCR=${SCR:-$PWD/slurm_scratch_$SLURM_JOB_ID}
mkdir -p "$SCR"/{repo,venv,cache/{hf,transformers,datasets,hub,torch},pip-cache,offload,ckpt,logs,auto_runs}

export XDG_CACHE_HOME=$SCR/cache
export HF_HOME=$SCR/cache/hf
export TRANSFORMERS_CACHE=$SCR/cache/transformers
export HF_DATASETS_CACHE=$SCR/cache/datasets
export HUGGINGFACE_HUB_CACHE=$SCR/cache/hub
export TORCH_HOME=$SCR/cache/torch
export PIP_CACHE_DIR=$SCR/pip-cache
export OFFLOAD_DIR=$SCR/offload
export HF_HUB_ENABLE_HF_TRANSFER=1
export SAFETENSORS_FAST=0
export TORCH_ALLOW_TF32=1
export GEMINI_API_KEY=""   # judge off by default

# Clone or use an existing checkout
REPO_DIR=${REPO_DIR:-$SCR/repo}
if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone --depth=1 https://github.com/Srinivas-Raghav-VC/f_repo "$REPO_DIR"
fi
cd "$REPO_DIR"

# Venv and deps (fast set)
python3 -m venv "$SCR/venv"
source "$SCR/venv/bin/activate"
python -m pip install -U pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install -r requirements-fast.txt

LOG_DIR=$SCR/logs VENV="$SCR/venv" DISABLE_JUDGE=1 \
  ./run_mmie_hpc.sh \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl \
    --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl \
    --ckpt_dir "$SCR/ckpt" --no_quantization --auto --auto_plots --device cuda

echo "[sbatch] outputs in $SCR"

