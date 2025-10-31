# Minimal local sanity check for MMIE on Qwen 1.5B (PowerShell).
# - Verifies environment
# - Checks dataset presence
# - Runs a tiny selection-only pass (no training)

Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))

Write-Host "[sanity] Environment preflight"
# Resolve Python interpreter name
function Get-Py {
  if (Get-Command python -ErrorAction SilentlyContinue) { return 'python' }
  elseif (Get-Command python3 -ErrorAction SilentlyContinue) { return 'python3' }
  else { throw 'No Python interpreter found (need python or python3)' }
}
$PY = Get-Py

& $PY scripts/preflight.py

Write-Host "[sanity] Dataset check"
& $PY scripts/check_datasets.py --paths `
  data/forget_hi.jsonl data/retain_en.jsonl data/mixed.jsonl `
  data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl

# Disable judge by default for speed in sanity runs
if (-not $env:GEMINI_API_KEY) { $env:GEMINI_API_KEY = '' }
# Prefer 8-bit if bitsandbytes is available
if (-not $env:LOAD_IN_8BIT) { $env:LOAD_IN_8BIT = '1' }
if (-not $env:SAFETENSORS_FAST) { $env:SAFETENSORS_FAST = '0' }

$device = & $PY - << 'PY'
try:
    import torch
    print('cuda' if (hasattr(torch,'cuda') and torch.cuda.is_available()) else 'cpu')
except Exception:
    print('cpu')
PY

Write-Host "[sanity] Using device: $device"

Write-Host "[sanity] Selection-only smoke (no training)"
& $PY mmie.py `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl `
  --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl `
  --lora_steps 0 --reft_steps 0 --train_sae_steps 0 `
  --sample_cap 20 --max_len 64 `
  --select_mode semantic --min_layer 6 --select_top_k 2 `
  --print_layer_scores `
  --device $device `
  --out sanity_selection.json

if ($device -eq 'cuda') {
  Write-Host "[sanity] Optional SAE gate smoke (very short)"
  & $PY mmie.py `
    --model Qwen/Qwen2.5-1.5B-Instruct `
    --forget data/forget_hi.jsonl --retain data/retain_en.jsonl --mixed data/mixed.jsonl `
    --xlang data/urdu.jsonl data/punjabi.jsonl data/bengali.jsonl `
    --ckpt_dir ckpt_sanity `
    --train_sae_steps 50 --sae_k 32 --sae_expansion 8 `
    --lora_steps 0 --reft_steps 0 `
    --select_mode semantic --min_layer 6 --select_top_k 1 `
    --sample_cap 40 --max_len 64 `
    --device cuda `
    --out sanity_sae_gate.json
}

Write-Host "[sanity] Done. Outputs: sanity_selection.json (and sanity_sae_gate.json if CUDA)."
