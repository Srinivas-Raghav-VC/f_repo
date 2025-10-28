#!/usr/bin/env python3
"""
Environment preflight for MMIE runs.
Prints versions, CUDA info, memory hints, and key env vars.

Usage:
  python scripts/preflight.py
"""
import os, platform, sys

def line(k, v):
    print(f"{k:24} {v}")

def main():
    line("Python", sys.version.split()[0])
    line("OS", f"{platform.system()} {platform.release()} ({platform.version()})")
    try:
        import torch
        line("torch", torch.__version__)
        line("CUDA available", torch.cuda.is_available())
        if torch.cuda.is_available():
            line("CUDA device count", torch.cuda.device_count())
            try:
                i = torch.cuda.current_device()
            except Exception:
                i = 0
            name = torch.cuda.get_device_name(i)
            line("CUDA device", f"{i}: {name}")
            try:
                free, total = torch.cuda.mem_get_info()
                line("VRAM (free/total)", f"{free//(2**20)} MiB / {total//(2**20)} MiB")
            except Exception:
                pass
    except Exception as e:
        line("torch", f"not found ({e})")

    for pkg in ("transformers", "accelerate", "safetensors", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            v = getattr(mod, "__version__", "?")
            line(pkg, v)
        except Exception as e:
            line(pkg, f"not found ({e.__class__.__name__})")

    # key env vars that affect loading behavior
    for k in ("SAFETENSORS_FAST", "OFFLOAD_DIR", "LOAD_IN_8BIT", "LOAD_IN_4BIT", "HF_HOME", "HF_TOKEN"):
        v = os.environ.get(k)
        line(k, v if v is not None else "<unset>")

    print("\nHints:")
    print("- On Windows, set SAFETENSORS_FAST=0 to avoid paging file (1455) errors.")
    print("- To reduce VRAM, set OFFLOAD_DIR to a writable folder; install accelerate.")
    print("- If bitsandbytes is installed, set LOAD_IN_4BIT=1 (or 8BIT=1) for quantized loading.")

if __name__ == "__main__":
    main()

