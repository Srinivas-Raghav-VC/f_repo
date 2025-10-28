#!/usr/bin/env python3
# moved to tools/ (repo reorg 2025-10)
import os, json, argparse
from pathlib import Path
import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

# reuse utilities from mmie
from mmie import LIDEnsemble, LIDConfig, generate, extraction_strength, token_kl_to_base, nll, loader, load_causal_lm


def maybe_resume_lora(model, tok, device):
    try:
        from peft import LoraConfig, get_peft_model
        cfg=LoraConfig(r=4,lora_alpha=16,lora_dropout=0.0,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
        model=get_peft_model(model,cfg)
        ckpt_dir = os.environ.get("CKPT_DIR", ".")
        path = os.path.join(ckpt_dir, "lora_adapters.pt")
        if os.path.exists(path):
            sd=torch.load(path, map_location=device)
            model.load_state_dict(sd, strict=False)
            print(f"[rev] loaded {path}")
        else:
            print(f"[rev] no adapter at {path}; attached fresh LoRA for recovery test")
        return model
    except Exception as e:
        print(f"[rev] PEFT not available: {e}")
        return model


@torch.no_grad()
def metrics(model, base, tok, lid, forget, retain, device, max_len=128):
    es = extraction_strength(generate(model,tok,forget[:200],device,max_new_tokens=64), lid, target_code="hi", use_script_guard=True)
    kl_vals=[]
    for batch in [retain[i:i+8] for i in range(0, min(len(retain),120), 8)]:
        enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=max_len).to(device)
        kl_vals.append(token_kl_to_base(model, base, enc).item())
    tkl = float(np.mean(kl_vals)) if kl_vals else 0.0
    return es, tkl


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--forget", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out", default="reversibility_report.json")
    args = ap.parse_args()

    hf_token=os.environ.get("HF_TOKEN")
    device=args.device

    # data
    from mmie import read_jsonl
    forget = read_jsonl(args.forget)
    retain = read_jsonl(args.retain)

    # LID
    lid=LIDEnsemble(LIDConfig(vote_require_majority=True))

    # models
    tok=AutoTokenizer.from_pretrained(args.model, token=hf_token, use_fast=True)
    base=load_causal_lm(args.model, tok, device, hf_token, eval_mode=True)
    [p.requires_grad_(False) for p in base.parameters()]

    edited=load_causal_lm(args.model, tok, device, hf_token)
    edited = maybe_resume_lora(edited, tok, device)
    edited.train()

    # pre metrics
    es_pre, tkl_pre = metrics(edited, base, tok, lid, forget, retain, device, args.max_len)

    # tiny recovery finetune on forget only
    Lf = loader(tok, forget, device, bs=8, max_len=args.max_len)
    itf = iter(Lf)
    opt = torch.optim.AdamW([p for p in edited.parameters() if p.requires_grad], lr=args.lr)
    for _ in range(args.steps):
        try:
            batch = next(itf)
        except StopIteration:
            itf = iter(Lf); batch = next(itf)
        loss = nll(edited, batch)
        opt.zero_grad(); loss.backward(); opt.step()
    edited.eval()

    # post metrics
    es_post, tkl_post = metrics(edited, base, tok, lid, forget, retain, device, args.max_len)

    payload = {
        "pre": {"ES_forget": es_pre, "token_KL_retain": tkl_pre},
        "post": {"ES_forget": es_post, "token_KL_retain": tkl_post},
        "steps": args.steps, "lr": args.lr
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
