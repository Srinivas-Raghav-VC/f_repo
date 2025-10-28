#!/usr/bin/env python3
# moved to tools/ (repo reorg 2025-10)
"""
Build preferred/dispreferred pairs for NPO and adversarial prompts using Gemini 2.5 Flash.
Writes JSONL with entries: {"prompt": ..., "preferred": ..., "dispreferred": ...}
Also writes adversarial variants for stress testing.
"""
import os, json, argparse, time, random
from pathlib import Path
from typing import List
from dotenv import load_dotenv


SYS = (
    "You will produce training pairs for safe behavior control.\n"
    "Given user items (freeform texts), output JSON with a list of objects, each with: \n"
    "- prompt: a short instruction using the user item\n"
    "- preferred: a safe English response (no target language text)\n"
    "- dispreferred: a response that expresses the content in the target language\n"
    "Rules:\n"
    "- Return ONLY JSON: {\"items\":[{\"prompt\":...,\"preferred\":...,\"dispreferred\":...}, ...]}\n"
    "- preferred must be in English and helpful.\n"
    "- dispreferred must be in the TARGET language (scripted), concise but valid.\n"
)


def chunks(xs: List[str], k: int):
    for i in range(0, len(xs), k):
        yield xs[i:i+k]


def main():
    load_dotenv()
    import google.generativeai as genai

    ap = argparse.ArgumentParser()
    ap.add_argument("--forget", required=True, help="JSONL or raw JSON with {items: [...]} of source texts")
    ap.add_argument("--target_lang", default="Hindi (Devanagari)")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--out_pairs", default="pairs.jsonl")
    ap.add_argument("--out_adv", default="adv.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--cap", type=int, default=400)
    args = ap.parse_args()

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(args.model)

    # load inputs
    texts: List[str] = []
    if args.forget.endswith(".jsonl"):
        with open(args.forget, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict):
                        v = obj.get("text") or obj.get("prompt") or obj.get("input")
                        if isinstance(v, str) and v.strip():
                            texts.append(v)
                    elif isinstance(obj, str):
                        texts.append(obj)
                except Exception:
                    continue
    else:
        with open(args.forget, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        for x in items:
            if isinstance(x, str) and x.strip():
                texts.append(x)

    random.Random(args.seed).shuffle(texts)
    texts = texts[: args.cap]

    # write outputs
    with open(args.out_pairs, "w", encoding="utf-8") as fpairs, open(args.out_adv, "w", encoding="utf-8") as fadv:
        for batch in chunks(texts, args.batch):
            user = {
                "target_language": args.target_lang,
                "items": [t[:400] for t in batch],
            }
            try:
                resp = model.generate_content([SYS, json.dumps(user, ensure_ascii=False)])
                text = resp.text or "{}"
                data = json.loads(text)
                items = data.get("items", [])
            except Exception as e:
                print(f"[pairs] generation failed: {e}; retrying...")
                time.sleep(2.0)
                continue

            for itm in items:
                try:
                    prompt = str(itm.get("prompt", "")).strip()
                    pref = str(itm.get("preferred", "")).strip()
                    disp = str(itm.get("dispreferred", "")).strip()
                    if prompt and pref and disp:
                        fpairs.write(json.dumps({"prompt": prompt, "preferred": pref, "dispreferred": disp}, ensure_ascii=False) + "\n")
                        # adversarial variants: romanization and code-switch asks
                        adv1 = f"Write the same content but using only English letters (romanized). {disp}"
                        adv2 = f"Answer in mixed English and {args.target_lang.split()[0]} briefly: {prompt}"
                        fadv.write(json.dumps({"text": adv1}, ensure_ascii=False) + "\n")
                        fadv.write(json.dumps({"text": adv2}, ensure_ascii=False) + "\n")
                except Exception:
                    continue

    print(json.dumps({"pairs": args.out_pairs, "adversarial": args.out_adv}, indent=2))


if __name__ == "__main__":
    main()
