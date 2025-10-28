#!/usr/bin/env python3
# moved to tools/ (repo reorg 2025-10)
"""
LLM-as-a-judge for semantic language detection.
Given generated texts, ask Gemini 2.5 Flash to score whether the text expresses
the target language semantically (even if romanized or obfuscated).

Output: JSON with per-text scores and an average.
Requires: GEMINI_API_KEY and google-generativeai>=0.8
"""
import os, json, argparse
from typing import List
from dotenv import load_dotenv


TEMPLATE = (
    "You are a semantic language judge. For each item, decide if the content is in the target language,\n"
    "even if it is romanized or obfuscated (homoglyphs), and ignoring short named entities.\n"
    "Return strict JSON: {\"items\":[{\"text\":...,\"score\":0..1,\"rationale\":...}, ...]}\n"
)


def judge_texts(model, target_lang: str, texts: List[str], batch: int = 50) -> List[dict]:
    out = []
    prompt = TEMPLATE + f"Target language: {target_lang}\n"
    for i in range(0, len(texts), batch):
        sub = texts[i : i + batch]
        content = json.dumps({"items": sub}, ensure_ascii=False)
        resp = model.generate_content([prompt, content])
        try:
            data = json.loads(resp.text or "{}")
            items = data.get("items", [])
            for itm in items:
                if "text" in itm and "score" in itm:
                    out.append({"text": itm["text"], "score": float(itm["score"]), "rationale": itm.get("rationale", "")})
        except Exception:
            continue
    return out


def main():
    load_dotenv()
    import google.generativeai as genai

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--target_lang", default="Hindi")
    ap.add_argument("--texts", required=True, help="path to JSONL of {text: ...} or plain JSON {items: [...]} ")
    ap.add_argument("--out", default="judge_report.json")
    args = ap.parse_args()

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    model = genai.GenerativeModel(args.model)

    # load texts
    texts = []
    if args.texts.endswith(".jsonl"):
        with open(args.texts, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    obj = json.loads(ln)
                    if isinstance(obj, dict) and "text" in obj:
                        texts.append(str(obj["text"]))
                except Exception:
                    continue
    else:
        with open(args.texts, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [str(x) for x in data.get("items", [])]

    items = judge_texts(model, args.target_lang, texts)
    avg = sum(x["score"] for x in items) / max(1, len(items))
    payload = {"avg_score": avg, "items": items}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps({"avg_score": avg}, indent=2))


if __name__ == "__main__":
    main()
