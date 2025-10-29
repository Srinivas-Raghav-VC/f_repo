#!/usr/bin/env python3
"""
LLM-as-a-judge for semantic language detection using the official google.genai client.
Given texts, ask a Gemini model to score whether each text expresses the target language
semantically (even if romanized). Saves JSON with per-text scores and average.
Requires: GEMINI_API_KEY and google-genai>=1.33.0
"""
import os, json, argparse
from typing import List
from dotenv import load_dotenv

from google import genai
from google.genai import types

TEMPLATE = (
    "You are a semantic language judge. For each item, decide if the content is in the target language,\n"
    "even if it is romanized or obfuscated (homoglyphs), and ignoring short named entities.\n"
    'Return strict JSON following the schema.'
)

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"}
                },
                "required": ["score"],
                "additionalProperties": True
            }
        }
    },
    "required": ["items"]
}


def judge_texts(client: genai.Client, model_name: str, target_lang: str, texts: List[str], batch: int = 50) -> List[dict]:
    out: List[dict] = []
    prompt = f"{TEMPLATE}\nTarget language: {target_lang}\n"
    for i in range(0, len(texts), batch):
        sub = texts[i : i + batch]
        content = json.dumps({"items": sub}, ensure_ascii=False)
        resp = client.models.generate_content(
            model=model_name,
            contents=[prompt, content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=SCHEMA,
            ),
        )
        data = getattr(resp, "parsed", None)
        if not data:
            try:
                data = json.loads(resp.text or "{}")
            except Exception:
                data = {}
        items = data.get("items", [])
        for itm in items:
            try:
                out.append({
                    "text": itm.get("text", ""),
                    "score": float(itm.get("score", 0.0)),
                    "rationale": itm.get("rationale", "")
                })
            except Exception:
                continue
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--target_lang", default="Hindi")
    ap.add_argument("--texts", required=True, help="path to JSONL of {text: ...} or plain JSON {items: [...]} ")
    ap.add_argument("--out", default="judge_report.json")
    args = ap.parse_args()

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = genai.Client(api_key=key)

    # load texts
    texts: List[str] = []
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

    items = judge_texts(client, args.model, args.target_lang, texts)
    avg = sum(x.get("score", 0.0) for x in items) / max(1, len(items))
    payload = {"avg_score": avg, "items": items}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps({"avg_score": avg}, indent=2))


if __name__ == "__main__":
    main()
