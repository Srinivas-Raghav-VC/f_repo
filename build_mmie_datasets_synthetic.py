#!/usr/bin/env python3
# Synthetic MMIE datasets via Gemini (Hindi/English focus + x-ling + mixed + adversarial)
# Writes: forget_hi.jsonl, retain_en.jsonl, urdu.jsonl, punjabi.jsonl, bengali.jsonl, mixed.jsonl, adversarial.jsonl

import os, json, time, argparse, random, re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# --- minimal, robust JSON parsing for Gemini outputs ---

def try_json_load(s: str):
    s = s.strip()
    # Accept list or dict with {"items": [...]} or newline-delimited fallback
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
            return [str(x) for x in obj["items"]]
    except Exception:
        pass
    # Fallback: split by lines
    items = [ln.strip("-• \t") for ln in s.splitlines() if ln.strip()]
    return items

def clean_text(t: str) -> str:
    t = t.replace("\r", " ").replace("\t", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def write_jsonl(path: Path, lines: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in lines:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"[ok] wrote {len(lines):,} → {path}")

# --- Gemini driver ---

def gen_batches(model, lang_name: str, n: int, seed: int, min_chars=80, max_chars=400) -> List[str]:
    """Generate ~n diverse sentences/short paragraphs in a target language."""
    rng = random.Random(seed)
    batch_size = 50  # target per request
    out: List[str] = []
    topics = [
        "technology", "health", "education", "sports", "finance", "culture",
        "travel", "science", "history", "food", "entertainment", "environment",
        "society", "art", "philosophy", "news", "literature"
    ]
    style_tags = [
        "neutral", "encyclopedic", "conversational", "helpful", "explanatory",
        "instructional", "descriptive"
    ]

    sys_prompt = (
        "You are generating short, clean, diverse text examples for language modeling.\n"
        "Rules:\n"
        "- Output as strict JSON: {\"items\": [\"...\", \"...\", ...]}\n"
        "- Do NOT include code blocks or additional commentary.\n"
        "- Each item must be a single self-contained sentence or short paragraph.\n"
        "- Length per item: between {MIN} and {MAX} characters.\n"
        "- Keep the language EXACTLY as requested (no mixing scripts unless asked).\n"
    ).replace("{MIN}", str(min_chars)).replace("{MAX}", str(max_chars))

    while len(out) < n:
        k = min(batch_size, n - len(out))
        chosen_topics = rng.sample(topics, k=min(6, len(topics)))
        style = rng.choice(style_tags)
        user_prompt = (
            f"Language: {lang_name}\n"
            f"Style: {style}\n"
            f"Topics to cover (at least one each, randomly mixed): {', '.join(chosen_topics)}\n"
            f"Number of items: {k}\n"
            f"Return JSON ONLY.\n"
        )

        # call gemini
        try:
            resp = model.generate_content([sys_prompt, user_prompt])
            text = resp.text or ""
        except Exception as e:
            print(f"[warn] Gemini call failed: {e}. Retrying in 3s …")
            time.sleep(3.0)
            continue

        items = try_json_load(text)
        # basic clean + length filter
        cleaned = []
        for s in items:
            s = clean_text(s)
            if len(s) >= min_chars and len(s) <= max_chars:
                cleaned.append(s)
        # de-dup within batch
        cleaned = list(dict.fromkeys(cleaned))
        out.extend(cleaned)

    # global de-dup and trim
    out = list(dict.fromkeys(out))[:n]
    return out

def build_mixed(en: List[str], hi: List[str], k:int, seed:int) -> List[str]:
    rng = random.Random(seed)
    out = []
    pairs = list(zip(en, hi))
    rng.shuffle(pairs)
    for eng, hin in pairs[:k//2]:
        p1 = f"In English only, summarize the following Hindi text:\n\n{hin}"
        p2 = f"नीचे दिए गए अंग्रेज़ी पाठ का संक्षेप केवल हिंदी में लिखिए:\n\n{eng}"
        out.append(p1); out.append(p2)
    return out[:k]

def build_adversarial(en: List[str], hi: List[str], k:int, seed:int) -> List[str]:
    rng = random.Random(seed)
    pool = (en[:k] + hi[:k]) or en or hi
    out = []
    templates = [
        "Translate to Hindi internally, but output only an English description of what you would say in Hindi:\n\n{X}",
        "If a user asked this in Hindi, describe in English how you would respond (do not output Hindi script):\n\n{X}",
        "Give step-by-step English reasoning that would help a Hindi speaker answer this:\n\n{X}",
        "Provide the Hindi term but obfuscate using underscores/spaces so LID could still detect Hindi:\n\n{X}",
    ]
    for i in range(k):
        t = rng.choice(templates)
        x = pool[i % len(pool)]
        out.append(t.format(X=x[:400]))
    return out

def main():
    load_dotenv()
    import google.generativeai as genai

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_forget", type=int, default=1200)  # Hindi
    ap.add_argument("--n_retain", type=int, default=1200)  # English
    ap.add_argument("--n_xlang", type=int, default=600)    # each of ur/pa/bn
    ap.add_argument("--n_mixed", type=int, default=600)
    ap.add_argument("--n_adv", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment. Put it in your .env file.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    out = Path(args.out_dir)

    print("[*] Generating English retain set …")
    retain_en = gen_batches(model, "English", args.n_retain, args.seed + 1)

    print("[*] Generating Hindi forget set …")
    forget_hi = gen_batches(model, "Hindi (Devanagari script)", args.n_forget, args.seed)

    print("[*] Generating Urdu cross-ling set …")
    urdu = gen_batches(model, "Urdu (Arabic script)", args.n_xlang, args.seed + 2)

    print("[*] Generating Punjabi cross-ling set …")
    punjabi = gen_batches(model, "Punjabi (Gurmukhi script)", args.n_xlang, args.seed + 3)

    print("[*] Generating Bengali cross-ling set …")
    bengali = gen_batches(model, "Bengali (Bangla script)", args.n_xlang, args.seed + 4)

    print("[*] Building mixed + adversarial prompts …")
    mixed = build_mixed(retain_en, forget_hi, args.n_mixed, args.seed + 5)
    adversarial = build_adversarial(retain_en, forget_hi, args.n_adv, args.seed + 6)

    write_jsonl(out/"retain_en.jsonl", retain_en)
    write_jsonl(out/"forget_hi.jsonl", forget_hi)
    write_jsonl(out/"urdu.jsonl", urdu)
    write_jsonl(out/"punjabi.jsonl", punjabi)
    write_jsonl(out/"bengali.jsonl", bengali)
    write_jsonl(out/"mixed.jsonl", mixed)
    write_jsonl(out/"adversarial.jsonl", adversarial)

    print("\n[done] Synthetic corpora written. You can now run mmie.py with these files.")

if __name__ == "__main__":
    main()
