#!/usr/bin/env python3
"""
Builds semantic control sets:
- Hindi romanized (forget_hi_roman.jsonl)
- English-as-Devanagari (retain_en_deva.jsonl) via a simple surrogate mapping
- Devanagari gibberish (deva_gibberish.jsonl)

Usage:
  python scripts/build_controls.py --forget forget_hi.jsonl --retain retain_en.jsonl --out_dir data_controls
"""
import json, argparse, os, pathlib, random
from typing import List

try:
    from transliteration_utils import batch_devanagari_to_latin
    HAS_ROM = True
except Exception:
    HAS_ROM = False

def read_jsonl(p:str, lim:int=None)->List[str]:
    out=[]
    with open(p,'r',encoding='utf-8') as f:
        for i,l in enumerate(f):
            if lim is not None and i>=lim: break
            l=l.strip()
            if not l: continue
            try:
                obj=json.loads(l)
                if isinstance(obj,dict) and 'text' in obj:
                    out.append(str(obj['text']))
                else:
                    out.append(str(obj))
            except Exception:
                out.append(l)
    return out

def write_jsonl(path: str, texts: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for t in texts:
            f.write(json.dumps({'text': t}, ensure_ascii=False) + "\n")

def romanize_hi(texts: List[str]) -> List[str]:
    if HAS_ROM:
        try:
            return batch_devanagari_to_latin(texts)
        except Exception:
            return texts
    return texts

DEVA_MAP = {
    # naive surrogate; not linguistically correct—sufficient for script-only controls
    **{c: 'क' for c in 'aeiou'},
    **{c: 'ख' for c in 'bc'},
    **{c: 'ग' for c in 'dg'},
    **{c: 'च' for c in 'f'},
    **{c: 'ज' for c in 'jh'},
    **{c: 'ट' for c in 'kq'},
    **{c: 'त' for c in 'l'},
    **{c: 'प' for c in 'm'},
    **{c: 'न' for c in 'n'},
    **{c: 'स' for c in 's'},
    **{c: 'र' for c in 'r'},
    **{c: 'व' for c in 'v'},
    **{c: 'य' for c in 'y'},
    **{c: 'ह' for c in 't'},
}

def english_to_deva(texts: List[str]) -> List[str]:
    out=[]
    for t in texts:
        out.append(''.join(DEVA_MAP.get(ch.lower(), ' ') if ch.isalpha() else ch for ch in t))
    return out

def deva_gibberish(texts: List[str], seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    letters = [chr(c) for c in range(0x0915, 0x0939+1)]
    matras = [chr(c) for c in [0x093E,0x093F,0x0940,0x0941,0x0942,0x0947,0x0948,0x094B,0x094C]]
    out = []
    for t in texts:
        n = max(8, min(200, len(t)))
        s = []
        for i in range(n):
            s.append(rng.choice(letters if i % 3 != 2 else matras))
        out.append(''.join(s))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--forget', required=True)
    ap.add_argument('--retain', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    forget = read_jsonl(args.forget)
    retain = read_jsonl(args.retain)

    hi_rom = romanize_hi(forget)
    en_deva = english_to_deva(retain)
    deva_gib = deva_gibberish(retain)

    write_jsonl(os.path.join(args.out_dir, 'forget_hi_roman.jsonl'), hi_rom)
    write_jsonl(os.path.join(args.out_dir, 'retain_en_deva.jsonl'), en_deva)
    write_jsonl(os.path.join(args.out_dir, 'deva_gibberish.jsonl'), deva_gib)
    print('wrote controls to', args.out_dir)

if __name__ == '__main__':
    main()

