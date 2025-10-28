#!/usr/bin/env python3
"""
Quick dataset sanity check for MMIE JSONL inputs.

Usage:
  python scripts/check_datasets.py --paths data/forget_hi.jsonl data/retain_en.jsonl data/mixed.jsonl
"""
import argparse, json, os

def sniff(path: str, n: int = 5):
    ok = True; count = 0; bad = 0
    if not os.path.exists(path):
        return {'exists': False, 'count': 0, 'sample': []}
    sample = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            count += 1
            try:
                obj = json.loads(line)
                txt = obj['text'] if isinstance(obj, dict) and 'text' in obj else str(obj)
                if len(sample) < n:
                    sample.append(txt[:120])
            except Exception:
                bad += 1
    return {'exists': True, 'count': count, 'bad': bad, 'sample': sample}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', nargs='+', required=True)
    args = ap.parse_args()
    out = {p: sniff(p) for p in args.paths}
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()

