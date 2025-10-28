#!/usr/bin/env python3
import json, os, pathlib

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

def write(name, lines):
    p = DATA / name
    with p.open("w", encoding="utf-8") as f:
        for t in lines:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print("wrote", p)

# extend or overwrite as you like
if __name__ == "__main__":
    print("Datasets already included in data/. Use this script to add more if desired.")
