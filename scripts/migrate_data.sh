#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
shopt -s nullglob
for f in *.jsonl; do
  echo "moving $f -> data/$f"
  mv -f "$f" "data/$f"
done
echo "[ok] moved JSONL files into data/" 

