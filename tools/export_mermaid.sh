#!/usr/bin/env bash
set -euo pipefail

# Export Mermaid diagrams in docs/figs/ to PNG/SVG.
# Requires one of:
#  1) npx @mermaid-js/mermaid-cli  (Node.js)
#  2) Docker (minlag/mermaid-cli)

INPUT=${1:-docs/figs/pipeline.mmd}
OUTPNG=${2:-docs/figs/pipeline.png}
OUTSVG=${3:-docs/figs/pipeline.svg}

if command -v npx >/dev/null 2>&1; then
  npx -y @mermaid-js/mermaid-cli@10.9.1 -i "$INPUT" -o "$OUTPNG" -b white
  npx -y @mermaid-js/mermaid-cli@10.9.1 -i "$INPUT" -o "$OUTSVG" -b transparent
  echo "[mmdc] wrote $OUTPNG and $OUTSVG"
  exit 0
fi

if command -v docker >/dev/null 2>&1; then
  docker run --rm -v "$(pwd)":"/data" minlag/mermaid-cli:10.9.1 mmdc -i "/data/$INPUT" -o "/data/$OUTPNG" -b white
  docker run --rm -v "$(pwd)":"/data" minlag/mermaid-cli:10.9.1 mmdc -i "/data/$INPUT" -o "/data/$OUTSVG" -b transparent
  echo "[docker mmdc] wrote $OUTPNG and $OUTSVG"
  exit 0
fi

echo "No exporter found (need Node+npx or Docker). Try: npm i -g @mermaid-js/mermaid-cli" >&2
exit 1

