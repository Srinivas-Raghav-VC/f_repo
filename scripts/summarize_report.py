#!/usr/bin/env python3
import json, argparse, sys

GATE_ORDER = [
    "G1_ES50", "G1S_ES50_sem", "G2_PPL10",
    "G3_MIX30", "G3S_MIX30_sem", "G4_NoRedistrib",
    "G5_NoXLeak", "G6_MIA0",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('report', help='eval_report.json path')
    args = ap.parse_args()
    with open(args.report, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gates = data.get('gates', {})
    decisions = data.get('decision', {})
    for arm in sorted(gates.keys()):
        print(f"== {arm} ==")
        g = gates[arm]
        for k in GATE_ORDER:
            if k in g:
                print(f"  {k:18}: {'PASS' if g[k] else 'FAIL'}")
        print(f"  decision         : {decisions.get(arm, 'N/A')}")
        print()

if __name__ == '__main__':
    main()

