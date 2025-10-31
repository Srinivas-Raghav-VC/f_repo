#!/usr/bin/env python3
import json, argparse, sys

GATE_ORDER = [
    "G1_ES50", "G1S_ES50_sem", "G2_PPL10",
    "G3_MIX30", "G3S_MIX30_sem", "G4_NoRedistrib",
    "G5_NoXLeak", "G6_MIA0", "G7_AdvES",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('report', help='eval_report.json path')
    args = ap.parse_args()
    with open(args.report, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gates = data.get('gates', {})
    decisions = data.get('decision', {})
    # Optional: print comprehension proxies if available
    summ = data.get('summary', {})
    for arm in sorted([k for k in summ.keys() if k != 'base']):
        comp1 = summ[arm].get('comp_hi2en_en_ratio_mean')
        comp2 = summ[arm].get('comp_langid_acc_mean')
        if comp1 is not None or comp2 is not None:
            print(f"-- {arm} comprehension --")
            if comp1 is not None:
                print(f"  HI→EN outputs in English (ratio): {comp1:.3f}")
            if comp2 is not None:
                print(f"  LID(Yes/No) accuracy on HI:       {comp2:.3f}")
            print()
    for arm in sorted(gates.keys()):
        print(f"== {arm} ==")
        g = gates[arm]
        for k in GATE_ORDER:
            if k in g:
                print(f"  {k:18}: {'PASS' if g[k] else 'FAIL'}")
        print(f"  decision         : {decisions.get(arm, 'N/A')}")
        # Optional: corrected p-values (FDR)
        sm = data.get('summary', {}).get(arm, {})
        pmap = sm.get('gate_pvalues_fdr_corrected')
        if isinstance(pmap, dict):
            print("  pvals_FDR        :", {k: round(v,4) for k,v in pmap.items()})
        # U-LiRA if present
        ul = sm.get('ulira')
        if isinstance(ul, dict):
            print(f"  U-LiRA+          : AUC={ul.get('AUC_mean','?')}, ACC={ul.get('ACC_mean','?')}")
        print()

if __name__ == '__main__':
    main()
