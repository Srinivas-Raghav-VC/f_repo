#!/usr/bin/env python3
"""
Generate a standard set of plots from eval_report.json.
Saves under plots/<model>__<stem>/ or plots/<stem>/ when model is absent.

Plots (saved as PNGs):
  - es_forget_bar.png (script-aware)
  - es_forget_semantic_bar.png (if present)
  - ppl_retain_bar.png
  - es_mixed_bar.png
  - crossling_es_bar.png (per-language deltas vs base)
  - probes_auc_bar.png
  - mia_bar.png (AUC/ACC)
  - layer_scores_bar.png (combo scores)
"""
import os, json, argparse, re
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def _folder_for(report_path, model):
    stem = Path(report_path).stem
    if model:
        sm = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        return Path("plots")/f"{sm}__{stem}"
    return Path("plots")/stem

def _bar(ax, labels, vals, title, ylabel, yerr=None):
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=yerr, color=['#4C78A8' if i==0 else '#F58518' for i in range(len(labels))], alpha=0.9, capsize=3)
    ax.set_xticks(x, labels, rotation=20, ha='right')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('report')
    args = ap.parse_args()
    with open(args.report, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model = data.get('model', None)
    outdir = _folder_for(args.report, model)
    outdir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        print('[plots] matplotlib not installed; skipping plot generation.')
        return

    # Base
    base = data.get('summary', {}).get('base', {})
    base_es = base.get('es_forget', np.nan)
    base_mix = base.get('es_mixed', np.nan)
    base_es_sem = base.get('es_semantic', np.nan)
    base_ppl = base.get('ppl_retain', np.nan)

    # Arms
    arms = [k for k in data.get('summary', {}).keys() if k != 'base']
    def take(path, default=np.nan):
        cur = data.get('summary', {})
        for p in path:
            if p not in cur:
                return default
            cur = cur[p]
        return cur

    # ES forget (script-aware)
    labels = ['base'] + arms
    vals = [base_es] + [take([a, 'es_forget_mean']) for a in arms]
    errs = [0.0] + [abs(take([a,'es_forget_ci',0]) - take([a,'es_forget_mean'])) for a in arms]
    fig, ax = plt.subplots(figsize=(7,4))
    _bar(ax, labels, vals, 'ES (forget, script-aware)', 'ES', yerr=errs)
    fig.tight_layout(); fig.savefig(outdir/'es_forget_bar.png', dpi=160); plt.close(fig)

    # ES forget semantic (if present)
    if not np.isnan(base_es_sem):
        labels = ['base'] + [a for a in arms if not np.isnan(take([a,'es_forget_semantic_mean']))]
        vals = [base_es_sem] + [take([a,'es_forget_semantic_mean']) for a in labels[1:]]
        fig, ax = plt.subplots(figsize=(7,4))
        _bar(ax, labels, vals, 'ES (forget, script-blind/semantic)', 'ES')
        fig.tight_layout(); fig.savefig(outdir/'es_forget_semantic_bar.png', dpi=160); plt.close(fig)

    # PPL retain
    vals = [base_ppl] + [take([a,'ppl_retain_mean']) for a in arms]
    fig, ax = plt.subplots(figsize=(7,4))
    _bar(ax, labels=['base']+arms, vals=vals, title='Perplexity (retain)', ylabel='PPL')
    fig.tight_layout(); fig.savefig(outdir/'ppl_retain_bar.png', dpi=160); plt.close(fig)

    # ES mixed
    vals = [base_mix] + [take([a,'es_mixed_mean']) for a in arms]
    fig, ax = plt.subplots(figsize=(7,4))
    _bar(ax, labels=['base']+arms, vals=vals, title='ES (mixed prompts)', ylabel='ES')
    fig.tight_layout(); fig.savefig(outdir/'es_mixed_bar.png', dpi=160); plt.close(fig)

    # Cross-ling ES deltas (arms - base), averaged over seeds
    # Compute arm means from raw seeds if needed
    cross = data.get('arms', {})
    langs = []
    deltas = {}
    base_x = base.get('crossling_es', {})
    for a, arm in cross.items():
        acc = {}
        for s in arm.get('seeds', []):
            for ln, v in s.get('crosslingual_es', {}).items():
                acc.setdefault(ln, []).append(v)
        for ln, vs in acc.items():
            langs.append(ln)
            deltas.setdefault(a, {})[ln] = float(np.mean(vs) - base_x.get(ln, np.nan))
    langs = sorted(set(langs))
    if langs and deltas:
        fig, ax = plt.subplots(figsize=(max(6, 1.8*len(langs)), 4))
        for i, a in enumerate(sorted(deltas.keys())):
            vals = [deltas[a].get(ln, np.nan) for ln in langs]
            x = np.arange(len(langs)) + i*0.35
            ax.bar(x, vals, width=0.35, label=a)
        ax.set_xticks(np.arange(len(langs))+0.35/2, langs, rotation=20, ha='right')
        ax.set_title('Cross-lingual ES deltas (arm - base)')
        ax.set_ylabel('ΔES')
        ax.grid(axis='y', alpha=0.2)
        ax.legend()
        fig.tight_layout(); fig.savefig(outdir/'crossling_es_bar.png', dpi=160); plt.close(fig)

    # Probes AUC mean per arm (other layers)
    aucs = {}
    for a, arm in data.get('arms', {}).items():
        vals = []
        for s in arm.get('seeds', []):
            for _, pr in s.get('probes_other_layers', {}).items():
                vals.append(pr.get('auc', np.nan))
        if vals:
            aucs[a] = float(np.nanmean(vals))
    if aucs:
        labels = sorted(aucs.keys())
        fig, ax = plt.subplots(figsize=(7,4))
        _bar(ax, labels, [aucs[l] for l in labels], 'Probes AUC (other layers)', 'AUC')
        fig.tight_layout(); fig.savefig(outdir/'probes_auc_bar.png', dpi=160); plt.close(fig)

    # MIA
    labs = []; av=[]; ac=[]
    for a in arms:
        m = take([a,'mia'])
        if isinstance(m, dict):
            labs.append(a); av.append(m.get('AUC_mean', np.nan)); ac.append(m.get('ACC_mean', np.nan))
    if labs:
        fig, ax = plt.subplots(figsize=(7,4))
        x = np.arange(len(labs))
        ax.bar(x-0.17, av, width=0.34, label='AUC')
        ax.bar(x+0.17, ac, width=0.34, label='ACC')
        ax.set_xticks(x, labs, rotation=20, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('MIA (closer to 0.5 is better)')
        ax.legend(); ax.grid(axis='y', alpha=0.2)
        fig.tight_layout(); fig.savefig(outdir/'mia_bar.png', dpi=160); plt.close(fig)

    # Layer score bars
    ls = data.get('layer_scores', {})
    if ls:
        pairs = sorted([(int(k), v.get('combo', np.nan)) for k,v in ls.items()], key=lambda t: t[0])
        lidx = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        fig, ax = plt.subplots(figsize=(max(7, 0.3*len(lidx)), 4))
        ax.bar(lidx, vals, color='#59A14F')
        ax.set_xlabel('Layer index'); ax.set_ylabel('Combo score')
        ax.set_title('Layer selection scores')
        ax.grid(axis='y', alpha=0.2)
        fig.tight_layout(); fig.savefig(outdir/'layer_scores_bar.png', dpi=160); plt.close(fig)

    print(f"[plots] wrote plots to {outdir}")

if __name__ == '__main__':
    main()

