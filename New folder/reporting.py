"""
Reporting Module
================
Generate reports and save results.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(results: Dict, path: Path):
    """Save results to JSON file."""
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)


def generate_final_report(all_results: Dict, config, output_dir: Path):
    """Generate comprehensive final report."""
    
    report_lines = []
    
    report_lines.append("=" * 70)
    report_lines.append("MMIE RESEARCH: FINAL REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    # ==================== SEMANTIC SUBSPACE ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("1. SEMANTIC SUBSPACE ANALYSIS")
    report_lines.append("=" * 70)
    
    if "semantic" in all_results:
        sem = all_results["semantic"]
        interp = sem.get("interpretation", {})
        report_lines.append(f"  Best semantic layer: {interp.get('best_semantic_layer', 'N/A')}")
        report_lines.append(f"  Gap (parallel vs different): {interp.get('best_gap', 0):.3f}")
        report_lines.append(f"  Semantic subspace exists: {interp.get('semantic_subspace_exists', False)}")
        report_lines.append(f"  Language always separable: {interp.get('language_always_separable', False)}")
    else:
        report_lines.append("  [Not run]")
    
    # ==================== STEERING ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("2. STEERING GRID SEARCH")
    report_lines.append("=" * 70)
    
    if "steering" in all_results:
        steer = all_results["steering"]
        best = steer.get("best_combinations", [{}])[0]
        report_lines.append(f"  Best layer: {best.get('layer', 'N/A')}")
        report_lines.append(f"  Best coefficient: {best.get('coeff', 'N/A')}")
        report_lines.append(f"  Success rate: {best.get('success_rate', 0):.1%}")
    else:
        report_lines.append("  [Not run]")
    
    # ==================== CAUSALITY ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("3. CAUSALITY TEST")
    report_lines.append("=" * 70)
    
    if "causality" in all_results:
        caus = all_results["causality"]
        summ = caus.get("summary", {})
        report_lines.append(f"  Average change from patching: {summ.get('avg_change', 0):+.3f}")
        report_lines.append(f"  Causal pairs: {summ.get('causal_pairs', 0)}/{summ.get('total_pairs', 0)}")
        report_lines.append(f"  Conclusion: {summ.get('conclusion', 'N/A')}")
    else:
        report_lines.append("  [Not run]")
    
    # ==================== COHERENCE ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("4. COHERENCE TEST (Goldilocks Zone)")
    report_lines.append("=" * 70)
    
    if "coherence" in all_results:
        coh = all_results["coherence"]
        report_lines.append(f"  Goldilocks coefficient: {coh.get('goldilocks_zone', 'N/A')}")
        
        summ = coh.get("coeff_summary", {})
        for coeff, data in sorted(summ.items()):
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0
            report_lines.append(f"    coeff={coeff}: {acc:.0%} correct")
    else:
        report_lines.append("  [Not run]")
    
    # ==================== CROSS-LANGUAGE ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("5. CROSS-LANGUAGE LEAK TEST")
    report_lines.append("=" * 70)
    
    if "crosslang" in all_results:
        cross = all_results["crosslang"]
        langs = cross.get("languages", {})
        
        for lang, data in langs.items():
            report_lines.append(f"  {lang.upper()}:")
            report_lines.append(f"    Baseline: {data.get('baseline_mean', 0):.2f}")
            report_lines.append(f"    Steered:  {data.get('steered_mean', 0):.2f}")
            report_lines.append(f"    Change:   {data.get('change', 0):+.2f}")
        
        interp = cross.get("interpretation", {})
        report_lines.append(f"\n  Leaking languages: {interp.get('leaking_languages', [])}")
        report_lines.append(f"  Clean separation: {interp.get('clean_separation', False)}")
    else:
        report_lines.append("  [Not run]")
    
    # ==================== SAE ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("6. SAE FEATURE ANALYSIS")
    report_lines.append("=" * 70)
    
    if "sae" in all_results:
        sae = all_results["sae"]
        metrics = sae.get("sae_metrics", {})
        interp = sae.get("interpretation", {})
        
        report_lines.append(f"  SAE val loss: {metrics.get('val_loss', 0):.4f}")
        report_lines.append(f"  SAE L0: {metrics.get('l0', 0):.1f}")
        report_lines.append(f"  Highly specific features: {interp.get('num_highly_specific', 0)}")
        report_lines.append(f"  Has language features: {interp.get('has_language_features', False)}")
    else:
        report_lines.append("  [Not run]")
    
    # ==================== ADVERSARIAL ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("7. ADVERSARIAL ROBUSTNESS")
    report_lines.append("=" * 70)
    
    if "adversarial" in all_results:
        adv = all_results["adversarial"]
        overall = adv.get("overall", {})
        
        report_lines.append(f"  Extraction rate: {overall.get('extraction_rate', 0):.1%}")
        report_lines.append(f"  Vulnerability: {overall.get('vulnerability', 'N/A')}")
        
        type_summ = adv.get("type_summary", {})
        for adv_type, data in type_summ.items():
            rate = data.get("rate", 0)
            report_lines.append(f"    {adv_type}: {rate:.1%}")
    else:
        report_lines.append("  [Not run]")
    
    # ==================== CONCLUSIONS ====================
    report_lines.append("\n" + "=" * 70)
    report_lines.append("CONCLUSIONS")
    report_lines.append("=" * 70)
    
    conclusions = []
    
    if "semantic" in all_results:
        if all_results["semantic"].get("interpretation", {}).get("semantic_subspace_exists"):
            conclusions.append("✓ Semantic subspace EXISTS for Hindi-English")
        else:
            conclusions.append("✗ Semantic subspace NOT confirmed")
    
    if "causality" in all_results:
        if all_results["causality"].get("summary", {}).get("conclusion") == "CAUSAL":
            conclusions.append("✓ Language control is CAUSAL (not just correlation)")
        else:
            conclusions.append("✗ Causality NOT confirmed")
    
    if "coherence" in all_results:
        goldilocks = all_results["coherence"].get("goldilocks_zone")
        if goldilocks:
            conclusions.append(f"✓ Goldilocks zone found at coeff={goldilocks}")
        else:
            conclusions.append("✗ No Goldilocks zone found")
    
    if "crosslang" in all_results:
        if all_results["crosslang"].get("interpretation", {}).get("clean_separation"):
            conclusions.append("✓ Clean separation (no cross-language leaks)")
        else:
            leaks = all_results["crosslang"].get("interpretation", {}).get("leaking_languages", [])
            conclusions.append(f"⚠ Cross-language leaks to: {leaks}")
    
    for c in conclusions:
        report_lines.append(f"  {c}")
    
    report_lines.append("\n" + "=" * 70)
    
    # Save report
    report_text = "\n".join(report_lines)
    
    with open(output_dir / "final_report.txt", 'w') as f:
        f.write(report_text)
    
    print(report_text)
    
    # Also save as JSON
    save_results({
        "timestamp": datetime.now().isoformat(),
        "conclusions": conclusions,
        "all_results": all_results,
    }, output_dir / "final_report.json")
