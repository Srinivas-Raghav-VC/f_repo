#!/usr/bin/env python3
# moved to tools/ (repo reorg 2025-10)
"""
Analysis pack: diagnose entanglement and steering difficulty.
Computes Jaccard@k overlap of SAE features (F vs R), principal angles between subspaces,
and gradient alignment between forget vs retain for adapter params.
Outputs a JSON report and brief diagnosis.
"""
import json, argparse, numpy as np, torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from mmie import TopKSAE, mean_activations, read_jsonl


def topk_support(z_abs_mean: np.ndarray, k: int) -> set:
    idx = np.argsort(z_abs_mean)[::-1][:k]
    return set(map(int, idx))


def principal_angles(U: np.ndarray, V: np.ndarray, d: int = 32) -> List[float]:
    """Return principal angles (in degrees) between column spaces of U and V."""
    # thin SVD for subspace bases
    def basis(X):
        Ux, _, _ = np.linalg.svd(X, full_matrices=False)
        return Ux[:, : min(d, Ux.shape[1])]
    Bu, Bv = basis(U), basis(V)
    M = Bu.T @ Bv
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    ang = np.degrees(np.arccos(s))
    return ang.tolist()


def jaccard_at_k(model, tok, sae: TopKSAE, texts_a: List[str], texts_b: List[str], layer: int, device: str, k: int = 64, cap_each: int = 256) -> float:
    def collect(texts):
        vals = []
        total = 0
        for batch in [texts[i : i + 32] for i in range(0, min(len(texts), cap_each), 32)]:
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model(**enc, output_hidden_states=True)
            H = out.hidden_states[layer + 1].mean(dim=1).to(torch.float32)
            Z = sae.E(H)
            vals.append(Z.abs().mean(dim=0).detach().cpu().numpy())
            total += len(batch)
            if total >= cap_each:
                break
        return np.mean(vals, axis=0)

    a = collect(texts_a)
    b = collect(texts_b)
    Sa = topk_support(a, k)
    Sb = topk_support(b, k)
    return float(len(Sa & Sb) / max(1, len(Sa | Sb)))


def grad_alignment(model, tok, texts: List[str], device: str, params_filter) -> torch.Tensor:
    """Compute a single-step gradient direction for given texts and return grad vector (flattened)."""
    model.zero_grad(set_to_none=True)
    batch = texts[:8]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    out = model(**enc, labels=enc["input_ids"])  # NLL
    out.loss.backward()
    vec = []
    for n, p in model.named_parameters():
        if p.grad is not None and params_filter(n, p):
            vec.append(p.grad.detach().flatten().cpu())
    if not vec:
        return torch.zeros(1)
    return torch.cat(vec)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    an = a / (a.norm() + 1e-8)
    bn = b / (b.norm() + 1e-8)
    return float((an * bn).sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--forget", required=True)
    ap.add_argument("--retain", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=[])
    ap.add_argument("--sae_k", type=int, default=32)
    ap.add_argument("--sae_expansion", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="analysis_report.json")
    args = ap.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    forget = read_jsonl(args.forget)
    retain = read_jsonl(args.retain)

    # Default layers if not provided
    n_layers = len(mdl.model.layers)
    Ls = args.layers or [max(0, n_layers // 4), max(0, n_layers // 2), max(0, (3 * n_layers) // 4)]

    results: Dict = {"layers": Ls, "jaccard@64": {}, "principal_angles": {}, "grad_alignment_cos": None}

    # Jaccard and angles per layer (using SAE encoders initialized fresh; prefer saved SAEs if available)
    for li in Ls:
        sae = TopKSAE(d=mdl.config.hidden_size, k=args.sae_k, expansion=args.sae_expansion).to(device)
        # If a saved SAE exists, load it for accuracy
        path = f"sae_layer{li}.pt"
        try:
            sd = torch.load(path, map_location=device)
            sae.load_state_dict(sd, strict=False)
            sae.eval()
        except Exception:
            pass
        j64 = jaccard_at_k(mdl, tok, sae, forget, retain, layer=li, device=device, k=64)
        # principal angles between mean activations for F and R
        Xf = mean_activations(mdl, tok, forget[:256], [li], device)[li]
        Xr = mean_activations(mdl, tok, retain[:256], [li], device)[li]
        ang = principal_angles(Xf, Xr, d=32)
        results["jaccard@64"][str(li)] = j64
        results["principal_angles"][str(li)] = ang

    # Gradient alignment on adapter-like params (q_proj/v_proj) as a proxy
    def filt(name, p):
        return any(k in name for k in ["q_proj", "v_proj"]) and p.requires_grad

    gf = grad_alignment(mdl, tok, forget[:64], device, filt)
    gr = grad_alignment(mdl, tok, retain[:64], device, filt)
    results["grad_alignment_cos"] = cosine_similarity(gf, gr)

    # Simple diagnosis
    jmean = float(np.mean(list(results["jaccard@64"].values()))) if results["jaccard@64"] else 0.0
    ang_mean = float(np.mean([np.mean(v) for v in results["principal_angles"].values()])) if results["principal_angles"] else 90.0
    cos = results["grad_alignment_cos"] or 0.0
    if jmean > 0.5 and ang_mean < 30 and cos > 0.2:
        diag = "hard"
    elif jmean > 0.3 and ang_mean < 45 and cos > 0.1:
        diag = "borderline"
    else:
        diag = "easy"
    results["diagnosis"] = diag

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"diagnosis": diag, "grad_cos": cos}, indent=2))


if __name__ == "__main__":
    main()
