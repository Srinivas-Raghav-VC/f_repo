#!/usr/bin/env python3
# MMIE end-to-end: LoRA vs ReFT+SAE; ES via LID ensemble; PPL; Mixed; Probes; MIA; X-ling leakage
# Checkpointing: LoRA -> lora_adapters.pt ; ReFT -> reft_adapters.pt ; SAE -> sae_layer{L}.pt

import os, json, argparse, math, random, itertools
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FUT_TIMEOUT

# Windows-specific: avoid memory-mapped safetensors which can trigger
# OSError 1455 ("paging file is too small") on some setups.
if os.name == 'nt':
    os.environ.setdefault('SAFETENSORS_FAST', '0')

from lid_ensemble import LIDEnsemble, LIDConfig
try:
    from backends.sae_lens_loader import load_sae_from_dir as _load_sae_lens
    _HAS_SAELENS_LOADER = True
except Exception:
    _HAS_SAELENS_LOADER = False

# Optional transliteration helpers for script-blind evaluation
try:
    from transliteration_utils import batch_devanagari_to_latin
    _HAS_DEVANAGARI_TO_LATIN = True
except Exception:
    _HAS_DEVANAGARI_TO_LATIN = False

# -------------------- utils --------------------

def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

PREFERRED_TEXT_KEYS=("prompt","text","completion","response","output","instruction","input","question","answer")

def read_jsonl(p:str, lim:int=None)->List[str]:
    out=[]
    with open(p,'r',encoding='utf-8') as f:
        for i,l in enumerate(f):
            if lim is not None and i>=lim: break
            l=l.strip()
            if not l: continue
            try:
                obj=json.loads(l)
            except json.JSONDecodeError:
                out.append(l)
                continue
            if isinstance(obj,str):
                out.append(obj)
                continue
            if isinstance(obj,dict):
                chosen: Optional[str]=None
                for key in PREFERRED_TEXT_KEYS:
                    val=obj.get(key)
                    if isinstance(val,str) and val.strip():
                        chosen=val
                        break
                if chosen is None and isinstance(obj.get("messages"), list):
                    for message in reversed(obj["messages"]):
                        if isinstance(message,dict):
                            content=message.get("content")
                            if isinstance(content,str) and content.strip():
                                chosen=content
                                break
                if chosen is None:
                    chosen=json.dumps(obj, ensure_ascii=False)
                out.append(chosen); continue
            out.append(str(obj))
    return out

def chunked(xs, n):
    it = iter(xs)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk: break
        yield chunk

def bootstrap_ci(values:List[float], alpha=0.05, n_boot=2000, seed=0):
    if not values: return (float('nan'), (float('nan'), float('nan')))
    rng = np.random.RandomState(seed)
    x = np.array(values, dtype=np.float32)
    boots = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    boots.sort()
    lo = boots[int(alpha/2*n_boot)]
    hi = boots[int((1-alpha/2)*n_boot)-1]
    return float(np.mean(x)), (float(lo), float(hi))

# Backward-compatible alias (was misnamed as BCa; this is percentile bootstrap)
def bca_ci(values:List[float], alpha=0.05, n_boot=2000, seed=0):
    return bootstrap_ci(values, alpha=alpha, n_boot=n_boot, seed=seed)

# Reusable infinite iterators that do NOT cache content in memory
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def infinite_from_factory(factory):
    # factory must return a fresh iterable/generator when called
    while True:
        for x in factory():
            yield x

# Runtime device helpers
def _get_model_device(model):
    try:
        dev = next(model.parameters()).device
        # In rare cases with lazy/meta tensors, fall back to cpu
        if str(dev) == 'meta':
            return torch.device('cpu')
        return dev
    except StopIteration:
        return torch.device('cpu')

def _to_model_device(model, enc):
    dev = _get_model_device(model)
    if isinstance(enc, dict):
        return {k: (v.to(dev) if hasattr(v, 'to') else v) for k,v in enc.items()}
    return enc.to(dev) if hasattr(enc, 'to') else enc

# -------------------- cross-lingual selection --------------------

def center(K):
    n=K.shape[0]; H=np.eye(n, dtype=np.float32)-np.ones((n,n), dtype=np.float32)/n
    return H@K@H

def linear_cka_debiased(X, Y):
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    Kc = center(K); Lc = center(L)
    hsic = float((Kc * Lc).sum())
    var1 = math.sqrt(float((Kc * Kc).sum())); var2 = math.sqrt(float((Lc * Lc).sum()))
    return float(hsic / (var1 * var2 + 1e-8))

def procrustes_sim(X, Y):
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    C = Xc.T @ Yc
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    return float(s.sum() / (np.linalg.norm(Xc, 'fro') * np.linalg.norm(Yc, 'fro') + 1e-8))

def anc_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """Average Neuron-wise Correlation (ANC): per-dimension Pearson correlation, then mean of abs.
    X, Y: [N, D] with same D and aligned samples (approximate by random subsample if needed).
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    n = min(len(X), len(Y))
    if n < 100 or X.shape[1] != Y.shape[1]:
        return 0.0
    # random align without replacement
    idxX = np.random.choice(len(X), n, replace=False)
    idxY = np.random.choice(len(Y), n, replace=False)
    Xs = X[idxX]; Ys = Y[idxY]
    Xs = Xs - Xs.mean(0, keepdims=True)
    Ys = Ys - Ys.mean(0, keepdims=True)
    sx = Xs.std(0, keepdims=True) + 1e-6
    sy = Ys.std(0, keepdims=True) + 1e-6
    corr = (Xs * Ys).mean(0) / (sx * sy).ravel()
    return float(np.nanmean(np.abs(corr)))

def _sanitize_json(obj):
    """Recursively replace NaN/Inf with None for strict JSON output."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj

# -------------------- LLM judge (optional) --------------------
def _judge_avg_score(texts: List[str], target_lang: str = "Hindi", model_name: str = "gemini-2.5-flash", timeout: float = 15.0) -> Optional[float]:
    """Average judge score in [0,1] using the official google.genai client.
    Forces JSON MIME and schema for robust parsing; soft timeout fallback.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None
    try:
        client = genai.Client(api_key=key)
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["score"],
                        "additionalProperties": True
                    }
                }
            },
            "required": ["items"]
        }
        tmpl = (
            "You are a semantic language judge. For each item, decide if the content is in the target language,\n"
            "even if it is romanized or obfuscated (homoglyphs), and ignoring short named entities.\n"
            "Return strict JSON following the schema."
        )
        prompt = f"{tmpl}\nTarget language: {target_lang}\n"
        payload = {"items": texts[:100]}

        def _call():
            resp = client.models.generate_content(
                model=model_name,
                contents=[prompt, json.dumps(payload, ensure_ascii=False)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
            data = getattr(resp, "parsed", None)
            if not data or "items" not in data:
                try:
                    data = json.loads(getattr(resp, "text", "") or "{}")
                except Exception:
                    data = {}
            items = data.get("items", [])
            scores = [float(x.get("score", 0.0)) for x in items if isinstance(x, dict)]
            return float(np.mean(scores)) if scores else None

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            try:
                return fut.result(timeout=timeout)
            except _FUT_TIMEOUT:
                return None
    except Exception:
        return None

def _refine_layers_with_judge(model, tok, device, forget: List[str], retain: List[str], candidates: List[int],
                              base_hi_out: List[str], base_en_out: List[str], base_judge_hi: float, base_judge_en: float,
                              scale: float, judge_beta: float, judge_model: str, judge_timeout: float) -> Dict[int, float]:
    """Return per-layer judge deltas: (base_hi - scaled_hi) - beta * max(0, scaled_en - base_en)."""
    deltas = {}
    # We judge on outputs; re-use the same prompts as base_hi_out/base_en_out lengths
    cap_hi = len(base_hi_out)
    cap_en = len(base_en_out)
    for li in candidates:
        hook = ResidualScaleHook(model, [li], scale=scale)
        try:
            hi_gen = generate(model, tok, forget[:cap_hi], device)
            en_gen = generate(model, tok, retain[:cap_en], device)
            j_hi = _judge_avg_score(hi_gen, target_lang="Hindi", model_name=judge_model, timeout=judge_timeout)
            j_en = _judge_avg_score(en_gen, target_lang="Hindi", model_name=judge_model, timeout=judge_timeout)
            if j_hi is None or j_en is None:
                deltas[li] = 0.0
            else:
                drop_hi = max(0.0, base_judge_hi - j_hi)
                hurt_en = max(0.0, j_en - base_judge_en)
                deltas[li] = float(drop_hi - judge_beta * hurt_en)
        except Exception:
            deltas[li] = 0.0
        finally:
            hook.remove()
    return deltas

@torch.no_grad()
def collect_layer_means(model,tok,texts,layers,device,max_len=256,cap=1000,per_token=True,tokens_per_seq=16):
    model.eval()
    acts={li:[] for li in layers}
    for batch in tqdm(chunked(texts[:cap], 8), desc="acts"):
        enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=max_len)
        enc=_to_model_device(model, enc)
        out=model(**enc, output_hidden_states=True)
        for li in layers:
            H = out.hidden_states[li+1]  # [B,T,D]
            if per_token:
                B,T,D = H.shape
                tsel = min(tokens_per_seq, T)
                idx = torch.randint(0, T, (B, tsel), device=H.device)
                picked = H[torch.arange(B).unsqueeze(1), idx]  # [B,tsel,D]
                acts[li].append(picked.reshape(-1, D).detach().cpu().numpy().astype(np.float32))
            else:
                acts[li].append(H.detach().mean(dim=1).cpu().numpy().astype(np.float32))
    return {li:(np.concatenate(acts[li],0) if acts[li] else np.zeros((0,model.config.hidden_size), dtype=np.float32)) for li in layers}

def select_layers(model,tok,hi,en,n_layers,device,cap=2000,top_k=3, use_anc: bool = False, min_layer: int = 0,
                  select_mode: str = 'contrast', script_blind_selection: bool = True,
                  xlang_sets: Optional[List[Tuple[str, List[str]]]] = None,
                  max_len: int = 256):
    layers=[li for li in range(n_layers) if li >= max(0, int(min_layer))]
    # Optionally make selection script-blind by romanizing Devanagari prompts
    hi_sel = _romanize_texts(hi) if script_blind_selection else hi
    A=collect_layer_means(model,tok,hi_sel,layers,device,max_len=max_len,cap=cap,per_token=True,tokens_per_seq=16)
    B=collect_layer_means(model,tok,en,layers,device,max_len=max_len,cap=cap,per_token=True,tokens_per_seq=16)
    # Precompute semantic probe AUCs once (batched across layers)
    semantic_aucs_main: Dict[int, Dict[str, float]] = {}
    semantic_neigh_maps: List[Dict[int, Dict[str, float]]] = []
    if select_mode == 'semantic':
        probe_cap = int(min(cap, 80))
        try:
            semantic_aucs_main = probes_auc(model, tok, hi_sel[:probe_cap], en[:probe_cap], layers, device, max_len=max_len)
        except Exception:
            semantic_aucs_main = {}
        if xlang_sets:
            for _, xt in xlang_sets:
                try:
                    semantic_neigh_maps.append(probes_auc(model, tok, xt[:probe_cap], en[:probe_cap], layers, device, max_len=max_len))
                except Exception:
                    semantic_neigh_maps.append({})
    scores={}
    for li in layers:
        X=A[li]; Y=B[li]
        n=min(len(X),len(Y),5000)
        if n<100:
            cka=0.0; proc=0.0; cos=0.0; anc=0.0
        else:
            idxX=np.random.choice(len(X),n,replace=False)
            idxY=np.random.choice(len(Y),n,replace=False)
            Xs=X[idxX]; Ys=Y[idxY]
            cka=linear_cka_debiased(Xs,Ys)
            proc=procrustes_sim(Xs,Ys)
            cos=float(np.dot(Xs.mean(0),Ys.mean(0))/(np.linalg.norm(Xs.mean(0))+1e-8)/(np.linalg.norm(Ys.mean(0))+1e-8))
            anc=anc_similarity(Xs, Ys)
        if select_mode == 'semantic':
            auc_hi = float(semantic_aucs_main.get(li, {}).get('auc', 0.5))
            if semantic_neigh_maps:
                neighbor_aucs = [float(m.get(li, {}).get('auc', 0.5)) for m in semantic_neigh_maps]
                neigh = float(np.mean(neighbor_aucs)) if neighbor_aucs else 0.5
            else:
                neigh = 0.5
            sem = max(0.0, auc_hi - neigh)
            combo = 0.6*sem + 0.2*(1.0-cka) + 0.2*(1.0-proc)
        elif select_mode == 'similarity':
            if use_anc:
                combo = 0.4*cka + 0.4*proc + 0.2*anc
            else:
                combo = 0.5*cka + 0.4*proc + 0.1*cos
        else:  # contrast: prefer divergence between sets
            if use_anc:
                combo = 0.4*(1.0-cka) + 0.4*(1.0-proc) + 0.2*(1.0-anc)
            else:
                combo = 0.5*(1.0-cka) + 0.4*(1.0-proc) + 0.1*(1.0-cos)
        scores[li] = {"cka":cka,"proc":proc,"cos":cos, "anc":anc if n>=100 else 0.0, "combo":combo}
    # For similarity we take max(combo); for contrast we also take max(combo) since it's already inverted
    chosen=sorted(scores,key=lambda l:scores[l]["combo"],reverse=True)[:min(len(layers), top_k)]
    return chosen,scores

# -------------------- SAE (compact TopK) --------------------

class TopKSAE(nn.Module):
    def __init__(self, d, expansion=16, k=32, *, dtype=None, device=None):
        super().__init__()
        self.m, self.k = d*expansion, k
        if dtype is None: dtype = torch.float32
        self.E = nn.Linear(d, self.m, bias=False, device=device, dtype=dtype)
        self.D = nn.Linear(self.m, d, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.E.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.D.weight, a=math.sqrt(5))
    def encode(self,x):
        z=self.E(x)
        if self.k<self.m:
            topk = torch.topk(z.abs(), k=self.k, dim=-1)
            mask = torch.zeros_like(z, dtype=z.dtype)
            mask.scatter_(-1, topk.indices, torch.ones_like(topk.values, dtype=z.dtype))
            z = z * mask
        return z
    def decode(self,z): return self.D(z)
    def forward(self,x):
        z=self.encode(x); xhat=self.decode(z); return xhat,z

class SAEGate:
    """Attach SAE-based feature attenuation hooks to selected transformer layers.
    For each chosen layer, load a TopKSAE and attenuate a set of latent features
    by a factor (1 - alpha) during forward pass. This acts as a causal do-intervention
    on the concept subspace approximated by the SAE.
    """
    def __init__(self, model, layer_ids: List[int], sae_per_layer: Dict[int, TopKSAE], feature_idx: Dict[int, List[int]], alpha: float=0.5):
        self.model = model
        self.layer_ids = list(layer_ids)
        self.sae = sae_per_layer
        self.feature_idx = {li: torch.tensor(idx, dtype=torch.long) for li, idx in feature_idx.items()}
        self.alpha = float(alpha)
        self.handles = []
        self._attach()

    def _attach(self):
        tblocks=self.model.model.layers
        ref = next(self.model.parameters())
        pdev, pdtype = ref.device, ref.dtype
        for li in self.layer_ids:
            if li not in self.sae or li not in self.feature_idx or len(self.feature_idx[li])==0:
                continue
            sae = self.sae[li].to(device=pdev, dtype=torch.float32).eval()
            feat_idx = self.feature_idx[li].to(pdev)

            def make_hook(i:int, sae_module:TopKSAE, idx_tensor:torch.Tensor):
                @torch.no_grad()
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out  # [B,T,D]
                    B,T,D = h.shape
                    # Ensure SAE and index tensor are on the same device as this layer's output
                    if sae_module.E.weight.device != h.device:
                        sae_module.to(device=h.device)
                    idx_local = idx_tensor.to(h.device) if idx_tensor.device != h.device else idx_tensor
                    x = h.reshape(-1, D).to(torch.float32)
                    z0 = sae_module.E(x)  # [B*T, m]
                    x0 = sae_module.D(z0)  # baseline recon
                    z_edit = z0.clone()
                    if idx_local.numel()>0:
                        z_edit[:, idx_local] *= (1.0 - self.alpha)
                    xhat = sae_module.D(z_edit)
                    delta = (xhat - x0).to(h.dtype)
                    h2 = (h + delta.reshape(B, T, D))
                    return (h2, *out[1:]) if isinstance(out, tuple) else h2
                return hook
            h = tblocks[li].register_forward_hook(make_hook(li, sae, feat_idx))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles = []

    def set_alpha(self, alpha: float):
        self.alpha = float(max(0.0, min(1.0, alpha)))

class LinearProjectHook:
    """Project hidden states onto the nullspace of a learned subspace W per selected layer.
    H <- H - H P, where P = W (W^T W)^{-1} W^T. Acts like a simple LEACE/INLP-lite scrub.
    """
    def __init__(self, model, layer_ids: List[int], W_per_layer: Dict[int, torch.Tensor]):
        self.model = model
        self.layer_ids = list(layer_ids)
        self.W = {}
        self.P = {}
        self.handles = []
        ref = next(self.model.parameters())
        dev = ref.device
        for li, W in W_per_layer.items():
            if W is None: continue
            Wt = torch.as_tensor(W, dtype=torch.float32, device=dev)
            # Precompute P = W (W^T W)^{-1} W^T ; add ridge for stability
            WT_W = (Wt.T @ Wt)
            inv = torch.linalg.pinv(WT_W + 1e-4*torch.eye(WT_W.shape[0], device=dev))
            P = Wt @ inv @ Wt.T
            self.W[li] = Wt; self.P[li] = P
        self._attach()

    def _attach(self):
        tblocks=self.model.model.layers
        for li in self.layer_ids:
            if li not in self.P: continue
            P = self.P[li]
            def make_hook(Pmat: torch.Tensor):
                @torch.no_grad()
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    B,T,D = h.shape
                    # Move projector if needed to the layer's device
                    Puse = Pmat if Pmat.device == h.device else Pmat.to(h.device)
                    x = h.reshape(-1, D).to(torch.float32)
                    x2 = x - (x @ Puse)
                    return (x2.reshape(B,T,D).to(h.dtype), *out[1:]) if isinstance(out, tuple) else x2.reshape(B,T,D).to(h.dtype)
                return hook
            h = tblocks[li].register_forward_hook(make_hook(P))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles = []

class ResidualScaleHook:
    """Multiply the residual stream at selected layers by a constant scale (for sensitivity tests)."""
    def __init__(self, model, layer_ids: List[int], scale: float = 0.85):
        self.model = model
        self.layer_ids = list(layer_ids)
        self.scale = float(scale)
        self.handles = []
        self._attach()
    def _attach(self):
        tblocks = self.model.model.layers
        s = self.scale
        for li in self.layer_ids:
            def make_hook():
                @torch.no_grad()
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h2 = h * s
                    return (h2, *out[1:]) if isinstance(out, tuple) else h2
                return hook
            h = tblocks[li].register_forward_hook(make_hook())
            self.handles.append(h)
    def remove(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles = []

@torch.no_grad()
def learn_script_subspace(model, tok, texts_a: List[str], texts_b: List[str], layer: int, device: str, max_len=256, bs=32, cap_each=256, k: int = 1) -> np.ndarray:
    """Learn a linear subspace that separates script A vs B via logistic regression weights.
    Returns W in shape [D, k] (k columns). If k>1, uses top-k right singular vectors of stacked weight vectors.
    """
    from sklearn.linear_model import LogisticRegression
    Hs = []; ys = []
    count = 0
    for texts, y in [(texts_a,1), (texts_b,0)]:
        for batch in chunked(texts[:cap_each], bs):
            enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=max_len).to(device)
            out=model(**enc, output_hidden_states=True)
            H=out.hidden_states[layer+1].mean(dim=1).detach().to(torch.float32).cpu().numpy()
            Hs.append(H)
            ys.extend([y]*H.shape[0])
            count += H.shape[0]
            if count >= 2*cap_each: break
    if not Hs:
        return np.zeros((model.config.hidden_size, k), dtype=np.float32)
    X = np.concatenate(Hs, 0)
    y = np.array(ys, dtype=np.int32)
    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        w = clf.coef_.reshape(1, -1)  # [1, D]
    except Exception:
        return np.zeros((model.config.hidden_size, k), dtype=np.float32)
    # If k>1, augment by small random rotations; otherwise return single dir
    W = w
    if k > 1:
        # use SVD on covariance weighted by class means as a cheap proxy
        U, S, Vt = np.linalg.svd(np.cov(X.T)+1e-6*np.eye(X.shape[1]), full_matrices=False)
        W = Vt[:k]
    return W.reshape(-1, min(k, W.shape[0]))

@torch.no_grad()
def get_layer_batch_means(model,tok,texts,layer,device,max_len=256,bs=32):
    for batch in chunked(texts, bs):
        enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=max_len)
        enc=_to_model_device(model, enc)
        out=model(**enc, output_hidden_states=True)
        H=out.hidden_states[layer+1].mean(dim=1)  # [B,D]
        yield H

def train_sae(model,tok,texts,layer,device,steps=5000,bs=64,seq_len=256,lr=4e-4,aux_coeff=1/32,k=32,expansion=16):
    sae = TopKSAE(d=model.config.hidden_size, k=k, expansion=expansion).to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    it = infinite_from_factory(lambda: get_layer_batch_means(model,tok,texts,layer,device,max_len=seq_len,bs=bs))
    losses=[]
    pbar=tqdm(range(steps),desc=f"SAE-L{layer}")
    for _ in pbar:
        x=next(it).to(device).to(torch.float32)
        sae.train()
        xhat,z=sae(x)
        loss=F.mse_loss(xhat,x)+aux_coeff*z.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if len(losses)%500==0:
            pbar.set_postfix(loss=np.mean(losses[-500:]))
    return sae, {"loss":float(np.mean(losses[-min(500,len(losses)):]))}

@torch.no_grad()
def _sae_quality(model, tok, texts, layer, device, sae: 'TopKSAE', cap_each: int = 256, bs: int = 32, max_len: int = 256):
    """Lightweight SAE quality proxies inspired by SAEBench categories.
    Returns a dict with:
      - recon_mse: mean MSE(xhat, x)
      - sparsity_l0: mean fraction of z near-zero
      - dead_fraction: fraction of features whose max activation < eps
    """
    eps = 1e-6
    n_feat = int(sae.E.weight.shape[0])
    recon_losses = []
    z_nz_frac = []
    z_max = torch.zeros(n_feat, dtype=torch.float32)
    taken = 0
    for batch in chunked(texts[:cap_each], bs):
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        enc = _to_model_device(model, enc)
        out = model(**enc, output_hidden_states=True)
        H = out.hidden_states[layer+1].mean(dim=1).to(torch.float32)  # [B, D]
        x = H
        z = sae.E(x)
        xhat = sae.D(z)
        recon_losses.append(F.mse_loss(xhat, x).detach().cpu())
        z_abs = z.abs()
        z_nz_frac.append((z_abs < eps).float().mean(dim=1).detach().cpu())
        z_max = torch.maximum(z_max, z_abs.max(dim=0).values.detach().cpu())
        taken += x.shape[0]
        if taken >= cap_each:
            break
    recon = float(torch.stack(recon_losses).mean().item()) if recon_losses else float('nan')
    sparsity_l0 = float(torch.cat(z_nz_frac).mean().item()) if z_nz_frac else float('nan')
    dead_frac = float(((z_max < 1e-3).float().mean().item()) if n_feat > 0 else 0.0)
    return {
        "recon_mse": recon,
        "sparsity_l0": sparsity_l0,
        "dead_fraction": dead_frac,
        "k": int(getattr(sae, 'k', 0)),
        "expansion": int(getattr(sae, 'expansion', 0)),
        "d": int(getattr(model.config, 'hidden_size', 0)),
    }

@torch.no_grad()
def pick_sae_features_forget_vs_retain(sae: TopKSAE, model, tok, forget: List[str], retain: List[str], layer: int, device: str, max_len=256, bs=32, cap_each=256, topk=64) -> List[int]:
    """Identify SAE latent features most associated with forget examples vs retain.
    Compute mean |z| per group and take top positive differences.
    """
    def collect(texts):
        vals = []
        count = 0
        for batch in chunked(texts[:cap_each], bs):
            enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=max_len)
            enc=_to_model_device(model, enc)
            out=model(**enc, output_hidden_states=True)
            H=out.hidden_states[layer+1]  # [B,T,D]
            X=H.mean(dim=1).to(torch.float32)  # [B,D]
            Z=sae.E(X)  # [B,m]
            vals.append(Z.abs().mean(dim=0).detach().cpu().numpy())
            count += len(batch)
            if count >= cap_each: break
        return np.mean(vals, axis=0) if vals else np.zeros((sae.E.weight.shape[0],), dtype=np.float32)
    f = collect(forget)
    r = collect(retain)
    diff = f - r
    idx = np.argsort(diff)[::-1]
    idx = [int(i) for i in idx[:max(1, topk)] if diff[i] > 0]
    return idx

@torch.no_grad()
def pick_semantic_sae_features(sae: TopKSAE, model, tok,
                               hindi_deva: List[str], hindi_roman: List[str], deva_gib: List[str],
                               layer: int, device: str, max_len=256, bs=32, cap_each=256, topk=64, tau: float = 0.0) -> List[int]:
    """Pick SAE features that are invariant to script and insensitive to script-only noise.
    Score(f) = min(|z| on Hindi-Devanagari, |z| on Hindi-Roman) - |z| on Devanagari-gibberish.
    Keep top-K with score > tau.
    """
    def collect(texts):
        vals = []
        count = 0
        for batch in chunked(texts[:cap_each], bs):
            enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=max_len)
            enc=_to_model_device(model, enc)
            out=model(**enc, output_hidden_states=True)
            H=out.hidden_states[layer+1]
            X=H.mean(dim=1).to(torch.float32)
            Z=sae.E(X)
            vals.append(Z.abs().mean(dim=0).detach().cpu().numpy())
            count += len(batch)
            if count >= cap_each: break
        if not vals:
            m = sae.E.weight.shape[0]
            return np.zeros((m,), dtype=np.float32)
        return np.mean(vals, axis=0)

    f_deva = collect(hindi_deva)
    f_roman = collect(hindi_roman)
    f_gib   = collect(deva_gib)
    score = np.minimum(f_deva, f_roman) - f_gib
    idx = np.argsort(score)[::-1]
    idx = [int(i) for i in idx[:max(1, topk)] if score[i] > tau]
    return idx

# -------------------- ReFT (minimal low-rank layer adapters) --------------------

class ReFTAdapter(nn.Module):
    def __init__(self,d,rank=4,*,dtype=None,device=None):
        super().__init__()
        if dtype is None: dtype = torch.float32
        self.A=nn.Linear(d,rank,bias=False,device=device,dtype=dtype)
        self.B=nn.Linear(rank,d,bias=False,device=device,dtype=dtype)
        nn.init.normal_(self.A.weight,std=1e-3); nn.init.normal_(self.B.weight,std=1e-3)
    def forward(self,h): return h + self.B(self.A(h))

def attach_reft(model,layers,device,rank=4):
    adapters=nn.ModuleDict(); handles=[]
    ref = next(model.parameters())
    pdtype, pdev = ref.dtype, ref.device
    tblocks=model.model.layers
    for li in layers:
        adapters[str(li)] = ReFTAdapter(model.config.hidden_size,rank=rank,dtype=pdtype,device=pdev)
        def make_hook(i:int):
            def hook(mod,inp,out):
                h = out[0] if isinstance(out, tuple) else out
                # Ensure adapter lives on the same device as this layer's output
                ad = adapters[str(i)]
                try:
                    pdev = next(ad.parameters()).device
                except StopIteration:
                    pdev = None
                if pdev is None or pdev != h.device:
                    ad.to(h.device)
                h2 = ad(h)
                return (h2, *out[1:]) if isinstance(out, tuple) else h2
            return hook
        h = tblocks[li].register_forward_hook(make_hook(li))
        handles.append(h)
    return adapters, handles

def apply_reft_from_file(model, layers, path, rank=4):
    adapters, handles = attach_reft(model, layers, device=next(model.parameters()).device, rank=rank)
    if os.path.exists(path):
        sd = torch.load(path, map_location=next(model.parameters()).device)
        adapters.load_state_dict(sd, strict=False)
        print(f"[reft] loaded {path}")
    else:
        print(f"[reft] no adapter file {path}, running base model for ReFT arm.")
    return adapters, handles

def ensure_generation_padding(model, tok):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None and tok.pad_token_id >= 0 else tok.eos_token_id
    eos_id = tok.eos_token_id if tok.eos_token_id is not None and tok.eos_token_id >= 0 else pad_id
    if pad_id is None or pad_id < 0: pad_id = eos_id
    if getattr(model.config, "pad_token_id", None) in (None, -1): model.config.pad_token_id = pad_id
    if getattr(model.config, "eos_token_id", None) in (None, -1): model.config.eos_token_id = eos_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        if getattr(gen_cfg, "pad_token_id", None) in (None, -1): gen_cfg.pad_token_id = pad_id
        if getattr(gen_cfg, "eos_token_id", None) in (None, -1): gen_cfg.eos_token_id = eos_id

def load_causal_lm(model_id: str, tok, device: str, hf_token=None, eval_mode=False):
    cfg = AutoConfig.from_pretrained(model_id, token=hf_token)
    on_cuda = (device.startswith("cuda") and torch.cuda.is_available())
    dtype = torch.float16 if on_cuda else torch.float32
    # Optional knobs via environment (no new CLI flags needed):
    # - OFFLOAD_DIR: set to a writable folder to allow accelerate to offload weights
    # - LOAD_IN_8BIT / LOAD_IN_4BIT: set to '1' to request quantized loading (requires bitsandbytes)
    offload_dir = os.environ.get('OFFLOAD_DIR') if on_cuda else None
    load_in_8bit = os.environ.get('LOAD_IN_8BIT') == '1'
    load_in_4bit = os.environ.get('LOAD_IN_4BIT') == '1'
    kwargs = dict(
        config=cfg,
        token=hf_token,
        torch_dtype=dtype,
        device_map="auto" if on_cuda else None,
        low_cpu_mem_usage=True,
    )
    if offload_dir:
        kwargs.update(dict(offload_folder=offload_dir, offload_state_dict=True))
    if load_in_4bit:
        kwargs.update(dict(load_in_4bit=True))
    elif load_in_8bit:
        kwargs.update(dict(load_in_8bit=True))
    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except OSError as e:
        # Retry once with SAFETENSORS_FAST=0 and an explicit offload folder on Windows
        if os.name == 'nt' and ('1455' in str(e) or 'paging file' in str(e).lower()):
            os.environ['SAFETENSORS_FAST'] = '0'
            if not offload_dir:
                offload_dir = os.path.join(os.getcwd(), 'offload')
                os.makedirs(offload_dir, exist_ok=True)
            kwargs.update(dict(offload_folder=offload_dir, offload_state_dict=True))
            mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        else:
            raise
    ensure_generation_padding(mdl, tok)
    try: mdl.gradient_checkpointing_enable()
    except Exception: pass
    if hasattr(mdl.config, "use_cache"): mdl.config.use_cache = False
    if not on_cuda: mdl.to("cpu")
    if eval_mode: mdl.eval()
    return mdl

def loader(tok,texts,device,bs=16,max_len=256):
    enc=tok(texts, return_tensors='pt',padding=True,truncation=True,max_length=max_len)
    ds=torch.utils.data.TensorDataset(enc["input_ids"],enc["attention_mask"])
    def collate(b):
        ids=torch.stack([x[0] for x in b]); att=torch.stack([x[1] for x in b])
        return {"input_ids":ids.to(device),"attention_mask":att.to(device)}
    return torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=True,collate_fn=collate)

def nll(model,batch):
    out=model(**batch, labels=batch["input_ids"])
    return out.loss

def kl_to_base(model,base_logits,batch):
    out=model(**batch)
    p=F.log_softmax(out.logits,dim=-1); q=F.softmax(base_logits,dim=-1)
    return F.kl_div(p,q,reduction="batchmean")

def npo_loss(model, ref_model, batch, beta: float = 0.1):
    """Negative Preference Optimization loss on a batch.
    Treats the batch labels as the dispreferred target; pushes model likelihood
    below the reference model while avoiding unstable GA.
    Uses token-mean losses as proxy for log-prob densities; constants cancel in the difference.
    """
    out_m = model(**{**batch, "labels": batch["input_ids"]})
    with torch.no_grad():
        out_r = ref_model(**{**batch, "labels": batch["input_ids"]})
    # delta ~ log P_model - log P_ref  (up to a negative constant factor)
    # losses are mean negative log-likelihoods; use difference with sign flipped
    delta = -(out_m.loss - out_r.loss)
    # Use functional API for broad Torch compatibility
    return -(2.0/beta) * F.logsigmoid(-beta * delta)

@torch.no_grad()
def token_kl_to_base(model, base, batch):
    """Average token-level KL(P_model || P_base) for a batch.
    Useful to quantify retain distributional drift beyond PPL.
    """
    out_m = model(**batch)
    out_b = base(**batch)
    p = F.log_softmax(out_m.logits, dim=-1)
    q = F.softmax(out_b.logits, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")

def train_lora(model,tok,forget,retain,device,steps=500,bs=16,max_len=256,lr=2e-4,hf_token=None,grad_clip=1.0, forget_obj: str = "ga", ckpt_dir: str = "."):
    try:
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
        cfg=LoraConfig(r=4,lora_alpha=16,lora_dropout=0.0,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
        model=get_peft_model(model,cfg)
    except Exception:
        print("[warn] PEFT not available; training full model heads (no LoRA checkpoint).")
        get_peft_model_state_dict=None
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    Lf=loader(tok,forget,device,bs,max_len); Lr=loader(tok,retain,device,bs,max_len)
    itf=infinite_loader(Lf); itr=infinite_loader(Lr)
    base=None
    model.train()
    for step in tqdm(range(steps),desc="LoRA"):
        if step%2==0:
            b=next(itf)
            if forget_obj=="npo":
                if base is None:
                    base=load_causal_lm(model.config._name_or_path, tok, device, hf_token, eval_mode=True)
                    [p.requires_grad_(False) for p in base.parameters()]
                loss=npo_loss(model, base, b)
            else:
                loss=-nll(model,b)
        else:
            b=next(itr)
            if base is None:
                base=load_causal_lm(model.config._name_or_path, tok, device, hf_token, eval_mode=True)
                [p.requires_grad_(False) for p in base.parameters()]
            with torch.no_grad(): base_logits=base(**b).logits.detach()
            loss=kl_to_base(model,base_logits,b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    # save adapters only
    try:
        if get_peft_model_state_dict is not None:
            state = get_peft_model_state_dict(model)
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, "lora_adapters.pt")
            torch.save(state, path)
            print(f"[lora] saved {path}")
    except Exception as e:
        print(f"[lora] save skipped: {e}")
    model.eval(); return model

def resume_lora(model, tok, device, ckpt_dir: str = "."):
    try:
        from peft import LoraConfig, get_peft_model
        cfg=LoraConfig(r=4,lora_alpha=16,lora_dropout=0.0,target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM")
        model=get_peft_model(model,cfg)
        path = os.path.join(ckpt_dir, "lora_adapters.pt")
        if os.path.exists(path):
            sd=torch.load(path, map_location=device)
            model.load_state_dict(sd, strict=False)
            print(f"[lora] loaded {path}")
        else:
            print(f"[lora] no adapter file at {path}, using base weights")
    except Exception as e:
        print(f"[lora] resume skipped: {e}")
    model.eval(); return model

def train_reft(model,tok,layers,forget,retain,device,rank=4,steps=500,bs=16,max_len=256,lr=2e-4,hf_token=None,grad_clip=1.0, forget_obj: str = "ga", ckpt_dir: str = "."):
    adapters,handles=attach_reft(model,layers,device,rank)
    opt=torch.optim.AdamW(adapters.parameters(),lr=lr)
    Lf=loader(tok,forget,device,bs,max_len); Lr=loader(tok,retain,device,bs,max_len)
    itf=infinite_loader(Lf); itr=infinite_loader(Lr)
    base=load_causal_lm(model.config._name_or_path, tok, device, hf_token, eval_mode=True)
    [p.requires_grad_(False) for p in base.parameters()]
    model.train()
    for step in tqdm(range(steps),desc="ReFT"):
        if step%2==0:
            b=next(itf)
            if forget_obj=="npo":
                loss=npo_loss(model, base, b)
            else:
                loss=-nll(model,b)
        else:
            b=next(itr)
            with torch.no_grad(): base_logits=base(**b).logits.detach()
            loss=kl_to_base(model,base_logits,b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), grad_clip)
        opt.step()
    # keep hooks for inference; save adapters
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, "reft_adapters.pt")
        torch.save(adapters.state_dict(), path)
        print(f"[reft] saved {path}")
    except Exception as e:
        print(f"[reft] save skipped: {e}")
    model.eval(); return model

# -------------------- generation & metrics --------------------

@torch.no_grad()
def generate(model,tok,prompts,device,max_new_tokens=64):
    out=[]
    for batch in chunked(prompts,8):
        enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=256)
        enc=_to_model_device(model, enc)
        ids=model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        # Decode only newly generated tokens, excluding the prompt
        prefix_len = enc["input_ids"].shape[1]
        gen_ids = ids[:, prefix_len:]
        out.extend(tok.batch_decode(gen_ids, skip_special_tokens=True))
    return out


from transformers import LogitsProcessor

class DynamicGatingLogitsProcessor(LogitsProcessor):
    def __init__(self, tok, lid: LIDEnsemble, prompt_len: int, gate: SAEGate|None, target_code: str = "hi", base_alpha: float = 0.3, high_alpha: float = 0.7, penalty: float = 2.0):
        self.tok = tok
        self.lid = lid
        self.prompt_len = prompt_len
        self.gate = gate
        self.target_code = target_code
        self.base_alpha = float(base_alpha)
        self.high_alpha = float(high_alpha)
        self.penalty = float(penalty)
        # Precompute risky token ids (script or romanized cues)
        self.risky_ids = self._compute_risky_token_ids()

    def _compute_risky_token_ids(self):
        risky = set()
        roman_cues = {" hai "," nahi "," kya "," kyun "," kaise "," kripya "," dhanyavaad "," shukriya "," tum "," aap "," mera "," meri "," hum "," unhe "," kam "," zyaada "," bahut "," kyunki "}
        for tid in range(self.tok.vocab_size):
            try:
                s = self.tok.decode([tid], skip_special_tokens=False)
            except Exception:
                continue
            if any('\u0900' <= ch <= '\u097F' for ch in s):
                risky.add(tid); continue
            ls = f" {s.lower()} "
            if any(cue in ls for cue in roman_cues):
                risky.add(tid)
        return risky

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Per sequence: decode generated continuation and check LID risk
        B = input_ids.shape[0]
        for b in range(B):
            gen_ids = input_ids[b, self.prompt_len:]
            if gen_ids.numel() == 0:
                if self.gate is not None:
                    self.gate.set_alpha(self.base_alpha)
                continue
            try:
                text = self.tok.decode(gen_ids, skip_special_tokens=True)
            except Exception:
                text = ""
            code,_ = self.lid.infer(text)
            risky = (code == self.target_code)
            if self.gate is not None:
                self.gate.set_alpha(self.high_alpha if risky else self.base_alpha)
            if risky:
                # apply penalty to risky ids for this sequence
                scores[b, list(self.risky_ids)] -= self.penalty
        return scores

def generate_with_dynamic_gating(model, tok, lid: LIDEnsemble, prompts, device, gate: SAEGate, base_alpha: float=0.3, high_alpha: float=0.7, max_new_tokens=64):
    """Per-token dynamic gating via logits processor + SAE alpha scheduling."""
    out=[]
    for batch in chunked(prompts,8):
        enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=256)
        enc=_to_model_device(model, enc)
        prompt_len = enc["input_ids"].shape[1]
        processor = DynamicGatingLogitsProcessor(tok, lid, prompt_len, gate, target_code="hi", base_alpha=base_alpha, high_alpha=high_alpha, penalty=2.0)
        ids=model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            logits_processor=[processor],
        )
        prefix_len = prompt_len
        gen_ids = ids[:, prefix_len:]
        out.extend(tok.batch_decode(gen_ids, skip_special_tokens=True))
    return out

class SemanticGatingLogitsProcessor(LogitsProcessor):
    """Script-blind gating: never penalize specific token IDs; only schedule SAE alpha
    based on a script-blind LID estimate over the generated continuation."""
    def __init__(self, tok, lid: LIDEnsemble, prompt_len: int, gate: SAEGate|None, target_code: str = "hi", base_alpha: float = 0.3, high_alpha: float = 0.7):
        self.tok = tok
        self.lid = lid
        self.prompt_len = prompt_len
        self.gate = gate
        self.target_code = target_code
        self.base_alpha = float(base_alpha)
        self.high_alpha = float(high_alpha)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = input_ids.shape[0]
        for b in range(B):
            gen_ids = input_ids[b, self.prompt_len:]
            if gen_ids.numel() == 0:
                if self.gate is not None:
                    self.gate.set_alpha(self.base_alpha)
                continue
            try:
                text = self.tok.decode(gen_ids, skip_special_tokens=True)
            except Exception:
                text = ""
            # Script-blind: if possible, romanize to avoid Devanagari script cues
            if _HAS_DEVANAGARI_TO_LATIN:
                try:
                    text = batch_devanagari_to_latin([text])[0]
                except Exception:
                    pass
            code,_ = self.lid.infer(text)
            risky = (code == self.target_code)
            if self.gate is not None:
                self.gate.set_alpha(self.high_alpha if risky else self.base_alpha)
        return scores

def generate_with_semantic_gating(model, tok, lid: LIDEnsemble, prompts, device, gate: SAEGate, base_alpha: float=0.3, high_alpha: float=0.7, max_new_tokens=64):
    out=[]
    for batch in chunked(prompts,8):
        enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=256)
        enc=_to_model_device(model, enc)
        prompt_len = enc["input_ids"].shape[1]
        processor = SemanticGatingLogitsProcessor(tok, lid, prompt_len, gate, target_code="hi", base_alpha=base_alpha, high_alpha=high_alpha)
        ids=model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            logits_processor=[processor],
        )
        prefix_len = prompt_len
        gen_ids = ids[:, prefix_len:]
        out.extend(tok.batch_decode(gen_ids, skip_special_tokens=True))
    return out

def _has_devanagari(s:str)->bool:
    return any('\u0900' <= ch <= '\u097F' for ch in s)

def extraction_strength(gens:List[str], lid:LIDEnsemble, target_code="hi", use_script_guard:bool=True)->float:
    vals=[]
    for t in gens:
        toks=t.split()
        if not toks: vals.append(0.0); continue
        n=len(toks); found=None
        for i in range(1,n+1):
            code,_=lid.infer(" ".join(toks[:i]))
            if code==target_code:
                found=i; break
        if found is None and use_script_guard:
            for i in range(1,n+1):
                if _has_devanagari(" ".join(toks[:i])):
                    found=i; break
        vals.append(0.0 if found is None else 1.0 - (found/n))
    return float(np.mean(vals)) if vals else 0.0

def _romanize_texts(texts: List[str]) -> List[str]:
    """Romanize Devanagari to Latin if possible; otherwise return as-is."""
    if _HAS_DEVANAGARI_TO_LATIN:
        try:
            return batch_devanagari_to_latin(texts)
        except Exception:
            return texts
    return texts

def _make_devanagari_gibberish(texts: List[str], seed: int = 0) -> List[str]:
    """Generate simple Devanagari-looking gibberish strings as negative controls."""
    rng = random.Random(seed)
    letters = [chr(c) for c in range(0x0915, 0x0939+1)]  # क..ह
    matras = [chr(c) for c in [0x093E,0x093F,0x0940,0x0941,0x0942,0x0947,0x0948,0x094B,0x094C]]
    out = []
    for t in texts:
        n = max(8, min(200, len(t)))
        s = []
        for i in range(n):
            s.append(rng.choice(letters if i % 3 != 2 else matras))
        out.append("".join(s))
    return out

@torch.no_grad()
def perplexity(model,tok,texts,device)->float:
    losses=[]
    for batch in chunked(texts,8):
        enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=256)
        enc=_to_model_device(model, enc)
        out=model(**enc, labels=enc["input_ids"])
        losses.append(out.loss.item())
    return float(math.exp(np.mean(losses))) if losses else float("inf")

@torch.no_grad()
def mean_activations(model, tok, texts, layers, device, max_len: int = 256):
    acts = {li: [] for li in layers}
    for batch in chunked(texts, 8):
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
        enc = _to_model_device(model, enc)
        out = model(**enc, output_hidden_states=True)
        for li in layers:
            H = out.hidden_states[li+1].mean(dim=1)           # [B, D]
            H = H.detach().to(torch.float32).cpu().numpy()
            H = np.nan_to_num(H, nan=0.0, posinf=1e4, neginf=-1e4)
            acts[li].append(H)
    out = {}
    for li in layers:
        if acts[li]:
            X = np.concatenate(acts[li], 0).astype(np.float32, copy=False)
            X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
        else:
            X = np.zeros((0, model.config.hidden_size), dtype=np.float32)
        out[li] = X
    return out


def probes_auc(model, tok, pos, neg, probe_layers, device, max_len: int = 256) -> Dict[int, Dict[str, float]]:
    from sklearn.model_selection import train_test_split
    res = {}
    Xp = mean_activations(model, tok, pos, probe_layers, device, max_len=max_len)
    Xn = mean_activations(model, tok, neg, probe_layers, device, max_len=max_len)
    for li in probe_layers:
        Xp_li, Xn_li = Xp[li], Xn[li]
        if len(Xp_li) == 0 or len(Xn_li) == 0:
            res[li] = {"acc": 0.5, "auc": 0.5}
            continue
        X = np.vstack([Xp_li, Xn_li]).astype(np.float32, copy=False)
        y = np.array([1] * len(Xp_li) + [0] * len(Xn_li), dtype=np.int32)
        # sanitize any remaining bad values
        X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)

        # need both classes in both splits
        try:
            idx = np.arange(len(X))
            tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=0)
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                res[li] = {"acc": 0.5, "auc": 0.5}
                continue
            clf = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            res[li] = {
                "acc": float(accuracy_score(y[te], (p >= 0.5).astype(int))),
                "auc": float(roc_auc_score(y[te], p)),
            }
        except Exception:
            # if sklearn still complains (e.g., degenerate data), fall back
            res[li] = {"acc": 0.5, "auc": 0.5}
    return res

@torch.no_grad()
def mia_loss(base,edited,tok,forget,nonmember,device):
    def losses(m, texts):
        L=[]
        for batch in chunked(texts,8):
            enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=256).to(device)
            out=m(**enc, labels=enc["input_ids"])
            L.append(float(out.loss.detach().cpu()))    
        return np.array(L, dtype=np.float32)
    Lb_f=losses(base,forget); Le_f=losses(edited,forget)
    Lb_n=losses(base,nonmember); Le_n=losses(edited,nonmember)
    y=np.array([1]*len(Lb_f)+[0]*len(Lb_n))
    delta=np.concatenate([Lb_f-Le_f, Lb_n-Le_n])
    if len(delta)==0: return {"AUC":0.5,"ACC":0.5}
    acc=float(((delta>=0).astype(int)==y).mean())
    try: auc=float(roc_auc_score(y,delta))
    except: auc=0.5
    return {"AUC":auc,"ACC":acc}

# -------------------- args & main --------------------

@dataclass
class Args:
    model:str; forget:str; retain:str; mixed:str
    xlang:List[str]; adversarial:str
    seeds:List[int]; train_sae_steps:int; sae_k:int; sae_expansion:int
    rank:int; device:str; out:str; sample_cap:int; probe_layers:List[int]; max_len:int
    use_xlmr:bool; use_gemini:bool; use_fasttext:bool
    lora_steps:int; reft_steps:int
    forget_obj:str
    sae_gate: bool; sae_gate_alpha: float; sae_gate_topk: int; report_token_kl: bool

    es_romanized: bool; dynamic_gate: bool
    semantic_dynamic_gate: bool; semantic_features: bool
    use_anc: bool; script_scrub: bool; scrub_k: int; sae_lens_dir: Optional[str]
    # Layer selection/report defaults
    select_top_k: int; min_layer: int; print_layer_scores: bool; select_mode: str; script_blind_selection: bool
    judge_assist_selection: bool; judge_cap: int; judge_pool: int; judge_scale: float; judge_alpha: float; judge_beta: float; judge_model: str; judge_timeout: float
    # Reproducibility / control
    selection_seed: Optional[int]; force_layers: Optional[List[int]]
    # Stability selection aggregation
    stability_select: int; stability_seeds: Optional[List[int]]; stability_strategy: str
    # New: SAE quality proxies
    sae_quality_eval: bool; sae_eval_cap: int
    # New: thresholds and semantic picker tau
    gate_es_forget_ratio: float; gate_es_mixed_ratio: float; gate_ppl_ratio: float
    semantic_tau: float
    ckpt_dir: str

def parse():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--forget",required=True)
    ap.add_argument("--retain",required=True)
    ap.add_argument("--mixed",required=True)
    ap.add_argument("--xlang",nargs="+",default=[])
    ap.add_argument("--adversarial",default="adversarial.jsonl")
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    ap.add_argument("--train_sae_steps",type=int,default=5000)
    ap.add_argument("--sae_k",type=int,default=32)
    ap.add_argument("--sae_expansion",type=int,default=16)
    ap.add_argument("--rank",type=int,default=4)
    ap.add_argument("--device",default="cuda")
    ap.add_argument("--out",default="eval_report.json")
    ap.add_argument("--sample_cap",type=int,default=1000)
    ap.add_argument("--probe_layers",nargs="+",type=int,default=[8,16,24])
    ap.add_argument("--select_top_k", type=int, default=3, help="Number of layers to select for edits")
    ap.add_argument("--min_layer", type=int, default=2, help="Exclude layers below this index from selection (avoid early lexical layers)")
    ap.add_argument("--print_layer_scores", action="store_true", default=True, help="Print CKA/Procrustes/Cos/ANC per layer and chosen set (default on)")
    ap.add_argument("--select_mode", choices=["contrast","similarity","semantic"], default="contrast", help="Layer selection mode: contrast (prefer divergence), similarity, or semantic (Hindi-vs-English specificity vs neighbors)")
    ap.add_argument("--script_blind_selection", dest="script_blind_selection", action="store_true", help="Romanize Devanagari to make selection script-blind")
    ap.add_argument("--no_script_blind_selection", dest="script_blind_selection", action="store_false")
    ap.set_defaults(script_blind_selection=True)
    # LLM judge assisted selection (optional)
    ap.add_argument("--judge_assist_selection", action="store_true", help="Use LLM judge to refine layer selection (requires GEMINI_API_KEY and google-generativeai)")
    ap.add_argument("--judge_cap", type=int, default=24, help="Total prompts for judge (half forget, half retain)")
    ap.add_argument("--judge_pool", type=int, default=6, help="Candidate layers to test with judge")
    ap.add_argument("--judge_scale", type=float, default=0.85, help="Residual scale for sensitivity test")
    ap.add_argument("--judge_alpha", type=float, default=0.5, help="Blend weight metric vs judge (0..1)")
    ap.add_argument("--judge_beta", type=float, default=0.5, help="Penalty weight on retain degradation (0..1)")
    ap.add_argument("--judge_model", type=str, default="gemini-2.5-flash", help="LLM judge model name")
    ap.add_argument("--judge_timeout", type=float, default=15.0, help="Timeout per judge call in seconds (fallback to metrics if exceeded)")
    # Reproducibility / control
    ap.add_argument("--selection_seed", type=int, default=42, help="Fix random seeds for selection (numpy/torch/random)")
    ap.add_argument("--force_layers", nargs="+", type=int, default=None, help="Bypass selection and force specific layer indices (e.g., 10 16 19)")
    # Stability selection (vote/average over multiple seeds)
    ap.add_argument("--stability_select", type=int, default=0, help="If >0, run selection multiple times with different seeds and aggregate")
    ap.add_argument("--stability_seeds", nargs="+", type=int, default=None, help="Explicit list of seeds to use for stability selection")
    ap.add_argument("--stability_strategy", choices=["vote","average"], default="vote", help="Aggregate by vote (default) or by average combo score")
    ap.add_argument("--max_len",type=int,default=256)
    ap.add_argument("--use_xlmr",action="store_true")
    ap.add_argument("--use_gemini",action="store_true")
    ap.add_argument("--use_fasttext",action="store_true")
    ap.add_argument("--lora_steps",type=int,default=500, help="LoRA optimisation steps")
    ap.add_argument("--reft_steps",type=int,default=500, help="ReFT optimisation steps")
    ap.add_argument("--forget_obj",choices=["ga","npo"],default="ga", help="Forget objective: gradient-ascent (ga) or Negative Preference Optimization (npo)")
    ap.add_argument("--sae_gate",action="store_true", help="Enable SAE feature gating during evaluation")
    ap.add_argument("--sae_gate_alpha",type=float,default=0.5, help="Attenuation strength for SAE-gated features (0..1)")
    ap.add_argument("--sae_gate_topk",type=int,default=64, help="Top-K SAE features per layer to attenuate")
    ap.add_argument("--report_token_kl",action="store_true", help="Compute token-level KL to base on retain")
    ap.add_argument("--es_romanized",action="store_true", help="Report ES on romanized outputs (transliteration aware)")
    ap.add_argument("--dynamic_gate",action="store_true", help="Heuristic dynamic SAE gating based on prompt LID risk")
    ap.add_argument("--semantic_dynamic_gate",action="store_true", help="Script-blind SAE gating: schedule alpha by semantic LID; no token penalties")
    ap.add_argument("--semantic_features",action="store_true", help="Pick SAE features invariant across Hindi scripts and insensitive to Devanagari gibberish")
    ap.add_argument("--use_anc",action="store_true", help="Include ANC similarity in layer-selection combo score")
    ap.set_defaults(use_anc=True)
    ap.add_argument("--script_scrub",action="store_true", help="Apply linear script subspace projection (LEACE/INLP-lite) on chosen layers during evaluation")
    ap.add_argument("--scrub_k",type=int,default=1, help="Dimensionality of script subspace to remove per layer")
    ap.add_argument("--sae_lens_dir",type=str,default=None, help="Directory containing SAELens-trained SAE weights for layers; if provided, load these for gating/analysis")
    ap.add_argument("--sae_quality_eval", action="store_true", help="Compute lightweight SAE quality metrics (recon MSE, sparsity, dead fraction) on chosen layers")
    ap.add_argument("--sae_eval_cap", type=int, default=256, help="Cap of samples per set for SAE quality eval")
    # Gating thresholds
    ap.add_argument("--gate_es_forget_ratio", type=float, default=0.5, help="Threshold ratio for ES on forget (edited <= ratio * base)")
    ap.add_argument("--gate_es_mixed_ratio", type=float, default=0.7, help="Threshold ratio for ES on mixed (edited <= ratio * base)")
    ap.add_argument("--gate_ppl_ratio", type=float, default=1.10, help="Max allowed PPL ratio on retain (edited/base <= this)")
    # Semantic picker score cutoff
    ap.add_argument("--semantic_tau", type=float, default=0.0, help="Minimum score threshold for semantic SAE feature selection")
    ap.add_argument("--ckpt_dir", type=str, default=".", help="Directory to read/write adapters and SAE weights")
    return Args(**vars(ap.parse_args()))

# Utility: save activations for sets (added)
@torch.no_grad()
def save_activations_for_sets(model, tok, device, layers, datasets: Dict[str, List[str]], out_stem: str, cap_per_set: int = 200):
    """Save mean activations per sequence for each dataset and layer.
    Writes compressed .npz files under activations/<out_stem>/L{layer}_{set}.npz
    """
    import os
    base_dir = os.path.join("activations", out_stem)
    os.makedirs(base_dir, exist_ok=True)
    Xs = {}
    for name, texts in datasets.items():
        X = mean_activations(model, tok, texts[:cap_per_set], layers, device)
        for li in layers:
            arr = X.get(li)
            if arr is None:
                continue
            path = os.path.join(base_dir, f"L{li}_{name}.npz")
            try:
                np.savez_compressed(path, acts=arr)
            except Exception as e:
                print(f"[acts] save failed {path}: {e}")

def main():
    load_dotenv()
    args=parse()
    device=args.device
    hf_token=os.environ.get("HF_TOKEN")
    # Optional deterministic selection
    if args.selection_seed is not None:
        try:
            import random
            random.seed(int(args.selection_seed))
        except Exception:
            pass
        try:
            np.random.seed(int(args.selection_seed))
        except Exception:
            pass
        try:
            torch.manual_seed(int(args.selection_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.selection_seed))
        except Exception:
            pass

    # Data
    forget=read_jsonl(args.forget)
    retain=read_jsonl(args.retain)
    mixed=read_jsonl(args.mixed)
    xlang_sets=[(Path(p).stem, read_jsonl(p)) for p in args.xlang]
    adversarial=read_jsonl(args.adversarial)

    # LID ensemble
    lid=LIDEnsemble(LIDConfig(
        use_fasttext=args.use_fasttext,
        fasttext_bin=os.environ.get("FASTTEXT_LID_PATH"),
        use_xlmr=args.use_xlmr,
        use_gemini=args.use_gemini,
        vote_require_majority=True
    ))

    # Base tokenizer/model
    tok=AutoTokenizer.from_pretrained(args.model, token=hf_token, use_fast=True)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None or tok.pad_token == tok.unk_token:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
    if tok.pad_token_id is None or tok.pad_token_id < 0 or tok.pad_token_id == getattr(tok, "unk_token_id", -2):
        if tok.eos_token_id is not None and tok.eos_token_id >= 0:
            tok.pad_token_id = tok.eos_token_id
    if tok.pad_token_id is None or tok.pad_token_id < 0:
        raise ValueError("Tokenizer must expose a valid pad_token_id")

    base=load_causal_lm(args.model, tok, device, hf_token, eval_mode=True)
    [p.requires_grad_(False) for p in base.parameters()]
    n_layers=len(base.model.layers)
    probe_layers=[l for l in args.probe_layers if l < n_layers] or list(range(n_layers))

    # Select layers (or force) with optional stability aggregation
    if args.force_layers:
        chosen = sorted({int(l) for l in args.force_layers if 0 <= int(l) < n_layers})[:max(1, int(args.select_top_k))]
        scores = {li: {"combo": None} for li in chosen}
        print(f"[layers] forced: {chosen}  (min_layer={args.min_layer}, top_k={args.select_top_k})")
    else:
        # Optional stability selection over multiple seeds
        if int(args.stability_select) > 0:
            seeds_list = (args.stability_seeds if args.stability_seeds else [11,23,37,61,89])
            if len(seeds_list) < int(args.stability_select):
                # extend deterministically
                base_seeds = [101,131,151,181,211,241,271,301]
                seeds_list = seeds_list + base_seeds[:int(args.stability_select)-len(seeds_list)]
            seeds_list = seeds_list[:int(args.stability_select)]

            vote = Counter()
            avg_scores: dict[int, float] = {}
            # Precompute once per run for judge refinement base slice
            # We reuse the logic below inside the loop per seed
            for s in seeds_list:
                # Seed for this selection
                try:
                    random.seed(int(s)); np.random.seed(int(s)); torch.manual_seed(int(s));
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(s))
                except Exception:
                    pass
                sel_chosen, sel_scores = select_layers(base, tok, forget, retain, n_layers, device,
                                                       cap=args.sample_cap, top_k=max(1,int(args.select_top_k)),
                                                       use_anc=args.use_anc, min_layer=int(args.min_layer),
                                                       select_mode=args.select_mode, script_blind_selection=args.script_blind_selection,
                                                       xlang_sets=xlang_sets, max_len=int(args.max_len))
                # Optional judge refinement per seed
                if args.judge_assist_selection:
                    try:
                        pool_k = max(len(sel_chosen), int(args.judge_pool))
                        ranked = sorted(sel_scores.items(), key=lambda kv: kv[1]["combo"], reverse=True)[:pool_k]
                        candidates = [li for li,_ in ranked]
                        cap_hi = max(4, min(len(forget), int(args.judge_cap)//2))
                        cap_en = max(4, min(len(retain), int(args.judge_cap) - cap_hi))
                        base_hi_out = generate(base, tok, forget[:cap_hi], device)
                        base_en_out = generate(base, tok, retain[:cap_en], device)
                        base_j_hi = _judge_avg_score(base_hi_out, target_lang="Hindi", model_name=args.judge_model, timeout=args.judge_timeout) or 0.0
                        base_j_en = _judge_avg_score(base_en_out, target_lang="Hindi", model_name=args.judge_model, timeout=args.judge_timeout) or 0.0
                        deltas = _refine_layers_with_judge(base, tok, device, forget, retain, candidates,
                                                           base_hi_out, base_en_out, base_j_hi, base_j_en,
                                                           args.judge_scale, args.judge_beta, args.judge_model, args.judge_timeout)
                        # Blend
                        met = np.array([sel_scores[li]["combo"] for li in candidates], dtype=np.float32)
                        if np.max(met) > np.min(met):
                            met = (met - np.min(met)) / (np.max(met) - np.min(met))
                        else:
                            met = np.zeros_like(met)
                        jd = np.array([max(0.0, deltas.get(li, 0.0)) for li in candidates], dtype=np.float32)
                        if np.max(jd) > 0:
                            jd = jd / (np.max(jd) + 1e-9)
                        blend = (args.judge_alpha * met + (1.0 - args.judge_alpha) * jd)
                        new_rank = [li for _, li in sorted(zip(blend, candidates), key=lambda p: p[0], reverse=True)]
                        sel_chosen = new_rank[:max(1, int(args.select_top_k))]
                    except Exception:
                        pass
                # Aggregate
                if args.stability_strategy == "vote":
                    for li in sel_chosen:
                        vote[li] += 1
                else:
                    # average combo across runs; for missing layers, initialize
                    for li, sc in sel_scores.items():
                        avg_scores[li] = avg_scores.get(li, 0.0) + float(sc.get("combo", 0.0))
            if args.stability_strategy == "vote":
                top = [li for li,_ in vote.most_common(max(1,int(args.select_top_k)))]
                chosen = top
                scores = {li: {"combo": None} for li in chosen}
                print(f"[layers] stability vote: {dict(vote)} -> chosen {chosen}")
            else:
                if avg_scores:
                    for k in list(avg_scores.keys()):
                        avg_scores[k] /= float(len(seeds_list))
                    top = [li for li,_ in sorted(avg_scores.items(), key=lambda kv: kv[1], reverse=True)[:max(1,int(args.select_top_k))]]
                else:
                    top = sel_chosen
                chosen = top
                scores = {li: {"combo": avg_scores.get(li)} for li in chosen}
                print(f"[layers] stability average: chosen {chosen}")
            # Reseed to the user-provided default for downstream steps
            if args.selection_seed is not None:
                try:
                    random.seed(int(args.selection_seed)); np.random.seed(int(args.selection_seed)); torch.manual_seed(int(args.selection_seed));
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(args.selection_seed))
                except Exception:
                    pass
        else:
            chosen, scores = select_layers(base, tok, forget, retain, n_layers, device,
                                           cap=args.sample_cap, top_k=max(1,int(args.select_top_k)),
                                           use_anc=args.use_anc, min_layer=int(args.min_layer),
                                           select_mode=args.select_mode, script_blind_selection=args.script_blind_selection,
                                           xlang_sets=xlang_sets, max_len=int(args.max_len))
    if len(chosen) < int(args.select_top_k):
        print(f"[layers] warning: requested top_k={args.select_top_k} but only selected {len(chosen)} layers (min_layer={args.min_layer}, n_layers={n_layers}).")
    if args.print_layer_scores:
        try:
            print(f"[layers] ranking (top 10 by combo) — mode={args.select_mode}, script_blind={args.script_blind_selection}:")
            rank = sorted(scores.items(), key=lambda kv: kv[1]["combo"], reverse=True)[:10]
            for li, sc in rank:
                print(f"  L{li:02d}  combo={sc['combo']:.3f}  CKA={sc['cka']:.3f}  Proc={sc['proc']:.3f}  Cos={sc['cos']:.3f}  ANC={sc['anc']:.3f}")
            print(f"[layers] chosen: {sorted(chosen)}  (min_layer={args.min_layer}, top_k={args.select_top_k}, use_anc={args.use_anc})")
        except Exception as e:
            print(f"[layers] print failed: {e}")

    # Optional: refine with LLM judge on a small candidate pool
    if args.judge_assist_selection:
        try:
            pool_k = max(len(chosen), int(args.judge_pool))
            ranked = sorted(scores.items(), key=lambda kv: kv[1]["combo"], reverse=True)[:pool_k]
            candidates = [li for li,_ in ranked]
            # Base outputs for a small slice
            cap_hi = max(4, min(len(forget), args.judge_cap//2))
            cap_en = max(4, min(len(retain), args.judge_cap - cap_hi))
            base_hi_out = generate(base, tok, forget[:cap_hi], device)
            base_en_out = generate(base, tok, retain[:cap_en], device)
            base_j_hi = _judge_avg_score(base_hi_out, target_lang="Hindi", model_name=args.judge_model, timeout=args.judge_timeout) or 0.0
            base_j_en = _judge_avg_score(base_en_out, target_lang="Hindi", model_name=args.judge_model, timeout=args.judge_timeout) or 0.0
            deltas = _refine_layers_with_judge(base, tok, device, forget, retain, candidates,
                                               base_hi_out, base_en_out, base_j_hi, base_j_en,
                                               args.judge_scale, args.judge_beta, args.judge_model, args.judge_timeout)
            # Blend scores
            # Normalize both to [0,1] for blending
            met = np.array([scores[li]["combo"] for li in candidates], dtype=np.float32)
            if np.max(met) > np.min(met):
                met = (met - np.min(met)) / (np.max(met) - np.min(met))
            else:
                met = np.zeros_like(met)
            jd = np.array([max(0.0, deltas.get(li, 0.0)) for li in candidates], dtype=np.float32)
            if np.max(jd) > 0:
                jd = jd / (np.max(jd) + 1e-9)
            blend = (args.judge_alpha * met + (1.0 - args.judge_alpha) * jd)
            new_rank = [li for _, li in sorted(zip(blend, candidates), key=lambda p: p[0], reverse=True)]
            chosen = new_rank[:max(1, int(args.select_top_k))]
            print(f"[judge] refined chosen: {sorted(chosen)} (alpha={args.judge_alpha}, beta={args.judge_beta}, scale={args.judge_scale})")
        except Exception as e:
            print(f"[judge] assist skipped: {e}")

    # SAEs (train or load)
    sae_info={}
    pool=forget+retain
    os.makedirs(args.ckpt_dir, exist_ok=True)
    for li in chosen:
        sae_path=os.path.join(args.ckpt_dir, f"sae_layer{li}.pt")
        # Prefer SAELens directory if provided
        if args.sae_lens_dir and _HAS_SAELENS_LOADER:
            try:
                # Load tensors and construct a TopKSAE with same expansion/k
                E_t, D_t = _load_sae_lens(layer_id=li, d=base.config.hidden_size, expansion=args.sae_expansion, directory=args.sae_lens_dir, device=device)
                if E_t is not None and D_t is not None:
                    sae = TopKSAE(d=base.config.hidden_size, k=args.sae_k, expansion=args.sae_expansion).to(device)
                    with torch.no_grad():
                        sae.E.weight.copy_(E_t)
                        sae.D.weight.copy_(D_t)
                    sae.eval()
                    sae_info[li] = {"path": f"{args.sae_lens_dir}", "loaded": True, "source": "sae_lens"}
                    print(f"[sae] loaded SAELens weights for L{li}")
                    # Save a local copy to reuse later
                    try:
                        torch.save(sae.state_dict(), sae_path)
                    except Exception:
                        pass
                    continue
            except Exception as e:
                print(f"[sae] SAELens load failed for L{li}: {e}")
        if args.train_sae_steps>0:
            sae,info = train_sae(base,tok,pool,li,device,steps=args.train_sae_steps,seq_len=args.max_len,k=args.sae_k,expansion=args.sae_expansion)
            torch.save(sae.state_dict(),sae_path)
            sae_info[li]={"path":sae_path, **info}
            print(f"[sae] saved {sae_path}")
        else:
            if os.path.exists(sae_path):
                sae_info[li]={"path":sae_path,"loaded":True}
                print(f"[sae] loaded {sae_path}")
            else:
                sae_info[li]={"path":sae_path,"loaded":False}

    results={"layers":chosen,"layer_scores":scores,"sae_info":sae_info,"arms":{}}

    # Optional: SAE quality proxies per chosen layer
    sae_quality = {}
    if args.sae_quality_eval and chosen:
        try:
            # Use a small mixture for robustness: forget + retain
            pool_q = (forget[:args.sae_eval_cap//2] + retain[:args.sae_eval_cap//2]) or (forget[:args.sae_eval_cap])
            for li in chosen:
                sae_path=os.path.join(args.ckpt_dir, f"sae_layer{li}.pt")
                if os.path.exists(sae_path):
                    try:
                        sae = TopKSAE(d=base.config.hidden_size, k=args.sae_k, expansion=args.sae_expansion).to(device)
                        sae.load_state_dict(torch.load(sae_path, map_location=device), strict=False)
                        sae.eval()
                        q = _sae_quality(base, tok, pool_q, li, device, sae, cap_each=max(32, min(args.sae_eval_cap, 256)))
                        sae_quality[li] = q
                    except Exception as e:
                        sae_quality[li] = {"error": str(e)}
        except Exception as e:
            print(f"[sae-quality] skipped: {e}")
    if sae_quality:
        results["sae_quality"] = sae_quality

    # Base metrics
    base_es_forget = extraction_strength(generate(base,tok,forget[:200],device), lid, target_code="hi", use_script_guard=True)
    base_ppl_retain = perplexity(base,tok,retain[:200],device)
    base_es_mixed  = extraction_strength(generate(base,tok,mixed[:200],device), lid, target_code="hi", use_script_guard=True)
    base_crossling = {}
    for lname, xt in xlang_sets:
        base_crossling[lname] = extraction_strength(generate(base,tok,xt[:120],device), lid, target_code="hi", use_script_guard=True)
    base_token_kl = None
    if args.report_token_kl:
        kl_vals=[]
        for batch in chunked(retain[:120], 8):
            enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=args.max_len).to(device)
            kl_vals.append(token_kl_to_base(base, base, enc).item())
        base_token_kl = float(np.mean(kl_vals)) if kl_vals else 0.0
    base_es_rom = None
    if args.es_romanized:
        from transliteration_utils import batch_devanagari_to_latin
        rom = batch_devanagari_to_latin(generate(base,tok,forget[:200],device))
        base_es_rom = extraction_strength(rom, lid, target_code="hi", use_script_guard=False)
    # Always compute semantic ES (script-blind) for base if possible
    base_es_sem = None
    base_mix_sem = None
    try:
        _gens = generate(base,tok,forget[:200],device)
        _rom = _romanize_texts(_gens)
        base_es_sem = extraction_strength(_rom, lid, target_code="hi", use_script_guard=False)
        _gens_m = generate(base,tok,mixed[:200],device)
        _rom_m = _romanize_texts(_gens_m)
        base_mix_sem = extraction_strength(_rom_m, lid, target_code="hi", use_script_guard=False)
    except Exception as e:
        print(f"[base] semantic ES skipped: {e}")

    # Save base activations (no args, default behavior)
    out_stem = Path(args.out).stem
    base_sets = {
        "forget": forget[:200],
        "retain": retain[:200],
        "mixed": mixed[:200],
        "adversarial": adversarial[:200],
    }
    for lname, xt in xlang_sets:
        base_sets[f"xlang_{lname}"] = xt[:120]
    try:
        save_activations_for_sets(base, tok, device, chosen or list(range(n_layers))[:3], base_sets, out_stem, cap_per_set=200)
    except Exception as e:
        print(f"[acts] base activation save skipped: {e}")

    summary={"base":{"es_forget":base_es_forget,
                      "ppl_retain":base_ppl_retain,
                      "es_mixed":base_es_mixed,
                      "crossling_es":base_crossling,
                      **({"token_kl_retain":base_token_kl} if base_token_kl is not None else {})}}
    if base_es_rom is not None:
        summary["base"]["es_romanized"] = base_es_rom
    if base_es_sem is not None:
        summary["base"]["es_semantic"] = base_es_sem
    if base_mix_sem is not None:
        summary["base"]["es_mixed_semantic"] = base_mix_sem

    # Optional: precompute SAE-gate feature indices and SAE modules per chosen layer
    sae_gate_features: Dict[int, List[int]] = {}
    sae_modules: Dict[int, TopKSAE] = {}
    if args.sae_gate and chosen:
        for li in chosen:
            sae_path=os.path.join(args.ckpt_dir, f"sae_layer{li}.pt")
            if os.path.exists(sae_path):
                try:
                    sae = TopKSAE(d=base.config.hidden_size, k=args.sae_k, expansion=args.sae_expansion).to(device)
                    sae.load_state_dict(torch.load(sae_path, map_location=device), strict=False)
                    sae.eval()
                    if args.semantic_features:
                        try:
                            forget_rom = _romanize_texts(forget)
                            gib = _make_devanagari_gibberish(retain, seed=0)
                            idx = pick_semantic_sae_features(sae, base, tok, forget, forget_rom, gib, li, device, max_len=args.max_len, bs=32, cap_each=256, topk=args.sae_gate_topk, tau=args.semantic_tau)
                        except Exception as e:
                            print(f"[sae-gate] semantic picker failed on L{li}: {e}; falling back")
                            idx = pick_sae_features_forget_vs_retain(sae, base, tok, forget, retain, li, device, max_len=args.max_len, bs=32, cap_each=256, topk=args.sae_gate_topk)
                    else:
                        idx = pick_sae_features_forget_vs_retain(sae, base, tok, forget, retain, li, device, max_len=args.max_len, bs=32, cap_each=256, topk=args.sae_gate_topk)
                    sae_gate_features[li] = idx
                    sae_modules[li] = sae
                    print(f"[sae-gate] L{li}: picked {len(idx)} features")
                except Exception as e:
                    print(f"[sae-gate] skip L{li}: {e}")
            else:
                print(f"[sae-gate] no SAE file for L{li}; gating skipped")

    # Arms per seed
    for seed in args.seeds:
        set_seed(seed)

        # LoRA: train or resume
        lora=load_causal_lm(args.model, tok, device, hf_token)
        if args.lora_steps>0:
            lora=train_lora(lora,tok,forget,retain,device,steps=args.lora_steps,hf_token=hf_token, forget_obj=args.forget_obj, ckpt_dir=args.ckpt_dir)
        else:
            lora=resume_lora(lora,tok,device, ckpt_dir=args.ckpt_dir)

        # ReFT: train or resume
        reft=load_causal_lm(args.model, tok, device, hf_token)
        if args.reft_steps>0:
            reft=train_reft(reft,tok,chosen,forget,retain,device,rank=args.rank,steps=args.reft_steps,hf_token=hf_token, forget_obj=args.forget_obj, ckpt_dir=args.ckpt_dir)
        else:
            apply_reft_from_file(reft, chosen, os.path.join(args.ckpt_dir, "reft_adapters.pt"), rank=args.rank)

        for name,model in {"lora":lora,"reft":reft}.items():
            gate=None
            scrub=None
            if args.sae_gate and sae_modules:
                try:
                    gate = SAEGate(model, chosen, sae_modules, sae_gate_features, alpha=args.sae_gate_alpha)
                except Exception as e:
                    print(f"[sae-gate] attach failed: {e}")
            if args.script_scrub:
                try:
                    # Learn script directions using Hindi-Deva vs Hindi-Roman controls
                    forget_rom = _romanize_texts(forget)
                    W_per_layer = {}
                    for li in chosen:
                        W = learn_script_subspace(model, tok, forget, forget_rom, li, device, max_len=args.max_len, bs=32, cap_each=256, k=max(1,args.scrub_k))
                        if W.size > 0:
                            W_per_layer[li] = W
                    if W_per_layer:
                        scrub = LinearProjectHook(model, chosen, W_per_layer)
                        print("[scrub] attached linear projector")
                except Exception as e:
                    print(f"[scrub] attach failed: {e}")

            if args.dynamic_gate and gate is not None:
                if getattr(args, 'semantic_dynamic_gate', False):
                    gens_f = generate_with_semantic_gating(model,tok,lid,forget[:200],device,gate,base_alpha=max(0.0, args.sae_gate_alpha-0.2),high_alpha=min(1.0, args.sae_gate_alpha+0.2))
                else:
                    gens_f = generate_with_dynamic_gating(model,tok,lid,forget[:200],device,gate,base_alpha=max(0.0, args.sae_gate_alpha-0.2),high_alpha=min(1.0, args.sae_gate_alpha+0.2))
            else:
                gens_f = generate(model,tok,forget[:200],device)
            es_forget = extraction_strength(gens_f, lid, target_code="hi", use_script_guard=True)
            # Script-blind semantic ES for forget set
            es_forget_sem = None
            try:
                rom_f = _romanize_texts(gens_f)
                es_forget_sem = extraction_strength(rom_f, lid, target_code="hi", use_script_guard=False)
            except Exception:
                pass
            ppl_retain = perplexity(model,tok,retain[:200],device)
            gens_m = generate(model,tok,mixed[:150],device)
            es_mixed  = extraction_strength(gens_m, lid, target_code="hi", use_script_guard=True)
            es_mixed_sem = None
            try:
                rom_m = _romanize_texts(gens_m)
                es_mixed_sem = extraction_strength(rom_m, lid, target_code="hi", use_script_guard=False)
            except Exception:
                pass
            others=[l for l in probe_layers if l not in chosen] or probe_layers
            probes = probes_auc(model,tok,forget[:150],retain[:150],others,device)
            mia = mia_loss(base,model,tok,forget[:120],retain[:120],device)
            token_kl_mean = None
            if args.report_token_kl:
                vals=[]
                for batch in chunked(retain[:120], 8):
                    enc=tok(batch, return_tensors='pt',padding=True,truncation=True,max_length=args.max_len).to(device)
                    vals.append(token_kl_to_base(model, base, enc).item())
                token_kl_mean = float(np.mean(vals)) if vals else None
            xes={}
            for lname,xt in xlang_sets:
                xes[lname]=extraction_strength(generate(model,tok,xt[:120],device), lid, target_code="hi", use_script_guard=True)

            # Save activations for this arm (forget/retain only, to limit size)
            try:
                arm_sets = {"forget": forget[:200], "retain": retain[:200]}
                save_activations_for_sets(model, tok, device, chosen or list(range(n_layers))[:3], arm_sets, f"{out_stem}_{name}", cap_per_set=200)
            except Exception as e:
                print(f"[acts] arm {name} activation save skipped: {e}")

            results["arms"].setdefault(name,{}).setdefault("seeds",[]).append({
                "seed":seed,
                "es_forget":es_forget,
                **({"es_forget_semantic": es_forget_sem} if es_forget_sem is not None else {}),
                "ppl_retain":ppl_retain,
                "es_mixed":es_mixed,
                **({"es_mixed_semantic": es_mixed_sem} if es_mixed_sem is not None else {}),
                "probes_other_layers":probes,"mia":mia,"crosslingual_es":xes,
                **({"token_kl_retain":token_kl_mean} if token_kl_mean is not None else {})
            })
            if gate is not None:
                gate.remove()
            if scrub is not None:
                scrub.remove()

    # Aggregate & gates
    def agg(vals): return bootstrap_ci(vals,alpha=0.05,n_boot=2000,seed=0)
    for name,arm in results["arms"].items():
        es_vals=[s["es_forget"] for s in arm["seeds"]]
        es_sem_vals=[s.get("es_forget_semantic") for s in arm["seeds"] if s.get("es_forget_semantic") is not None]
        ppl_vals=[s["ppl_retain"] for s in arm["seeds"]]
        mix_vals=[s["es_mixed"] for s in arm["seeds"]]
        mix_sem_vals=[s.get("es_mixed_semantic") for s in arm["seeds"] if s.get("es_mixed_semantic") is not None]
        m_es,ci_es=agg(es_vals); m_ppl,ci_ppl=agg(ppl_vals); m_mix,ci_mix=agg(mix_vals)
        m_es_sem,ci_es_sem = (agg(es_sem_vals) if es_sem_vals else (float('nan'), (float('nan'), float('nan'))))
        m_mix_sem,ci_mix_sem = (agg(mix_sem_vals) if mix_sem_vals else (float('nan'), (float('nan'), float('nan'))))
        aucs=[]
        for s in arm["seeds"]:
            for _,pr in s["probes_other_layers"].items(): aucs.append(pr["auc"])
        redist_flag = (len(aucs)>0 and float(np.mean(aucs))>=0.55)
        cl_deltas=[]
        for s in arm["seeds"]:
            for lname, v in s["crosslingual_es"].items():
                base_x = base_crossling.get(lname)
                if base_x is not None:
                    cl_deltas.append(v - base_x)
        xleak = (len(cl_deltas)>0 and float(np.mean(cl_deltas))>0.10)
        mia_auc=float(np.mean([s["mia"]["AUC"] for s in arm["seeds"]]))
        mia_acc=float(np.mean([s["mia"]["ACC"] for s in arm["seeds"]]))
        mia_ok = (abs(mia_auc-0.5)<=0.05 and abs(mia_acc-0.5)<=0.05)
        tkl_vals=[s.get("token_kl_retain") for s in arm["seeds"] if s.get("token_kl_retain") is not None]
        tkl_mean=float(np.mean(tkl_vals)) if tkl_vals else None

        summary[name]={
            "es_forget_mean":m_es,"es_forget_ci":ci_es,
            **({"es_forget_semantic_mean": m_es_sem, "es_forget_semantic_ci": ci_es_sem} if es_sem_vals else {}),
            "ppl_retain_mean":m_ppl,"ppl_retain_ci":ci_ppl,
            "es_mixed_mean":m_mix,"es_mixed_ci":ci_mix,
            **({"es_mixed_semantic_mean": m_mix_sem, "es_mixed_semantic_ci": ci_mix_sem} if mix_sem_vals else {}),
            "redistribution_flag":redist_flag,
            "crosslingual_leakage_flag":xleak,
            "mia":{"AUC_mean":mia_auc,"ACC_mean":mia_acc,"ok":mia_ok},
            **({"token_kl_retain_mean":tkl_mean} if tkl_mean is not None else {}),
        }

    def gate(arm):
        if arm not in summary: return {}
        base_es=summary["base"]["es_forget"]; base_mix=summary["base"]["es_mixed"]; base_ppl=summary["base"]["ppl_retain"]
        base_es_sem = summary["base"].get("es_semantic")
        base_mix_sem = summary["base"].get("es_mixed_semantic")
        es_ok = (summary[arm]["es_forget_mean"] <= (args.gate_es_forget_ratio*base_es if base_es>0 else 0.1))
        es_sem_ok = True
        if base_es_sem is not None and ("es_forget_semantic_mean" in summary[arm]):
            es_sem_ok = (summary[arm]["es_forget_semantic_mean"] <= (args.gate_es_forget_ratio*base_es_sem if base_es_sem>0 else 0.1))
        ppl_ok = (summary[arm]["ppl_retain_mean"] / (base_ppl+1e-9) <= args.gate_ppl_ratio)
        mix_ok = (summary[arm]["es_mixed_mean"] <= (args.gate_es_mixed_ratio*base_mix if base_mix>0 else 0.1))
        mix_sem_ok = True
        if base_mix_sem is not None and ("es_mixed_semantic_mean" in summary[arm]):
            mix_sem_ok = (summary[arm]["es_mixed_semantic_mean"] <= (args.gate_es_mixed_ratio*base_mix_sem if base_mix_sem>0 else 0.1))
        red_ok = (not summary[arm]["redistribution_flag"])
        xlk_ok = (not summary[arm]["crosslingual_leakage_flag"])
        mia_ok = summary[arm]["mia"]["ok"]
        return {"G1_ES50":es_ok,
                "G1S_ES50_sem": es_sem_ok,
                "G2_PPL10":ppl_ok,
                "G3_MIX30":mix_ok,
                "G3S_MIX30_sem": mix_sem_ok,
                "G4_NoRedistrib":red_ok,
                "G5_NoXLeak":xlk_ok,
                "G6_MIA0":mia_ok}

    gates={arm:gate(arm) for arm in ["lora","reft"] if arm in summary}
    decisions={arm:("PROCEED" if all(gates[arm].values()) else "STOP") for arm in gates}

    payload={
        "model": args.model,
        "ckpt_dir": args.ckpt_dir,
        "chosen_layers":results["layers"],
        "layer_scores":results["layer_scores"],
        "sae_info":results["sae_info"],
        **({"sae_quality": results.get("sae_quality")} if results.get("sae_quality") else {}),
        "summary":summary,
        "gates":gates,
        "decision":decisions,
        "notes":{
            "lid_backends":"langid + script-guard",
            "sae":"TopK SAE (fp32) with k/expansion configurable; optional SAE-gating",
            "threat_model":"MMIE gates catch redistribution, cross-ling leakage, privacy via MIA"
        }
    }
    with open(args.out,"w",encoding="utf-8") as f: json.dump(_sanitize_json(payload),f,indent=2,ensure_ascii=False)
    print(json.dumps({"decision":decisions,"gates":gates},indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
