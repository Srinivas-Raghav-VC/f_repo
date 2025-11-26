"""
Model Manager
=============
Handles model loading, activation collection, and steering.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
import os


class ModelManager:
    """Manages model, tokenizer, and all interventions."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_layers = None
        self.hidden_dim = None
    
    def load(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        token = os.environ.get("HF_TOKEN")
        
        print(f"  Loading {self.config.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            token=token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token
        )
        self.model.eval()
        
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size
        
        print(f"  ✓ Loaded: {self.num_layers} layers, {self.hidden_dim} hidden dim")
    
    def get_activation(self, text: str, layer_idx: int) -> torch.Tensor:
        """Get activation at specific layer for last token."""
        activation = None
        
        def hook(module, input, output):
            nonlocal activation
            hidden = output[0] if isinstance(output, tuple) else output
            activation = hidden[:, -1, :].detach().float()
        
        handle = self.model.model.layers[layer_idx].register_forward_hook(hook)
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        handle.remove()
        return activation.squeeze(0)
    
    def get_all_layer_activations(self, text: str) -> Dict[int, torch.Tensor]:
        """Get activations from all layers."""
        activations = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = hidden[:, -1, :].detach().float()
            return hook
        
        for layer_idx in range(self.num_layers):
            handle = self.model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(handle)
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        for h in hooks:
            h.remove()
        
        return {k: v.squeeze(0) for k, v in activations.items()}
    
    def compute_steering_direction(
        self, 
        source_texts: List[str], 
        target_texts: List[str],
        layer_idx: int,
        normalize: bool = True
    ) -> torch.Tensor:
        """Compute steering direction from source to target."""
        
        # Collect activations
        source_acts = torch.stack([
            self.get_activation(t, layer_idx) for t in source_texts
        ])
        target_acts = torch.stack([
            self.get_activation(t, layer_idx) for t in target_texts
        ])
        
        # Compute means
        source_mean = source_acts.mean(dim=0)
        target_mean = target_acts.mean(dim=0)
        
        # Direction
        direction = target_mean - source_mean
        
        if normalize:
            direction = F.normalize(direction, dim=0)
        
        return direction
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 50,
        steering_layer: Optional[int] = None,
        steering_direction: Optional[torch.Tensor] = None,
        steering_coeff: float = 0.0,
        patch_activation: Optional[torch.Tensor] = None,
        patch_layer: Optional[int] = None,
    ) -> str:
        """Generate text with optional steering or patching."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        hooks = []
        
        # Add steering hook
        if steering_layer is not None and steering_direction is not None and steering_coeff > 0:
            def steering_hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[:, -1, :] += steering_coeff * steering_direction.to(hidden.device)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            
            handle = self.model.model.layers[steering_layer].register_forward_hook(steering_hook)
            hooks.append(handle)
        
        # Add patching hook
        if patch_layer is not None and patch_activation is not None:
            def patch_hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden[:, -1, :] = patch_activation.to(hidden.device)
                return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
            
            handle = self.model.model.layers[patch_layer].register_forward_hook(patch_hook)
            hooks.append(handle)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        finally:
            for h in hooks:
                h.remove()
        
        return generated


def compute_hindi_score(text: str) -> float:
    """Fraction of characters that are Devanagari (Hindi script)."""
    if not text:
        return 0.0
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total = len(text.replace(' ', '').replace('\n', ''))
    return hindi_chars / max(total, 1)


def compute_urdu_score(text: str) -> float:
    """Fraction of characters that are Arabic script (Urdu)."""
    if not text:
        return 0.0
    urdu_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total = len(text.replace(' ', '').replace('\n', ''))
    return urdu_chars / max(total, 1)


def compute_bengali_score(text: str) -> float:
    """Fraction of characters that are Bengali script."""
    if not text:
        return 0.0
    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    total = len(text.replace(' ', '').replace('\n', ''))
    return bengali_chars / max(total, 1)


def compute_punjabi_score(text: str) -> float:
    """Fraction of characters that are Gurmukhi (Punjabi) script."""
    if not text:
        return 0.0
    punjabi_chars = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')
    total = len(text.replace(' ', '').replace('\n', ''))
    return punjabi_chars / max(total, 1)


def compute_english_score(text: str) -> float:
    """Fraction of characters that are ASCII letters."""
    if not text:
        return 0.0
    ascii_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total = len(text.replace(' ', '').replace('\n', ''))
    return ascii_chars / max(total, 1)


def compute_all_language_scores(text: str) -> Dict[str, float]:
    """Compute scores for all languages."""
    return {
        "hindi": compute_hindi_score(text),
        "urdu": compute_urdu_score(text),
        "bengali": compute_bengali_score(text),
        "punjabi": compute_punjabi_score(text),
        "english": compute_english_score(text),
    }
