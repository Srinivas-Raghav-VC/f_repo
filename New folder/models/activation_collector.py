"""
Activation Collection Utilities
================================

Collect activations from specific layers of the model.
"""

import torch
from typing import Optional, List, Dict, Any


class ActivationCollector:
    """
    Collect activations from a specific layer.
    """
    
    def __init__(self, model, layer_idx: int):
        """
        Args:
            model: The language model
            layer_idx: Index of layer to collect from
        """
        self.model = model
        self.layer_idx = layer_idx
        self.activation = None
        self.hook_handle = None
    
    def _hook_fn(self, module, input, output):
        """Hook function to capture activations."""
        # output is typically (hidden_states, ...) tuple
        hidden = output[0] if isinstance(output, tuple) else output
        # Store activation of last token
        self.activation = hidden[:, -1, :].detach().clone()
    
    def get_activation(
        self,
        text: str,
        tokenizer,
        max_length: int = 128,
    ) -> torch.Tensor:
        """
        Get activation for a single text input.
        
        Args:
            text: Input text
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        
        Returns:
            Activation tensor of shape [hidden_dim]
        """
        # Register hook
        layer = self.model.model.layers[self.layer_idx]
        handle = layer.register_forward_hook(self._hook_fn)
        
        try:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.model.device)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(**inputs, output_hidden_states=False)
            
            # Return activation
            return self.activation.squeeze(0)
            
        finally:
            handle.remove()
    
    def get_activations_batch(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 128,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Get activations for multiple texts.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer
            max_length: Maximum sequence length
            batch_size: Batch size for processing
        
        Returns:
            Activations tensor of shape [N, hidden_dim]
        """
        all_activations = []
        
        # Register hook once
        layer = self.model.model.layers[self.layer_idx]
        activations_list = []
        
        def batch_hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Last token activations for each item in batch
            activations_list.append(hidden[:, -1, :].detach().clone())
        
        handle = layer.register_forward_hook(batch_hook)
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                ).to(self.model.device)
                
                with torch.no_grad():
                    _ = self.model(**inputs, output_hidden_states=False)
            
            # Concatenate all activations
            return torch.cat(activations_list, dim=0)
            
        finally:
            handle.remove()


class MultiLayerActivationCollector:
    """Collect activations from multiple layers simultaneously."""
    
    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.activations = {}
    
    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden[:, -1, :].detach().clone()
        return hook_fn
    
    def get_activations(
        self,
        text: str,
        tokenizer,
        max_length: int = 128,
    ) -> Dict[int, torch.Tensor]:
        """Get activations from all specified layers."""
        
        self.activations = {}
        handles = []
        
        try:
            # Register hooks for all layers
            for layer_idx in self.layer_indices:
                layer = self.model.model.layers[layer_idx]
                handle = layer.register_forward_hook(self._make_hook(layer_idx))
                handles.append(handle)
            
            # Forward pass
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(self.model.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Return copy of activations
            return {k: v.squeeze(0) for k, v in self.activations.items()}
            
        finally:
            for handle in handles:
                handle.remove()
