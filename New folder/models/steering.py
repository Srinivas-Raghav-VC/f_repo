"""
Steering Utilities
==================

Functions for computing and applying steering directions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable


def compute_steering_direction(
    source_activations: torch.Tensor,
    target_activations: torch.Tensor,
    method: str = "mean_diff",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute steering direction from source to target.
    
    Args:
        source_activations: Activations from source distribution [N, hidden_dim]
        target_activations: Activations from target distribution [M, hidden_dim]
        method: "mean_diff" (default), "pca", "lda"
        normalize: Whether to normalize the direction
    
    Returns:
        Steering direction [hidden_dim]
    """
    if method == "mean_diff":
        source_mean = source_activations.mean(dim=0)
        target_mean = target_activations.mean(dim=0)
        direction = target_mean - source_mean
        
    elif method == "pca":
        # Use first PC of the difference
        combined = torch.cat([source_activations, target_activations], dim=0)
        combined = combined - combined.mean(dim=0)
        _, _, V = torch.svd(combined)
        direction = V[:, 0]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if normalize:
        direction = F.normalize(direction, dim=0)
    
    return direction


def apply_steering(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
    coefficient: float,
    position: str = "last",
) -> torch.Tensor:
    """
    Apply steering direction to hidden states.
    
    Args:
        hidden_states: Hidden states [batch, seq_len, hidden_dim]
        direction: Steering direction [hidden_dim]
        coefficient: Scaling factor
        position: "last" (last token only), "all" (all positions)
    
    Returns:
        Modified hidden states
    """
    if position == "last":
        hidden_states[:, -1, :] = hidden_states[:, -1, :] + coefficient * direction
    elif position == "all":
        hidden_states = hidden_states + coefficient * direction.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unknown position: {position}")
    
    return hidden_states


def create_steering_hook(
    direction: torch.Tensor,
    coefficient: float,
    position: str = "last",
) -> Callable:
    """
    Create a forward hook for steering.
    
    Args:
        direction: Steering direction
        coefficient: Scaling factor
        position: Where to apply steering
    
    Returns:
        Hook function
    """
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        if position == "last":
            hidden[:, -1, :] = hidden[:, -1, :] + coefficient * direction.to(hidden.device)
        elif position == "all":
            hidden = hidden + coefficient * direction.to(hidden.device).unsqueeze(0).unsqueeze(0)
        
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden
    
    return hook


class SteeringManager:
    """Manage steering interventions during generation."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
    
    def add_steering(
        self,
        layer_idx: int,
        direction: torch.Tensor,
        coefficient: float,
        position: str = "last",
    ):
        """Add steering to a layer."""
        layer = self.model.model.layers[layer_idx]
        hook = create_steering_hook(direction, coefficient, position)
        handle = layer.register_forward_hook(hook)
        self.hooks.append(handle)
    
    def clear(self):
        """Remove all steering hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.clear()