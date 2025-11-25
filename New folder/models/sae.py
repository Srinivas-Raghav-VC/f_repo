"""
Sparse Autoencoder Implementations
===================================

JumpReLU and TopK SAE architectures for interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import numpy as np

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class JumpReLUSAE(nn.Module):
    """
    JumpReLU Sparse Autoencoder.
    
    Uses a threshold-based activation that is differentiable via
    straight-through estimator.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        threshold: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder (tied weights optional, using separate here)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        # Initialize
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.kaiming_normal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        pre_act = self.encoder(x)
        # JumpReLU: ReLU with threshold
        return F.relu(pre_act - self.threshold) * (pre_act > self.threshold).float()
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def get_sparsity(self, z: torch.Tensor) -> float:
        """Compute sparsity (fraction of non-zero activations)."""
        return (z > 0).float().mean().item()
    
    def get_l0(self, z: torch.Tensor) -> float:
        """Compute L0 (average number of active features)."""
        return (z > 0).float().sum(dim=-1).mean().item()


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder.
    
    Forces exactly K features to be active.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.kaiming_normal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with TopK selection."""
        pre_act = self.encoder(x)
        pre_act = F.relu(pre_act)
        
        # Keep only top-k
        if self.k < self.hidden_dim:
            topk_vals, topk_idx = torch.topk(pre_act, self.k, dim=-1)
            mask = torch.zeros_like(pre_act)
            mask.scatter_(-1, topk_idx, 1.0)
            return pre_act * mask
        return pre_act
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    
    def get_sparsity(self, z: torch.Tensor) -> float:
        return (z > 0).float().mean().item()
    
    def get_l0(self, z: torch.Tensor) -> float:
        return (z > 0).float().sum(dim=-1).mean().item()


def train_sae(
    activations: torch.Tensor,
    sae_type: str = "jumprelu",
    hidden_dim: int = 4096,
    sae_dim: int = 16384,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cuda",
    k: int = 64,
    threshold: float = 0.05,
    val_split: float = 0.1,
) -> Tuple[nn.Module, Dict]:
    """
    Train a sparse autoencoder on activations.
    
    Args:
        activations: Tensor of shape [N, hidden_dim]
        sae_type: "jumprelu" or "topk"
        hidden_dim: Input dimension (model hidden size)
        sae_dim: SAE hidden dimension
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        device: Device
        k: K for TopK SAE
        threshold: Threshold for JumpReLU
        val_split: Validation split fraction
    
    Returns:
        Tuple of (trained SAE, metrics dict)
    """
    logger.info(f"Training {sae_type} SAE: input={hidden_dim}, hidden={sae_dim}")
    
    # Move to device
    activations = activations.to(device)
    
    # Split train/val
    n = len(activations)
    n_val = int(n * val_split)
    perm = torch.randperm(n)
    train_acts = activations[perm[n_val:]]
    val_acts = activations[perm[:n_val]]
    
    # Create SAE
    if sae_type == "jumprelu":
        sae = JumpReLUSAE(hidden_dim, sae_dim, threshold=threshold)
    elif sae_type == "topk":
        sae = TopKSAE(hidden_dim, sae_dim, k=k)
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")
    
    sae = sae.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Training loop
    train_loader = DataLoader(
        TensorDataset(train_acts),
        batch_size=batch_size,
        shuffle=True
    )
    
    best_val_loss = float('inf')
    metrics = {"train_losses": [], "val_losses": [], "l0s": [], "sparsities": []}
    
    for epoch in range(epochs):
        sae.train()
        epoch_loss = 0
        
        for (batch,) in train_loader:
            optimizer.zero_grad()
            
            x_hat, z = sae(batch)
            
            # Reconstruction loss
            loss = F.mse_loss(x_hat, batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        sae.eval()
        with torch.no_grad():
            val_x_hat, val_z = sae(val_acts)
            val_loss = F.mse_loss(val_x_hat, val_acts).item()
            l0 = sae.get_l0(val_z)
            sparsity = sae.get_sparsity(val_z)
        
        metrics["train_losses"].append(epoch_loss / len(train_loader))
        metrics["val_losses"].append(val_loss)
        metrics["l0s"].append(l0)
        metrics["sparsities"].append(sparsity)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if epoch % max(1, epochs // 3) == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: val_loss={val_loss:.4f}, L0={l0:.1f}")
    
    sae.eval()
    
    # Final metrics
    with torch.no_grad():
        _, final_z = sae(val_acts)
        final_l0 = sae.get_l0(final_z)
        final_sparsity = sae.get_sparsity(final_z)
    
    metrics["val_loss"] = best_val_loss
    metrics["l0"] = final_l0
    metrics["sparsity"] = final_sparsity
    
    return sae, metrics
