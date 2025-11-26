"""
Sparse Autoencoder Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple


class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, threshold: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.kaiming_normal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = self.encoder(x)
        return F.relu(pre_act - self.threshold) * (pre_act > self.threshold).float()
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, k: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.kaiming_normal_(self.decoder.weight)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_act = F.relu(self.encoder(x))
        
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


def train_sae(
    activations: torch.Tensor,
    hidden_dim: int = 4096,
    sae_dim: int = 16384,
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 32,
    k: int = 64,
    threshold: float = 0.05,
    sae_type: str = "jumprelu",
) -> Tuple[nn.Module, Dict]:
    """Train SAE on activations."""
    
    device = activations.device
    
    # Train/val split
    n = len(activations)
    n_val = max(1, int(n * 0.1))
    perm = torch.randperm(n)
    train_acts = activations[perm[n_val:]]
    val_acts = activations[perm[:n_val]]
    
    # Create SAE
    if sae_type == "jumprelu":
        sae = JumpReLUSAE(hidden_dim, sae_dim, threshold=threshold)
    else:
        sae = TopKSAE(hidden_dim, sae_dim, k=k)
    
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Training
    train_loader = DataLoader(TensorDataset(train_acts), batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        sae.train()
        for (batch,) in train_loader:
            optimizer.zero_grad()
            x_hat, z = sae(batch)
            loss = F.mse_loss(x_hat, batch)
            loss.backward()
            optimizer.step()
    
    # Validation
    sae.eval()
    with torch.no_grad():
        val_x_hat, val_z = sae(val_acts)
        val_loss = F.mse_loss(val_x_hat, val_acts).item()
        l0 = (val_z > 0).float().sum(dim=-1).mean().item()
    
    metrics = {
        "val_loss": val_loss,
        "l0": l0,
        "sparsity": (val_z > 0).float().mean().item(),
    }
    
    return sae, metrics
