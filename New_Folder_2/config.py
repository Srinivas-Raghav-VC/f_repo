"""
Configuration and Environment Setup
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """Research configuration."""
    
    # Paths
    data_dir: str = "."
    output_dir: str = "./results"
    
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Experiment parameters
    layers: List[int] = field(default_factory=lambda: [7, 10, 13, 16, 19, 22])
    coefficients: List[float] = field(default_factory=lambda: [1.0, 2.0, 5.0, 10.0, 20.0])
    
    # Sample sizes
    n_samples_per_lang: int = 50
    n_parallel_sentences: int = 15
    n_test_prompts: int = 20
    
    # Quick mode overrides
    quick_mode: bool = False
    
    # Reproducibility
    seed: int = 42
    
    # SAE parameters
    sae_hidden_dim: int = 16384
    sae_epochs: int = 5
    sae_lr: float = 1e-3
    sae_k: int = 64
    sae_threshold: float = 0.05
    
    def __post_init__(self):
        if self.quick_mode:
            self.n_samples_per_lang = 15
            self.n_parallel_sentences = 10
            self.n_test_prompts = 5
            self.layers = self.layers[:3]
            self.coefficients = [2.0, 5.0, 10.0]
            self.sae_epochs = 2
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def setup_environment():
    """Setup environment variables and check dependencies."""
    
    # Check HF token
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set. Model download may fail.")
    
    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Warning: CUDA not available. Running on CPU will be slow.")
    
    return token
