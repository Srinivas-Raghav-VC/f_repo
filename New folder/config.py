"""
Configuration Module for MMIE Research
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass
class Config:
    """Research configuration."""
    
    # Data paths
    data_dir: str = "/mnt/user-data/uploads"
    output_dir: str = "results"
    
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_dtype: str = "bfloat16"
    
    # Layers to analyze
    layers: List[int] = field(default_factory=lambda: [7, 13, 19, 22])
    
    # SAE settings
    sae_type: str = "jumprelu"  # jumprelu, topk, both
    sae_dim: int = 4096
    sae_expansion: int = 4
    sae_k: int = 64  # for TopK
    sae_threshold: float = 0.05  # for JumpReLU
    sae_epochs: int = 5
    sae_lr: float = 1e-3
    
    # Steering settings
    steering_coeffs: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    steering_method: str = "add"  # add, subtract
    
    # Data settings
    num_samples: int = 100
    max_length: int = 128
    batch_size: int = 4
    
    # Clustering settings
    num_clusters: int = 5
    
    # Computation
    seed: int = 42
    device: str = "cuda"
    quick_test: bool = False
    
    # Data file names
    forget_file: str = "forget_hindi.jsonl"
    retain_file: str = "retain_english.jsonl"
    mixed_file: str = "mixed_hinglish.jsonl"
    adversarial_file: str = "adversarial.jsonl"
    urdu_file: str = "urdu_test.jsonl"
    punjabi_file: str = "punjabi_test.jsonl"
    bengali_file: str = "bengali_test.jsonl"
    
    def __post_init__(self):
        """Validate and adjust config."""
        if self.quick_test:
            self.num_samples = min(self.num_samples, 20)
            self.sae_epochs = 2
            self.layers = self.layers[:2] if len(self.layers) > 2 else self.layers
    
    def save(self, path: str):
        """Save config to JSON."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path to data file."""
        return Path(self.data_dir) / filename
    
    @property
    def hidden_dim(self) -> int:
        """Model hidden dimension (for Llama 3.1 8B)."""
        return 4096
    
    @property
    def sae_hidden_dim(self) -> int:
        """SAE hidden dimension."""
        return self.hidden_dim * self.sae_expansion


# Default data files expected
DATA_FILES = {
    "forget": "forget_hindi.jsonl",
    "retain": "retain_english.jsonl", 
    "mixed": "mixed_hinglish.jsonl",
    "adversarial": "adversarial.jsonl",
    "urdu": "urdu_test.jsonl",
    "punjabi": "punjabi_test.jsonl",
    "bengali": "bengali_test.jsonl",
}
