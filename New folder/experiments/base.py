"""
Base Experiment Class
=====================
Common functionality for all experiments.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseExperiment(ABC):
    """Base class for experiments."""
    
    def __init__(self, config, results_manager):
        self.config = config
        self.results_manager = results_manager
        self.model = None
        self.tokenizer = None
        self.device = config.device
        
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results."""
        pass
    
    def load_data(self, filename: str) -> List[Dict]:
        """Load JSONL data file."""
        path = self.config.get_data_path(filename)
        data = []
        
        if not path.exists():
            logger.warning(f"Data file not found: {path}")
            return data
            
        with open(path) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} samples from {filename}")
        return data
    
    def compute_hindi_score(self, text: str) -> float:
        """
        Compute Hindi/Devanagari script ratio in text.
        Returns: fraction of characters that are Devanagari (0-1)
        """
        if not text:
            return 0.0
        
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_chars = len(text.replace(' ', ''))
        
        return hindi_chars / max(total_chars, 1)
    
    def compute_english_score(self, text: str) -> float:
        """Compute English/ASCII ratio in text."""
        if not text:
            return 0.0
            
        ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = len(text.replace(' ', ''))
        
        return ascii_letters / max(total_chars, 1)
    
    def set_seed(self, seed: Optional[int] = None):
        """Set random seeds for reproducibility."""
        seed = seed or self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "n": len(values)
    }
