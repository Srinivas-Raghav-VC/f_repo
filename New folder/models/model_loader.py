"""
Model Loading Utilities
=======================
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    cache_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with appropriate settings.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        dtype: Data type (bfloat16, float16, float32)
        cache_dir: Optional cache directory
    
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Set padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for generation
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device == "auto" else None,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    if device != "auto":
        model = model.to(device)
    
    model.eval()
    
    logger.info(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} hidden dim")
    
    return model, tokenizer


def get_model_info(model) -> dict:
    """Get model configuration info."""
    return {
        "num_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_heads": model.config.num_attention_heads,
        "vocab_size": model.config.vocab_size,
        "model_type": model.config.model_type,
    }
