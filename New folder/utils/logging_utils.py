"""
Logging Utilities
=================
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_logger_initialized = False


def setup_logging(
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        verbose: Enable verbose (DEBUG) logging
    """
    global _logger_initialized
    
    if _logger_initialized:
        return
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Format
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    # Handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(fmt, date_fmt))
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always verbose in file
        file_handler.setFormatter(logging.Formatter(fmt, date_fmt))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("tokenizers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    _logger_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
