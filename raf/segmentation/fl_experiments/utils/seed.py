"""Reproducibility utilities for fixing random seeds."""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    print(f"🌱 Setting random seed to {seed} for reproducibility")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set torch number of threads to 1 for deterministic behavior
    torch.set_num_threads(1)
    
    print("✅ Random seed fixed for reproducible results")


def get_generator(seed: int = 42) -> torch.Generator:
    """Get a PyTorch generator with fixed seed.
    
    Args:
        seed: Random seed value
        
    Returns:
        PyTorch generator with fixed seed
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator 