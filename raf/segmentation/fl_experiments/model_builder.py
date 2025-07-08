"""Model builder utility.

Provides `build_model` to return a SegFormer model (ViT-S backbone)
initialized with MAE-pretrained weights from HuggingFace.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig  # type: ignore

_HF_WEIGHTS = "nvidia/segformer-b0-finetuned-ade-512-512"  # ViT-S backbone (B0) pretrained by MAE


def _get_device(device_id: int | str | None = None) -> torch.device:
    """Get PyTorch device with optional GPU selection.
    
    Args:
        device_id: GPU device ID (int), device string ('cuda:0'), or None for auto-selection
        
    Returns:
        torch.device object
    """
    if device_id is None:
        # Auto-selection: prefer CUDA_VISIBLE_DEVICES or default GPU
        if torch.cuda.is_available():
            # Check if CUDA_VISIBLE_DEVICES is set
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
            if visible_devices:
                # Use first visible device
                device = torch.device("cuda:0")
                print(f"🎯 Using GPU from CUDA_VISIBLE_DEVICES: {visible_devices}")
            else:
                # Use default GPU
                device = torch.device("cuda")
                print(f"🎯 Using default GPU: cuda:0")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU (no GPU available)")
    elif isinstance(device_id, str):
        # Device string like 'cuda:1', 'cpu'
        device = torch.device(device_id)
        print(f"🎯 Using specified device: {device_id}")
    elif isinstance(device_id, int):
        # GPU ID number
        if torch.cuda.is_available():
            if device_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{device_id}")
                print(f"🎯 Using GPU {device_id}: {torch.cuda.get_device_name(device_id)}")
            else:
                print(f"⚠️  GPU {device_id} not available (only {torch.cuda.device_count()} GPUs found)")
                print(f"🔄 Falling back to GPU 0: {torch.cuda.get_device_name(0)}")
                device = torch.device("cuda:0")
        else:
            print(f"⚠️  GPU {device_id} not available (CUDA not available)")
            print("🔄 Falling back to CPU")
            device = torch.device("cpu")
    else:
        print("⚠️  Invalid device_id, using auto-selection")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device


def build_model(num_classes: int, device_id: int | str | None = None) -> Tuple[SegformerForSemanticSegmentation, torch.device]:
    """Build SegFormer-B0 (ViT-S) model with correct num_classes.

    Args:
        num_classes: Number of segmentation classes.
        device_id: GPU device ID, device string, or None for auto-selection

    Returns:
        (model, device)
    """

    cfg = SegformerConfig.from_pretrained(_HF_WEIGHTS, num_labels=num_classes)
    model = SegformerForSemanticSegmentation.from_pretrained(
        _HF_WEIGHTS,
        config=cfg,
        ignore_mismatched_sizes=True,  # classifier(150→num_classes) 크기 무시하고 로드
    )
    assert isinstance(model, SegformerForSemanticSegmentation)
    device = _get_device(device_id)
    model.to(device)
    
    # Print GPU memory info if using GPU
    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
        print(f"📊 GPU Memory: {memory_allocated:.1f}GB / {memory_total:.1f}GB allocated")
    
    return model, device 