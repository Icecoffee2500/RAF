"""Model builder utility.

Provides `build_model` to return a SegFormer model (ViT-S backbone)
initialized from scratch without pretrained weights.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig  # type: ignore

_SEGFORMER_CONFIG = "nvidia/segformer-b0-finetuned-ade-512-512"  # For architecture only


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


def build_model(num_classes: int, device_id: int | str | None = None, *, pretrained_encoder: bool = False) -> Tuple[SegformerForSemanticSegmentation, torch.device]:
    """Build SegFormer-B0 model initialized from scratch (no pretrained weights).

    Args:
        num_classes: Number of segmentation classes.
        device_id: GPU device ID, device string, or None for auto-selection

    Returns:
        (model, device)
    """
    print(f"🏗️  Building SegFormer-B0 from scratch with {num_classes} classes")
    print(f"🔧 Architecture config from: {_SEGFORMER_CONFIG}")
    if pretrained_encoder:
        print("🔄 Loading ImageNet-pretrained encoder weights (SegFormer MiT)")
    else:
        print("🆕 All weights: initialized from scratch (no pretrained)")
    print(f"🎯 Target classes: {num_classes} (Cityscapes semantic segmentation)")
    
    # Load configuration only (not weights)
    config = SegformerConfig.from_pretrained(_SEGFORMER_CONFIG, num_labels=num_classes)
    
    if pretrained_encoder:
        model = SegformerForSemanticSegmentation.from_pretrained(
            _SEGFORMER_CONFIG,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        # Create model with random initialization
        model = SegformerForSemanticSegmentation(config)
    
    # Verify no pretrained weights were loaded
    assert isinstance(model, SegformerForSemanticSegmentation)
    
    device = _get_device(device_id)
    model = model.to(device)
    
    # Print GPU memory info if using GPU
    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3  # GB
        print(f"📊 GPU Memory: {memory_allocated:.1f}GB / {memory_total:.1f}GB allocated")
    
    return model, device 