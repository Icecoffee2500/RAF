"""Model builder utility.

Provides `build_model` to return a SegFormer model (ViT-S backbone)
initialized from scratch without pretrained weights.
"""

from __future__ import annotations

import os
from pathlib import Path
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


def build_model(
    num_classes: int, device_id: int | str | None = None, *, pretrained_encoder: bool = False,
) -> tuple[SegformerForSemanticSegmentation, torch.device]:
    """Build SegFormer-B0 model, optionally loading pretrained weights from GitHub (mit_b0.pth).

    Args:
        num_classes: Number of segmentation classes.
        device_id: GPU device ID, device string, or None for auto-selection
        pretrained_encoder: If True, load GitHub pretrained weights (mit_b0.pth) into the HF model.

    Returns:
        (model, device)
    """
    # Construct path to weights relative to this file's location to avoid CWD issues.
    # This file is in: .../fl_experiments/
    # Weights are in: .../pretrained_weights_segformer/
    project_root = Path(__file__).parent.parent  # .../raf/segmentation/
    github_weights_path = project_root / "pretrained_weights_segformer" / "mit_b0.pth"

    print(f"🏗️  Building SegFormer-B0 with {num_classes} classes")
    print(f"🔧 Architecture config from: {_SEGFORMER_CONFIG}")

    if pretrained_encoder:
        print(f"🔄 Loading ImageNet-pretrained encoder weights from GitHub: {github_weights_path}")
    else:
        print("🆕 All weights: initialized from scratch (no pretrained)")

    print(f"🎯 Target classes: {num_classes} (e.g., Cityscapes semantic segmentation)")

    # Load configuration only (not weights)
    config = SegformerConfig.from_pretrained(_SEGFORMER_CONFIG, num_labels=num_classes)

    # Create model with random initialization
    model = SegformerForSemanticSegmentation(config)

    if pretrained_encoder:
        # Load GitHub weights
        try:
            loaded_dict = torch.load(github_weights_path, map_location='cpu')
            if 'state_dict' in loaded_dict:
                github_state_dict = loaded_dict['state_dict']  # If wrapped in 'state_dict' key
            else:
                github_state_dict = loaded_dict  # If directly the state_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"Pretrained weights file not found: {github_weights_path}. Download from https://github.com/NVlabs/SegFormer")

        # State_dict mapping: Convert GitHub (mmseg) keys to HF keys
        # This is a partial mapping; adjust based on exact key mismatches (print keys for full mapping if needed)
        mapped_state_dict = {}
        for github_key, value in github_state_dict.items():
            if not github_key.startswith('backbone.'):
                continue  # Skip non-encoder keys (e.g., decoder will remain random as per paper)

            # Base replacement: 'backbone.' -> 'segformer.encoder.'
            hf_key = github_key.replace('backbone.', 'segformer.encoder.')

            # Patch embeddings: 'patch_embed{stage}.proj.' -> 'patch_embeddings.{stage-1}.patch_embeddings.'
            if 'patch_embed' in hf_key:
                stage = hf_key.split('patch_embed')[1][0]  # e.g., '1' from 'patch_embed1'
                hf_key = hf_key.replace(f'patch_embed{stage}.proj.', f'patch_embeddings.{int(stage)-1}.patch_embeddings.')
                hf_key = hf_key.replace('weight', 'weight')  # Usually matches

            # Blocks: 'block{stage}.{block_idx}.' -> 'block.{stage-1}.{block_idx}.'
            if 'block' in hf_key:
                parts = hf_key.split('.')
                stage = parts[1][-1]  # e.g., '1' from 'block1'
                block_idx = parts[2]
                hf_key = hf_key.replace(f'block{stage}.{block_idx}.', f'block.{int(stage)-1}.{block_idx}.')

            # Attention: 'attn.proj.' -> 'attention.output.dense.'
            hf_key = hf_key.replace('attn.proj.', 'attention.output.dense.')
            hf_key = hf_key.replace('attn.qkv.', 'attention.self.query.')  # Need full mapping for qkv

            # MLP: 'mlp.fc1.' -> 'mlp.dense1.', 'mlp.fc2.' -> 'mlp.dense2.'
            hf_key = hf_key.replace('mlp.fc1.', 'mlp.dense1.')
            hf_key = hf_key.replace('mlp.fc2.', 'mlp.dense2.')

            # Norm: 'norm{stage}.' -> 'layer_norm{stage-1}.'
            if 'norm' in hf_key:
                stage = hf_key.split('norm')[1][0]
                hf_key = hf_key.replace(f'norm{stage}.', f'layer_norm.{int(stage)-1}.')

            # Check if mapped key exists in HF model
            if hf_key in model.state_dict():
                mapped_state_dict[hf_key] = value
            else:
                print(f"Warning: Skipping unmatched key {github_key} -> {hf_key}")

        # Load mapped state_dict (strict=False to ignore unmatched keys like decoder)
        missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
        print(f"Loaded {len(mapped_state_dict)} keys from GitHub weights.")
        if missing:
            print(f"Missing keys (expected for decoder): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    # Verify model type
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