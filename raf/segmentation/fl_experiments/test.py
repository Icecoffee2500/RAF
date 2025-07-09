"""Test script for evaluating saved model checkpoints.

Usage:
python -m raf.segmentation.fl_experiments.test --checkpoint checkpoints/my_experiment/best.pth
python -m raf.segmentation.fl_experiments.test --checkpoint checkpoints/my_experiment/latest.pth --device cuda:1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf

from .model_builder import build_model
from ..dataset.builder import DatasetBuilder
from .metrics import compute_miou
from .utils.seed import set_seed


def load_checkpoint(checkpoint_path: str | Path, device_id: int | str | None = None) -> tuple:
    """Load model checkpoint and return model, config, and metrics.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device_id: Device to load model on
        
    Returns:
        (model, device, config, checkpoint_info)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract info
    config = checkpoint['config']
    num_classes = config.get('num_classes', 19)
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Build model
    model, device = build_model(num_classes, device_id)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'best_miou': checkpoint['best_miou'],
        'metrics': checkpoint['metrics']
    }
    
    print(f"✅ Model loaded from epoch {checkpoint_info['epoch']}")
    print(f"🏆 Checkpoint mIoU: {checkpoint_info['metrics']['val_miou']:.3f}")
    
    return model, device, config, checkpoint_info


def create_test_dataloader(config: Dict[str, Any]) -> DataLoader:
    """Create test dataloader from config."""
    training_cfg = config.get("training", {})
    crop_size = training_cfg.get("crop_size", [512, 512])
    
    # Handle ListConfig
    if hasattr(crop_size, '_content'):
        crop_size = list(crop_size)
    
    if isinstance(crop_size, list) and len(crop_size) >= 2:
        resolution = min(crop_size)
        crop_h, crop_w = crop_size[0], crop_size[1] 
    else:
        resolution = crop_size
        crop_h = crop_w = crop_size
        
    dataset_config = edict({
        'dataset_root': config['data_root'],
        'train_split': 'train',
        'valid_split': 'val',
        'mode': 'gtFine',
        'target_type': 'semantic',
        'train_batch_size': 1,  # Use batch size 1 for testing
        'valid_batch_size': 1,
        'num_workers': 4,
        'pin_memory': True,
        'transform_config': edict({
            'resolution': resolution,
            'crop_size': min(crop_h, crop_w),
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
        })
    })
    
    builder = DatasetBuilder(dataset_config)
    return builder.get_valid_dataloader()


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, 
                  device: torch.device, num_classes: int = 19) -> Dict[str, float]:
    """Evaluate model on test data."""
    model.eval()
    
    losses = []
    ious = []
    
    print("🔍 Running evaluation...")
    
    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(pixel_values=imgs, labels=labels)
        losses.append(outputs.loss.item())
        
        # Compute mIoU
        miou = compute_miou(outputs.logits, labels, num_classes=num_classes)
        ious.append(miou)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataloader)} batches...")
    
    results = {
        "test_loss": float(sum(losses) / len(losses)),
        "test_miou": float(sum(ious) / len(ious)),
        "num_samples": len(dataloader)
    }
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Test trained SegFormer model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to checkpoint file")
    parser.add_argument("--device", "-d", type=str, default=None,
                       help="Device to run on (e.g., cuda:0, cpu)")
    parser.add_argument("--data_root", type=str, default=None,
                       help="Override data root path")
    parser.add_argument("--resolution", type=int, nargs=2, default=None,
                       help="Override test resolution [height, width] (e.g., --resolution 512 1024)")
    
    args = parser.parse_args()
    
    try:
        # Load checkpoint
        model, device, config, checkpoint_info = load_checkpoint(args.checkpoint, args.device)
        
        # Override data root if provided
        if args.data_root:
            config['data_root'] = args.data_root
            print(f"📁 Using data root: {args.data_root}")
        
        # Override resolution if provided
        if args.resolution:
            config['training']['crop_size'] = args.resolution
            print(f"🖼️  Using test resolution: {args.resolution[0]}×{args.resolution[1]}")
        
        # Create test dataloader
        test_loader = create_test_dataloader(config)
        
        # Run evaluation
        print("-" * 60)
        results = evaluate_model(model, test_loader, device, config.get('num_classes', 19))
        
        # Print results
        print("-" * 60)
        print("📊 Test Results:")
        print(f"   Test Loss: {results['test_loss']:.4f}")
        print(f"   Test mIoU: {results['test_miou']:.3f}")
        print(f"   Samples: {results['num_samples']}")
        print("-" * 60)
        
        # Compare with checkpoint metrics
        if 'val_miou' in checkpoint_info['metrics']:
            val_miou = checkpoint_info['metrics']['val_miou']
            print(f"📈 Comparison:")
            print(f"   Validation mIoU: {val_miou:.3f}")
            print(f"   Test mIoU: {results['test_miou']:.3f}")
            print(f"   Difference: {results['test_miou'] - val_miou:+.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 