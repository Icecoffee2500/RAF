"""Segmentation metrics utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_miou(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> float:
    """Compute mean IoU for semantic segmentation.
    
    Args:
        logits: Model output logits (B, C, H, W)
        targets: Ground truth labels (B, H, W)  
        num_classes: Number of classes
        ignore_index: Class index to ignore
        
    Returns:
        mIoU as float
    """
    # Upsample logits to match target resolution if needed
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(
            logits, 
            size=targets.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
    
    # Get predictions
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    
    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Remove ignore pixels
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]
    
    # Compute IoU for each class
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
    
    return sum(ious) / len(ious) if ious else 0.0


def compute_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> float:
    """Compute pixel accuracy for semantic segmentation.
    
    Args:
        logits: Model output logits (B, C, H, W)
        targets: Ground truth labels (B, H, W)
        ignore_index: Class index to ignore
        
    Returns:
        Pixel accuracy as float
    """
    # Upsample logits to match target resolution if needed
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(
            logits, 
            size=targets.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
    
    # Get predictions
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    
    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Remove ignore pixels
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]
    
    # Compute accuracy
    correct = (preds == targets).sum().float()
    total = targets.numel()
    
    return (correct / total).item() if total > 0 else 0.0


def compute_class_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 19, ignore_index: int = 255) -> dict:
    """Compute per-class accuracy for semantic segmentation.
    
    Args:
        logits: Model output logits (B, C, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes
        ignore_index: Class index to ignore
        
    Returns:
        Dictionary with per-class accuracy
    """
    # Upsample logits to match target resolution if needed
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(
            logits, 
            size=targets.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
    
    # Get predictions
    preds = torch.argmax(logits, dim=1)  # (B, H, W)
    
    # Flatten tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Remove ignore pixels
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]
    
    # Compute per-class accuracy
    class_acc = {}
    for cls in range(num_classes):
        cls_mask = targets == cls
        if cls_mask.sum() > 0:
            cls_correct = (preds[cls_mask] == targets[cls_mask]).sum().float()
            cls_total = cls_mask.sum().float()
            class_acc[f'class_{cls}'] = (cls_correct / cls_total).item()
        else:
            class_acc[f'class_{cls}'] = 0.0
    
    return class_acc 