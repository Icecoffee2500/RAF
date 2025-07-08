"""Federated client – trains locally for one epoch then returns state dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset
from easydict import EasyDict as edict

from ..model_builder import build_model
from ...dataset.builder import _build_transform
from ...dataset.cityscapes import CityscapesDataset


class FederatedClient:
    def __init__(self, client_id: int, indices: list[int], root: str | Path, cfg=None, *, resolution: int | None = None, num_classes: int = 19, batch_size: int = 4):
        self.id = client_id
        self.root = Path(root)
        self.batch_size = batch_size
        device_id_param = None
        if cfg is not None:
            device_id_param = cfg.get("device_id", None)
        self.model, self.device = build_model(num_classes, device_id_param)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=6e-5, weight_decay=0.01)

        # Determine resize / crop resolution for this client
        if resolution is not None:
            # Explicit per-client resolution provided (single int, keep Cityscapes 2:1 ratio)
            resize_resolution = int(resolution)
            crop_h = crop_w = resize_resolution
        elif cfg is not None:
            training_cfg = cfg.get("training", {})
            crop_size_cfg = training_cfg.get("crop_size", [512, 512])
            # Convert ListConfig to list if needed
            if hasattr(crop_size_cfg, '__iter__') and len(crop_size_cfg) >= 2:
                resize_resolution = min(crop_size_cfg)
                crop_h, crop_w = crop_size_cfg[0], crop_size_cfg[1]
            else:
                resize_resolution = int(crop_size_cfg)
                crop_h = crop_w = resize_resolution
        else:
            # Fallback to default
            resize_resolution = 512
            crop_h = crop_w = 512
        
        transform_config = edict({
            'resolution': resize_resolution,
            'crop_size': min(crop_h, crop_w),  # Use square crop for now
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
        })
        transform = _build_transform(transform_config, is_train=True)
        
        # Show resolution info for this client
        final_crop_size = transform_config["crop_size"]
        
        full_ds = CityscapesDataset(str(self.root), split='train', mode='fine', target_type='semantic', transform=transform)
        subset_ds = Subset(full_ds, indices)
        self.loader = DataLoader(subset_ds, shuffle=True, batch_size=self.batch_size, num_workers=4)

        # Store info for external access
        self.resize_resolution = resize_resolution
        self.num_samples = len(subset_ds)
        
        print(f"🏢 Client {client_id}: {len(subset_ds)} samples (from {len(full_ds)} total)")
        if resolution is not None:
            print(f"   📐 Custom Resolution: {resize_resolution}→{crop_h}×{crop_w} (resize→crop)")
        elif cfg is not None and hasattr(crop_size_cfg, '__iter__') and len(crop_size_cfg) >= 2:
            print(f"   📐 Resolution: {resize_resolution}→{crop_h}×{crop_w} (resize→crop)")
        else:
            print(f"   📐 Resolution: {resize_resolution}→{final_crop_size}×{final_crop_size} (resize→crop)")

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        losses = []
        for imgs, labels in self.loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(pixel_values=imgs, labels=labels)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        return {"client_id": self.id, "loss": float(sum(losses) / len(losses))}

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def load_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state_dict) 