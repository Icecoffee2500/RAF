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
    def __init__(self, client_id: int, indices: list[int], root: str | Path, num_classes: int = 19, batch_size: int = 4):
        self.id = client_id
        self.root = Path(root)
        self.batch_size = batch_size
        self.model, self.device = build_model(num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=6e-5, weight_decay=0.01)

        # 기존 transform 로직 재사용
        transform_config = edict({
            'resolution': 512,
            'crop_size': 512,
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
        })
        transform = _build_transform(transform_config, is_train=True)
        
        full_ds = CityscapesDataset(str(self.root), split='train', mode='fine', target_type='semantic', transform=transform)
        self.loader = DataLoader(Subset(full_ds, indices), shuffle=True, batch_size=self.batch_size, num_workers=4)

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