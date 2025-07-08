"""Federated server coordinating FedAvg."""

from __future__ import annotations

from statistics import mean
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..model_builder import build_model
from .federated_client import FederatedClient
from ...dataset.builder import DatasetBuilder
from ..metrics import compute_miou


class FederatedServer:
    def __init__(self, clients: List[FederatedClient], data_root: str):
        self.clients = clients
        # initialize global model from first client
        self.global_state = self.clients[0].get_state()
        
        # create validation set for evaluation
        dataset_config = edict({
            'dataset_root': data_root,
            'train_split': 'train',
            'valid_split': 'val',
            'mode': 'gtFine',
            'target_type': 'semantic',
            'train_batch_size': 4,
            'valid_batch_size': 4,
            'num_workers': 4,
            'pin_memory': True,
            'transform_config': edict({
                'resolution': 512,
                'crop_size': 512,
                'brightness': 0.5,
                'contrast': 0.5,
                'saturation': 0.5,
            })
        })
        builder = DatasetBuilder(dataset_config)
        self.val_loader = builder.get_valid_dataloader()

    # ------------------------------------------------------------------
    def aggregate(self) -> None:
        """FedAvg across client weights."""
        avg_state: Dict[str, torch.Tensor] = {}
        for k in self.global_state.keys():
            avg_state[k] = torch.mean(torch.stack([c.get_state()[k] for c in self.clients]), dim=0)
        self.global_state = avg_state
        for c in self.clients:
            c.load_state(self.global_state)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate_global(self) -> Dict[str, float]:
        """Evaluate global model on validation set."""
        # Use first client's model (they all have same weights after aggregation)
        model = self.clients[0].model
        device = self.clients[0].device
        model.eval()
        
        losses = []
        ious = []
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(pixel_values=imgs, labels=labels)
            losses.append(outputs.loss.item())
            
            miou = compute_miou(outputs.logits, labels, num_classes=19)
            ious.append(miou)
            
        return {
            "global_val_loss": float(sum(losses) / len(losses)),
            "global_val_miou": float(sum(ious) / len(ious))
        }

    # ------------------------------------------------------------------
    def train(self, epochs: int = 5) -> None:
        for epoch in range(epochs):
            losses = []
            for c in self.clients:
                metrics = c.train_one_epoch()
                losses.append(metrics["loss"])
            self.aggregate()
            
            # Evaluate global model
            global_metrics = self._evaluate_global()
            print({
                "epoch": epoch, 
                "avg_client_loss": mean(losses),
                **global_metrics
            }) 