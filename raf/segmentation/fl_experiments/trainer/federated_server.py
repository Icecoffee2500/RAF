"""Federated server coordinating FedAvg."""

from __future__ import annotations

import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for logging.")

from ..model_builder import build_model
from .federated_client import FederatedClient
from ...dataset.builder import DatasetBuilder
from ..metrics import compute_miou


class FederatedServer:
    def __init__(self, clients: List[FederatedClient], data_root: str, cfg=None):
        self.clients = clients
        self.cfg = cfg
        # initialize global model from first client
        self.global_state = self.clients[0].get_state()
        
        # Checkpoint tracking (same as centralized)
        self.best_miou = 0.0
        exp_name = cfg.get("exp_name", "experiment") if cfg else "experiment"
        self.checkpoint_dir = Path("checkpoints") / f"{exp_name}_federated"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # create validation set for evaluation (use same config as centralized)
        if cfg is not None:
            training_cfg = cfg.get("training", {})
            crop_size = training_cfg.get("crop_size", [512, 512])
            # Convert ListConfig to list if needed
            if hasattr(crop_size, '__iter__') and len(crop_size) >= 2:
                resolution = min(crop_size)
                crop_h, crop_w = crop_size[0], crop_size[1]
                self.resolution_info = f"{resolution}→{crop_h}×{crop_w}"
            else:
                resolution = crop_size
                crop_h = crop_w = crop_size
                self.resolution_info = f"{resolution}→{crop_size}×{crop_size}"
        else:
            # Fallback to default
            resolution = 512
            crop_h = crop_w = 512
            self.resolution_info = f"512→512×512"
        
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
                'resolution': resolution,
                'crop_size': min(crop_h, crop_w),  # Use square crop for now
                'brightness': 0.5,
                'contrast': 0.5,
                'saturation': 0.5,
            })
        })
        builder = DatasetBuilder(dataset_config)
        self.val_loader = builder.get_valid_dataloader()
        
        # Initialize wandb if available and configured
        self._init_wandb()

    # ------------------------------------------------------------------
    def _init_wandb(self) -> None:
        """Initialize wandb logging if available and configured."""
        if not WANDB_AVAILABLE:
            return
            
        if not self.cfg:
            print("ℹ️  No config available for wandb initialization")
            return
            
        logging_cfg = self.cfg.get("logging", {})
        wandb_cfg = logging_cfg.get("wandb", {})
        
        if wandb_cfg and wandb_cfg.get("project"):
            # Convert config to dict for wandb compatibility
            try:
                from omegaconf import OmegaConf
                config_dict = OmegaConf.to_container(self.cfg, resolve=True)
                if not isinstance(config_dict, dict):
                    config_dict = {"mode": "federated", "clients": len(self.clients)}
            except ImportError:
                config_dict = {"mode": "federated", "clients": len(self.clients)}
            except Exception as e:
                print(f"⚠️  Warning: Could not convert config for wandb: {e}")
                config_dict = {"mode": "federated", "clients": len(self.clients)}
            
            try:
                # Get experiment name from config and add timestamp
                base_exp_name = self.cfg.get("exp_name", "experiment")
                timestamp = time.strftime("%y%m%d%H%M", time.localtime())
                exp_name = f"{base_exp_name}_federated_{timestamp}"
                
                wandb.init(
                    project=wandb_cfg.get("project", "segformer_experiments"),
                    entity=wandb_cfg.get("entity", None),
                    config=config_dict,
                    name=exp_name,
                    tags=["federated", "segformer", f"{len(self.clients)}clients"]
                )
                
                # Define x-axis for important metrics
                wandb.define_metric("round")
                wandb.define_metric("avg_client_loss", step_metric="round")
                wandb.define_metric("global_val_loss", step_metric="round")
                wandb.define_metric("global_val_miou", step_metric="round")
                
                print(f"✅ Wandb initialized for federated: {wandb_cfg.get('project')}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize wandb: {e}")
                print("ℹ️  Continuing without wandb logging...")
        else:
            print("ℹ️  Wandb not configured (no project specified)")

    # ------------------------------------------------------------------
    def aggregate(self) -> None:
        """FedAvg across client weights."""
        avg_state: Dict[str, torch.Tensor] = {}
        for k in self.global_state.keys():
            # Get all client tensors for this parameter
            client_tensors = [c.get_state()[k] for c in self.clients]
            stacked = torch.stack(client_tensors)
            
            # Only average floating point tensors (weights, biases)
            if stacked.dtype.is_floating_point:
                avg_state[k] = torch.mean(stacked, dim=0)
            else:
                # For non-floating point tensors (buffers, indices), use first client's value
                avg_state[k] = client_tensors[0].clone()
                
        self.global_state = avg_state
        for c in self.clients:
            c.load_state(self.global_state)

    # ------------------------------------------------------------------
    def _save_checkpoint(self, round_num: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save federated model checkpoint."""
        # Use the first client's model to get the global model state
        # (all clients have the same weights after aggregation)
        global_model = self.clients[0].model
        
        checkpoint = {
            'round': round_num,
            'model_state_dict': global_model.state_dict(),  # Global model weights
            'global_state': self.global_state,              # Raw state dict for resuming
            'metrics': metrics,
            'config': self.cfg,
            'best_miou': self.best_miou
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"💾 New best federated model saved! mIoU: {metrics['global_val_miou']:.3f} -> {best_path}")

    # ------------------------------------------------------------------
    def _check_and_save_best(self, round_num: int, val_metrics: Dict[str, float]) -> bool:
        """Check if current global model is best and save checkpoint."""
        current_miou = val_metrics['global_val_miou']
        is_best = current_miou > self.best_miou
        
        if is_best:
            self.best_miou = current_miou
            
        # Save checkpoint
        self._save_checkpoint(round_num, val_metrics, is_best)
        
        return is_best

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
        eval_every = 10  # Evaluate every 10 rounds (same as centralized)
        
        print(f"🚀 Starting federated training for {epochs} rounds")
        print(f"🏢 Clients: {len(self.clients)}")
        print("🖼️  Client resolutions & dataset sizes:")
        for c in self.clients:
            res = getattr(c, "resize_resolution", "?")
            print(f"   • Client {c.id}: {c.num_samples} samples | resize→crop: {res}")
        print(f"📊 Global validation every {eval_every} rounds")
        print("-" * 60)
        
        # Initial validation before training starts
        print("🔍 Running initial validation...")
        initial_metrics = self._evaluate_global()
        print(f"📊 Initial | Val Loss: {initial_metrics['global_val_loss']:.4f} | "
              f"mIoU: {initial_metrics['global_val_miou']:.3f}")
        
        # Log initial validation to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                initial_wandb_metrics = {
                    "global_val_loss": initial_metrics["global_val_loss"],
                    "global_val_miou": initial_metrics["global_val_miou"],
                    "round": 0  # Use round 0 for initial validation
                }
                wandb.log(initial_wandb_metrics)
            except Exception as e:
                print(f"⚠️  Warning: Failed to log initial validation to wandb: {e}")
        
        print("-" * 60)
        
        training_start_time = time.time()
        
        for epoch in range(epochs):
            round_start_time = time.time()
            
            # Local training on all clients
            losses = []
            print(f"🏃 Round {epoch + 1}/{epochs} - Local training...")
            for c in self.clients:
                metrics = c.train_one_epoch()
                losses.append(metrics["loss"])
                print(f"   Client {c.id}: Loss {metrics['loss']:.4f}")
            
            # FedAvg aggregation
            print("🔄 Aggregating client models (FedAvg)...")
            self.aggregate()
            
            avg_client_loss = mean(losses)
            round_time = time.time() - round_start_time
            
            # Validation phase (only every eval_every rounds or last round)
            should_evaluate = (epoch + 1) % eval_every == 0 or epoch == epochs - 1
            
            # Prepare wandb metrics
            round_metrics = {
                "avg_client_loss": avg_client_loss,
                "round_time": round_time,
                "round": epoch + 1
            }
            
            if should_evaluate:
                # Global evaluation
                global_metrics = self._evaluate_global()
                
                # Check and save best model
                is_best = self._check_and_save_best(epoch + 1, global_metrics)
                
                # Add validation metrics to wandb
                round_metrics.update({
                    "global_val_loss": global_metrics["global_val_loss"],
                    "global_val_miou": global_metrics["global_val_miou"]
                })
                
                # Detailed logging for validation rounds
                current_miou = global_metrics['global_val_miou']
                best_indicator = " 🏆" if is_best else ""
                print(f"📈 Round {epoch + 1:3d}/{epochs} | "
                      f"Avg Client Loss: {avg_client_loss:.4f} | "
                      f"Global Val Loss: {global_metrics['global_val_loss']:.4f} | "
                      f"Global mIoU: {current_miou:.3f}{best_indicator} | "
                      f"Time: {round_time:.1f}s")
            else:
                # Simple logging for training-only rounds
                print(f"🏃 Round {epoch + 1:3d}/{epochs} | "
                      f"Avg Client Loss: {avg_client_loss:.4f} | "
                      f"Time: {round_time:.1f}s")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                try:
                    wandb.log(round_metrics)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to log to wandb: {e}")
            
            print("-" * 40)
        
        total_training_time = time.time() - training_start_time
        
        # Final summary
        print("✅ Federated training completed!")
        print(f"⏱️  Total time: {total_training_time:.1f}s ({total_training_time/60:.1f}m)")
        print(f"🚀 Average time per round: {total_training_time/epochs:.1f}s")
        print(f"🏆 Best Global mIoU: {self.best_miou:.3f}")
        print(f"💾 Checkpoints saved in: {self.checkpoint_dir}")
        print(f"   - Latest: {self.checkpoint_dir}/latest.pth")
        print(f"   - Best: {self.checkpoint_dir}/best.pth")
        print(f"🏢 Total rounds: {epochs}")
        print(f"👥 Clients participated: {len(self.clients)}")
        
        # Finalize wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                # Log final summary metrics
                wandb.log({
                    "total_training_time": total_training_time,
                    "average_time_per_round": total_training_time/epochs,
                    "best_global_miou": self.best_miou,
                    "total_rounds": epochs,
                    "num_clients": len(self.clients)
                })
                wandb.finish()
                print("📊 Wandb logging completed and closed")
            except Exception as e:
                print(f"⚠️  Warning: Failed to finalize wandb: {e}") 