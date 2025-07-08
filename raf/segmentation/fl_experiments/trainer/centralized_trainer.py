"""Centralized training script (single GPU/CPU).
Simplified, FB-style using existing DatasetBuilder.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import torch
from easydict import EasyDict as edict
from omegaconf import DictConfig, ListConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' for logging.")

from ..model_builder import build_model
from ...dataset.builder import DatasetBuilder
from ..metrics import compute_miou
from ..utils.seed import set_seed
from ..dataset_split import get_centralized_indices
from torch.utils.data import Subset

# --------------------------------------------------------------------------------------


class CentralizedTrainer:
    def __init__(self, cfg: Dict[str, Any] | DictConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg["data_root"])
        
        # Extract training config
        training_cfg = cfg.get("training", {})
        self.batch_size = training_cfg.get("batch_size", 4)
        self.num_classes = cfg.get("num_classes", 19)
        
        # Evaluation schedule
        evaluation_cfg = cfg.get("evaluation", {})
        self.eval_every = evaluation_cfg.get("eval_every", 10)
        
        # Build model with optional device specification
        device_id = cfg.get("device_id", None)  # Can be int, string, or None
        self.model, self.device = build_model(self.num_classes, device_id)
        
        # Setup optimizer with config values
        optimizer_cfg = training_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=optimizer_cfg.get("lr", 6e-5), 
            weight_decay=optimizer_cfg.get("weight_decay", 0.01),
            betas=optimizer_cfg.get("betas", [0.9, 0.999])
        )
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._make_dataloaders()
        
        # Initialize wandb if available and configured
        self._init_wandb()
        
        # Logging statistics
        self.total_samples_seen = 0
        
        # Checkpoint tracking
        self.best_miou = 0.0
        self.checkpoint_dir = Path("checkpoints") / cfg.get("exp_name", "experiment")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        seed = cfg.get("seed", 42)
        set_seed(seed)

    # ------------------------------------------------------------------
    def _init_wandb(self) -> None:
        """Initialize wandb logging if available and configured."""
        if not WANDB_AVAILABLE:
            return
            
        logging_cfg = self.cfg.get("logging", {})
        wandb_cfg = logging_cfg.get("wandb", {})
        
        if wandb_cfg and wandb_cfg.get("project"):
            # Convert Hydra config to dict for wandb compatibility
            try:
                from omegaconf import OmegaConf
                config_dict = OmegaConf.to_container(self.cfg, resolve=True)
                # Ensure it's a dict
                if not isinstance(config_dict, dict):
                    config_dict = {
                        "num_classes": self.num_classes,
                        "batch_size": self.batch_size,
                        "data_root": str(self.root)
                    }
            except ImportError:
                # Fallback: create a simplified config dict
                config_dict = {
                    "num_classes": self.num_classes,
                    "batch_size": self.batch_size,
                    "data_root": str(self.root)
                }
            except Exception as e:
                print(f"⚠️  Warning: Could not convert config for wandb: {e}")
                config_dict = {
                    "num_classes": self.num_classes,
                    "batch_size": self.batch_size,
                    "data_root": str(self.root)
                }
            
            try:
                # Get experiment name from config and add timestamp
                base_exp_name = self.cfg.get("exp_name", "experiment")
                timestamp = time.strftime("%y%m%d%H%M", time.localtime())
                exp_name = f"{base_exp_name}_{timestamp}"
                
                wandb.init(
                    project=wandb_cfg.get("project", "segformer_experiments"),
                    entity=wandb_cfg.get("entity", None),
                    config=config_dict,
                    name=exp_name,
                    tags=["centralized", "segformer"]
                )
                
                # Define x-axis for important metrics
                wandb.define_metric("epoch")
                wandb.define_metric("train_loss", step_metric="epoch")
                wandb.define_metric("val_loss", step_metric="epoch")
                wandb.define_metric("val_miou", step_metric="epoch")
                
                print(f"✅ Wandb initialized: {wandb_cfg.get('project')}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize wandb: {e}")
                print("ℹ️  Continuing without wandb logging...")
        else:
            print("ℹ️  Wandb not configured (no project specified)")

    # ------------------------------------------------------------------
    def _make_dataloaders(self):
        training_cfg = self.cfg.get("training", {})
        crop_size = training_cfg.get("crop_size", [512, 512])
        
        # Convert ListConfig to list if needed
        if isinstance(crop_size, ListConfig):
            crop_size = list(crop_size)
        
        # Use the first dimension for square crop, or handle rectangular crops
        if isinstance(crop_size, list) and len(crop_size) >= 2:
            resolution = min(crop_size)  # Use smaller dimension for resolution
            crop_h, crop_w = crop_size[0], crop_size[1] 
        else:
            resolution = crop_size
            crop_h = crop_w = crop_size
            
        dataset_config = edict({
            'dataset_root': str(self.root),
            'train_split': 'train',
            'valid_split': 'val',
            'mode': 'gtFine',
            'target_type': 'semantic',
            'train_batch_size': self.batch_size,
            'valid_batch_size': self.batch_size,
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
        
        # Get limited dataset indices (2100 samples total)
        print("🎯 Using limited dataset for centralized training (2100 samples)")
        limited_indices = get_centralized_indices(self.root)
        
        # Create subset of training dataset
        train_dataset = builder.get_train_dataset()
        limited_train_dataset = Subset(train_dataset, limited_indices)
        
        # Create limited training dataloader
        from torch.utils.data import DataLoader
        limited_train_loader = DataLoader(
            limited_train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True
        )
        
        print(f"📊 Training dataset: {len(limited_train_dataset)} samples (original: {len(train_dataset)})")
        
        # Keep validation dataset as is
        return limited_train_loader, builder.get_valid_dataloader()

    # ------------------------------------------------------------------
    def _step(self, batch: Any) -> Dict[str, float]:
        self.model.train()
        imgs, labels = (b.to(self.device) for b in batch)
        outputs = self.model(pixel_values=imgs, labels=labels)
        loss: torch.Tensor = outputs.loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
        # Update sample count
        self.total_samples_seen += imgs.size(0)
        
        return {"train_loss": float(loss.item())}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        losses = []
        ious = []
        
        eval_start_time = time.time()
        
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outputs = self.model(pixel_values=imgs, labels=labels)
            losses.append(outputs.loss.item())
            
            # Compute mIoU with proper num_classes
            miou = compute_miou(outputs.logits, labels, num_classes=self.num_classes)
            ious.append(miou)
        
        eval_time = time.time() - eval_start_time
        
        return {
            "val_loss": float(sum(losses) / len(losses)),
            "val_miou": float(sum(ious) / len(ious)),
            "val_eval_time": eval_time
        }

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
            print(f"💾 New best model saved! mIoU: {metrics['val_miou']:.3f} -> {best_path}")
        
    # ------------------------------------------------------------------
    def _check_and_save_best(self, epoch: int, val_metrics: Dict[str, float]) -> bool:
        """Check if current model is best and save checkpoint."""
        current_miou = val_metrics['val_miou']
        is_best = current_miou > self.best_miou
        
        if is_best:
            self.best_miou = current_miou
            
        # Save checkpoint
        self._save_checkpoint(epoch + 1, val_metrics, is_best)
        
        return is_best

    # ------------------------------------------------------------------
    def fit(self, epochs: int | None = None) -> None:
        if epochs is None:
            epochs = self.cfg.get("training", {}).get("epochs", 10)
        
        assert epochs is not None, "epochs cannot be None"
        
        print(f"🚀 Starting training for {epochs} epochs")
        print(f"📊 Validation every {self.eval_every} epochs")
        print(f"🎯 Dataset: {self.root}")
        print(f"🔢 Batch size: {self.batch_size}, Classes: {self.num_classes}")
        
        # Show actual image resolution being used
        training_cfg = self.cfg.get("training", {})
        crop_size = training_cfg.get("crop_size", [512, 512])
        if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            print(f"🖼️  Image resolution: {crop_size[0]}×{crop_size[1]} (H×W)")
        else:
            print(f"🖼️  Image resolution: {crop_size}×{crop_size}")
        
        print("-" * 60)
        
        # Initial validation before training starts
        print("🔍 Running initial validation...")
        initial_val_metrics = self._evaluate()
        print(f"📊 Initial | Val Loss: {initial_val_metrics['val_loss']:.4f} | "
              f"mIoU: {initial_val_metrics['val_miou']:.3f} | "
              f"Time: {initial_val_metrics['val_eval_time']:.1f}s")
        
        # Log initial validation to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                initial_wandb_metrics = {
                    "val_loss": initial_val_metrics["val_loss"],
                    "val_miou": initial_val_metrics["val_miou"],
                    "epoch": 0  # Use epoch 0 for initial validation
                }
                wandb.log(initial_wandb_metrics)
            except Exception as e:
                print(f"⚠️  Warning: Failed to log initial validation to wandb: {e}")
        
        print("-" * 60)
        training_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_losses = []
            epoch_samples = 0
            
            for batch in self.train_loader:
                metrics = self._step(batch)
                train_losses.append(metrics["train_loss"])
                epoch_samples += len(batch[0])  # batch size for this batch
            
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = float(sum(train_losses) / len(train_losses))
            
            # Basic epoch logging
            epoch_metrics = {
                "train_loss": avg_train_loss,
                "epoch_time": epoch_time,
                "samples_per_sec": epoch_samples / epoch_time if epoch_time > 0 else 0
            }
            
            # Validation phase (only every eval_every epochs)
            should_evaluate = (epoch + 1) % self.eval_every == 0 or epoch == epochs - 1
            
            if should_evaluate:
                val_metrics = self._evaluate()
                
                # Check and save best model
                is_best = self._check_and_save_best(epoch, val_metrics)
                
                # Add only val_loss and val_miou to wandb logging
                epoch_metrics.update({
                    "val_loss": val_metrics["val_loss"],
                    "val_miou": val_metrics["val_miou"]
                })
                
                # Detailed logging for validation epochs
                best_indicator = " 🏆" if is_best else ""
                print(f"📈 Epoch {epoch + 1:3d}/{epochs} | "
                      f"Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_metrics['val_loss']:.4f} | "
                      f"mIoU: {val_metrics['val_miou']:.3f}{best_indicator} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Samples: {epoch_samples}")
            else:
                # Simple logging for training-only epochs
                print(f"🏃 Epoch {epoch + 1:3d}/{epochs} | "
                      f"Loss: {avg_train_loss:.4f} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Samples: {epoch_samples} | "
                      f"Speed: {epoch_metrics['samples_per_sec']:.1f} samples/s")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                try:
                    # Add epoch to metrics for x-axis
                    # wandb_metrics = {**epoch_metrics, "epoch": epoch + 1}
                    wandb_metrics = {**epoch_metrics}
                    wandb.log(wandb_metrics)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to log to wandb: {e}")
        
        total_training_time = time.time() - training_start_time
        
        # Final summary
        print("-" * 60)
        print(f"✅ Training completed!")
        print(f"⏱️  Total time: {total_training_time:.1f}s ({total_training_time/60:.1f}m)")
        print(f"📊 Total samples processed: {self.total_samples_seen:,}")
        print(f"🚀 Average speed: {self.total_samples_seen/total_training_time:.1f} samples/s")
        print(f"🏆 Best mIoU: {self.best_miou:.3f}")
        print(f"💾 Checkpoints saved in: {self.checkpoint_dir}")
        print(f"   - Latest: {self.checkpoint_dir}/latest.pth")
        print(f"   - Best: {self.checkpoint_dir}/best.pth")
        
        if WANDB_AVAILABLE and wandb.run is not None:
            try:
                # Log final summary metrics
                wandb.log({
                    "total_training_time": total_training_time,
                    "average_samples_per_sec": self.total_samples_seen/total_training_time
                })
                wandb.finish()
            except Exception as e:
                print(f"⚠️  Warning: Failed to finalize wandb: {e}")


# --------------------------------------------------------------------------------------

def train_centralized(cfg: Dict[str, Any] | DictConfig) -> None:
    trainer = CentralizedTrainer(cfg)
    training_cfg = cfg.get("training", {})
    epochs = training_cfg.get("epochs", 10)
    trainer.fit(epochs) 