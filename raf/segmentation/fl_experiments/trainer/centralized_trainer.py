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
    from wandb.sdk.wandb_run import Run as WandbRun
except ImportError:
    WandbRun = None

from ..model_builder import build_model
from ...dataset.builder import DatasetBuilder
from ..metrics import compute_miou
from ..utils.scheduler import PolyLR
from ..dataset_split import get_centralized_indices
from torch.utils.data import Subset

# --------------------------------------------------------------------------------------


class CentralizedTrainer:
    def __init__(self, cfg: Dict[str, Any] | DictConfig, wandb_run: WandbRun | None = None, checkpoint_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.wandb_run = wandb_run
        self.root = Path(cfg["data"]["data_root"])
        
        # Extract training config
        training_cfg = cfg.get("training", {})
        self.batch_size = training_cfg.get("batch_size", 4)
        self.num_classes = cfg.data.get("num_classes", 19)
        
        # Evaluation schedule
        evaluation_cfg = cfg.get("evaluation", {})
        self.eval_every = evaluation_cfg.get("eval_every", 10)
        
        # Build model with optional device specification
        device_id = cfg.get("device_id", None)  # Can be int, string, or None
        model_cfg = cfg.get("model", {})
        pretrained_flag = model_cfg.get("pretrained", False)
        self.model, self.device = build_model(self.num_classes, device_id, pretrained_encoder=pretrained_flag)
        
        # Setup optimizer with config values
        optimizer_cfg = cfg.get("optimizer", {})
        # 여기는 왜 optimizer_cfg.get("name", "AdamW") 로 안할까?
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=optimizer_cfg.get("lr", 6e-5), 
            weight_decay=optimizer_cfg.get("weight_decay", 0.01),
            betas=optimizer_cfg.get("betas", [0.9, 0.999])
        )
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._make_dataloaders()

        # Setup scheduler
        scheduler_cfg = cfg.get("scheduler", {})
        if scheduler_cfg and scheduler_cfg.get("name") == "poly":
            self.scheduler = PolyLR(
                optimizer=self.optimizer,
                max_iter=scheduler_cfg.get("max_iter", 160000),
                power=scheduler_cfg.get("power", 1.0),
                min_lr=scheduler_cfg.get("min_lr", 0.0),
            )
        else:
            self.scheduler = None
        
        # Logging statistics
        self.total_samples_seen = 0
        
        # Checkpoint tracking
        self.best_miou = 0.0
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        else:
            # Fallback for standalone usage
            self.checkpoint_dir = Path("checkpoints") / cfg.get("exp_name", "experiment")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------------------------
    def _make_dataloaders(self):
        training_cfg = self.cfg.get("training", {})
        crop_size = training_cfg.get("train_crop_size", [512, 512])
        
        # Convert ListConfig to list if needed
        if isinstance(crop_size, ListConfig):
            crop_size = list(crop_size)
        
        # # Use the first dimension for square crop, or handle rectangular crops
        # if isinstance(crop_size, list) and len(crop_size) >= 2:
        #     resolution = min(crop_size)  # Use smaller dimension for resolution
        #     crop_h, crop_w = crop_size[0], crop_size[1] 
        # else:
        #     resolution = crop_size
        #     crop_h = crop_w = crop_size

        dataset_config = edict({
            'dataset_root': str(self.root),
            'train_split': 'train',
            'valid_split': 'val',
            'mode': 'gtFine',
            'target_type': 'semantic',
            'train_batch_size': self.cfg.training.batch_size,
            'valid_batch_size': self.cfg.training.batch_size,
            'num_workers': self.cfg.data.get('num_workers', 4),
            'pin_memory': self.cfg.data.get('pin_memory', True),
            'transform_config': edict({
                'random_scale': self.cfg.data.transforms.random_scale, # Pass augmentation config
                'random_flip': self.cfg.data.transforms.get('random_flip', False),
                'train_crop_size': self.cfg.data.transforms.train_random_crop,
                # 'test_crop_size': self.cfg.data.transforms.eval_random_crop, # Use test crop size for validation resize
                'normalize': self.cfg.data.transforms.normalize,
            })
        })
        
        builder = DatasetBuilder(dataset_config)
        
        return builder.get_train_dataloader(), builder.get_valid_dataloader()

    # ------------------------------------------------------------------
    def _step(self, batch: Any) -> Dict[str, float]:
        self.model.train()
        imgs, labels = (b.to(self.device) for b in batch)
        outputs = self.model(pixel_values=imgs, labels=labels)
        loss: torch.Tensor = outputs.loss
        
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
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
        data_cfg = self.cfg.get("data", {})
        train_crop_size = data_cfg.get("transforms", {}).get("train_random_crop", [1024, 1024])
        if isinstance(train_crop_size, (list, tuple)) and len(train_crop_size) >= 2:
            print(f"🖼️  Train Image resolution: {train_crop_size[0]}×{train_crop_size[1]} (H×W)")
        else:
            print(f"🖼️  Train Image resolution: {train_crop_size}×{train_crop_size}")
        
        print("-" * 60)
        
        # Initial validation before training starts
        print("🔍 Running initial validation...")
        initial_val_metrics = self._evaluate()
        print(f"📊 Initial | Val Loss: {initial_val_metrics['val_loss']:.4f} | "
              f"mIoU: {initial_val_metrics['val_miou']:.3f} | "
              f"Time: {initial_val_metrics['val_eval_time']:.1f}s")
        
        # Log initial validation to wandb
        if self.wandb_run:
            try:
                initial_wandb_metrics = {
                    "val_loss": initial_val_metrics["val_loss"],
                    "val_miou": initial_val_metrics["val_miou"],
                    "epoch": 0  # Use epoch 0 for initial validation
                }
                self.wandb_run.log(initial_wandb_metrics)
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
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_metrics = {
                "lr": current_lr
            }
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
                      f"LR: {current_lr:.2e} | "
                      f"Val Loss: {val_metrics['val_loss']:.4f} | "
                      f"mIoU: {val_metrics['val_miou']:.3f}{best_indicator} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Samples: {epoch_samples}")
            else:
                # Simple logging for training-only epochs
                print(f"🏃 Epoch {epoch + 1:3d}/{epochs} | "
                      f"Loss: {avg_train_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {epoch_time:.1f}s | "
                      f"Samples: {epoch_samples} | "
                      f"Speed: {epoch_metrics['samples_per_sec']:.1f} samples/s")
            
            # Log to wandb if available
            if self.wandb_run:
                try:
                    # Add epoch to metrics for x-axis
                    wandb_metrics = {**epoch_metrics, "epoch": epoch + 1}
                    self.wandb_run.log(wandb_metrics)
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