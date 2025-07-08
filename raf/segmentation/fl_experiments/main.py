"""Entrypoint for centralized or federated experiments.

Usage examples
--------------
# Basic usage (uses default experiment name)
python -m raf.segmentation.fl_experiments.main

# Custom experiment name
python -m raf.segmentation.fl_experiments.main exp_name=segformer_baseline

# Full example with all options
python -m raf.segmentation.fl_experiments.main exp_name=segformer_gpu1 device_id=1 mode=central

# GPU selection with experiment name
python -m raf.segmentation.fl_experiments.main exp_name=test_cpu device_id=cpu
python -m raf.segmentation.fl_experiments.main exp_name=test_gpu2 device_id=cuda:2

# Federated experiment
python -m raf.segmentation.fl_experiments.main exp_name=federated_test mode=federated

# Environment variable for GPU + experiment name
export CUDA_VISIBLE_DEVICES=1
python -m raf.segmentation.fl_experiments.main exp_name=my_experiment
"""

from __future__ import annotations

import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from .trainer.centralized_trainer import train_centralized
from .trainer.federated_client import FederatedClient
from .trainer.federated_server import FederatedServer
from .dataset_split import split_cityscapes
from .utils.seed import set_seed


def _parse_command_line_overrides() -> dict:
    """Parse command line arguments for quick overrides."""
    overrides = {}
    
    for arg in sys.argv:
        if arg.startswith("device_id="):
            device_str = arg.split("=", 1)[1]
            # Handle numeric values
            if device_str.isdigit():
                overrides["device_id"] = int(device_str)
            else:
                # Handle string values (remove quotes if present)
                device_str = device_str.strip("\"'")
                if device_str.lower() == "null" or device_str.lower() == "none":
                    overrides["device_id"] = None
                else:
                    overrides["device_id"] = device_str
        
        elif arg.startswith("exp_name="):
            exp_name = arg.split("=", 1)[1].strip("\"'")
            overrides["exp_name"] = exp_name
    
    return overrides


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    # Parse command line overrides
    overrides = _parse_command_line_overrides()
    
    # Apply device_id override if provided
    if "device_id" in overrides:
        cfg.device_id = overrides["device_id"]
        print(f"🎯 Device override from command line: {overrides['device_id']}")
    
    # Set experiment name (use override or config default)
    if "exp_name" in overrides:
        cfg.exp_name = overrides["exp_name"]
    
    exp_name = cfg.get("exp_name", "experiment")
    print(f"🏷️  Experiment name: {exp_name}")
    
    if cfg.mode == "central":
        train_centralized(cfg)
    elif cfg.mode == "federated":
        splits = split_cityscapes(cfg.data_root)
        training_cfg = cfg.get("training", {})
        clients = [
            FederatedClient(cid, idxs, cfg.data_root, cfg=cfg, batch_size=training_cfg.get("batch_size", 4)) 
            for cid, idxs in splits.items()
        ]
        server = FederatedServer(clients, cfg.data_root, cfg)
        training_cfg = cfg.get("training", {})
        epochs = training_cfg.get("epochs", 50)
        server.train(epochs)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    _main() 