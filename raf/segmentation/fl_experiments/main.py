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

# --- Convenience CLI preprocessing ----------------------------------------------------

def _preprocess_cli_args() -> None:
    """Convert custom flags (e.g., --resolution 512 256 128) into Hydra overrides.

    This allows users to specify per-client image resolutions and number of clients
    without writing Hydra-style dot-notation overrides.
    """
    import sys as _sys

    new_args: list[str] = []
    i = 0
    while i < len(_sys.argv):
        arg = _sys.argv[i]

        # Match resolution flag (accept --resolution or -r)
        if arg in ("--resolution", "--resolutions", "-r"):
            i += 1
            res_values: list[str] = []
            # Collect subsequent numeric values
            while i < len(_sys.argv) and not _sys.argv[i].startswith("--") and "=" not in _sys.argv[i]:
                res_values.append(_sys.argv[i])
                i += 1
            if res_values:
                res_str = ",".join(res_values)
                new_args.append(f"federated.resolutions=[{res_str}]")
            continue  # Skip normal increment – already advanced

        # Match number of clients flag
        if arg in ("--num_clients", "--clients"):
            if i + 1 < len(_sys.argv):
                num_val = _sys.argv[i + 1]
                new_args.append(f"federated.num_clients={num_val}")
                i += 2
                continue

        # Match data_root flag
        if arg in ("--data_root", "--data", "-d"):
            if i + 1 < len(_sys.argv):
                root_val = _sys.argv[i + 1]
                # Quote path if it contains '=' to avoid confusion for Hydra
                if "=" in root_val:
                    root_val = f'"{root_val}"'
                new_args.append(f"data_root={root_val}")
                i += 2
                continue

        # All other args pass through unchanged
        new_args.append(arg)
        i += 1

    # Replace argv in-place so Hydra sees converted overrides
    _sys.argv[:] = new_args


# --------------------------------------------------------------------------------------


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

        elif arg.startswith("data_root="):
            data_root_val = arg.split("=", 1)[1]
            # Strip possible quotes
            data_root_val = data_root_val.strip("\"'")
            overrides["data_root"] = data_root_val
    
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
        # --- Dynamically adapt dataset split constants ---------------------------------
        from . import dataset_split as _ds

        _ds.NUM_CLIENTS = cfg.federated.num_clients
        _ds.SAMPLES_PER_CLIENT = cfg.federated.get("samples_per_client", 700)
        _ds.NUM_TOTAL_SAMPLES = _ds.NUM_CLIENTS * _ds.SAMPLES_PER_CLIENT

        # -----------------------------------------------------------------------------

        splits = split_cityscapes(cfg.data_root)

        # Prepare per-client resolutions
        res_list = list(cfg.federated.resolutions)
        if len(res_list) < cfg.federated.num_clients:
            res_list.extend([res_list[-1]] * (cfg.federated.num_clients - len(res_list)))
        elif len(res_list) > cfg.federated.num_clients:
            res_list = res_list[: cfg.federated.num_clients]

        training_cfg = cfg.get("training", {})

        clients = [
            FederatedClient(
                cid,
                idxs,
                cfg.data_root,
                cfg=cfg,
                batch_size=training_cfg.get("batch_size", 4),
                resolution=res_list[cid],
            )
            for cid, idxs in splits.items()
        ]

        server = FederatedServer(clients, cfg.data_root, cfg)
        epochs = training_cfg.get("epochs", 50)
        server.train(epochs)
    else:
        raise ValueError(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    # Convert convenience CLI flags to Hydra-compatible overrides before Hydra parses args
    _preprocess_cli_args()
    _main() 