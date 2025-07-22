"""Entrypoint for centralized or federated experiments.

Usage examples
--------------
# Basic usage (uses default experiment name)
python -m raf.segmentation.fl_experiments.main
# Custom experiment name
python -m raf.segmentation.fl_experiments.main exp_name=segformer_cl

# ETRI experiment
python -m raf.segmentation.fl_experiments.main exp_name=segformer_cl data.data_root=/home/user_cau/taeheon_ws/RAF/data/cityscapes

# Full example with all options
python -m raf.segmentation.fl_experiments.main exp_name=segformer_gpu1 device_id=1 mode=central data_root=/path/to/your/data

# GPU selection with experiment name
python -m raf.segmentation.fl_experiments.main exp_name=test_cpu device_id=cpu
python -m raf.segmentation.fl_experiments.main exp_name=test_gpu2 device_id=cuda:2

# Federated experiment
python -m raf.segmentation.fl_experiments.main exp_name=federated_test mode=federated federated.num_clients=5
# Federated experiment with custom resolutions
python -m raf.segmentation.fl_experiments.main exp_name=federated_res mode=federated federated.resolutions=[256,512]

# Environment variable for GPU + experiment name
export CUDA_VISIBLE_DEVICES=1
python -m raf.segmentation.fl_experiments.main exp_name=my_experiment
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .trainer.federated_client import FederatedClient
from .trainer.federated_server import FederatedServer
from .dataset_split import split_cityscapes
from .utils.seed import set_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_wandb(cfg: DictConfig) -> wandb.sdk.wandb_run.Run | None:
    """Initialize wandb based on the configuration."""
    if not WANDB_AVAILABLE or not cfg.logging.get("wandb"):
        print("ℹ️  Wandb not configured or not installed.")
        return None

    wandb_cfg = cfg.logging.wandb
    if not wandb_cfg.get("project"):
        print("ℹ️  Wandb project name not specified.")
        return None

    # Common settings
    timestamp = time.strftime("%y%m%d%H%M", time.localtime())
    base_exp_name = cfg.get("exp_name", "experiment")
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Mode-specific settings
    if cfg.mode == "central":
        exp_name = f"{base_exp_name}_{timestamp}"
        tags = ["centralized", "segformer"]
        step_metric = "epoch"
    elif cfg.mode == "federated":
        exp_name = f"{base_exp_name}_federated_{timestamp}"
        tags = ["federated", "segformer", f"{cfg.federated.num_clients}clients"]
        step_metric = "epoch"
    else:
        return None # Unknown mode

    try:
        run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.get("entity"),
            config=config_dict,
            name=exp_name,
            tags=tags,
        )
        if run:
            wandb.define_metric(step_metric)
            print(f"✅ Wandb initialized for '{cfg.mode}' mode. Run: {exp_name}")
        return run
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize wandb: {e}")
        return None


@hydra.main(config_path="configs", config_name="base", version_base="1.2")
def _main(cfg: DictConfig) -> None:
    print(f"🎯 Device: {cfg.device_id}")
    exp_name = cfg.get("exp_name", "experiment")
    print(f"🏷️  Experiment name: {exp_name}")

    # Set random seed for reproducibility from the main entrypoint
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Get Hydra's output directory for this run and create a checkpoint subdir
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Checkpoints will be saved to: {checkpoint_dir}")

    # Setup wandb
    wandb_run = setup_wandb(cfg)

    try:
        if cfg.mode == "central":
            from .trainer.centralized_trainer import CentralizedTrainer
            trainer = CentralizedTrainer(cfg, wandb_run=wandb_run, checkpoint_dir=checkpoint_dir)
            training_cfg = cfg.get("training", {})
            epochs = training_cfg.get("epochs", 10)
            trainer.fit(epochs)

        elif cfg.mode == "federated":
            # --- Dynamically adapt dataset split constants ---------------------------------
            from . import dataset_split as _ds

            _ds.NUM_CLIENTS = cfg.federated.num_clients
            _ds.SAMPLES_PER_CLIENT = cfg.federated.get("samples_per_client", 700)
            _ds.NUM_TOTAL_SAMPLES = _ds.NUM_CLIENTS * _ds.SAMPLES_PER_CLIENT
            # -----------------------------------------------------------------------------

            splits = split_cityscapes(cfg.data.data_root, seed=cfg.data.get("split_seed", 0))

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
                    cfg.data.data_root,
                    cfg=cfg,
                    batch_size=training_cfg.get("batch_size", 4),
                    resolution=res_list[cid],
                )
                for cid, idxs in splits.items()
            ]

            server = FederatedServer(clients, cfg.data.data_root, cfg, wandb_run=wandb_run, checkpoint_dir=checkpoint_dir)
            epochs = training_cfg.get("epochs", 50)
            server.train(epochs)
        else:
            raise ValueError(f"Unknown mode {cfg.mode}")

    except Exception as e:
        print(f"❌ An error occurred during training: {e}")
        raise
    finally:
        if wandb_run:
            wandb_run.finish()
            print("📊 Wandb run finished.")


if __name__ == "__main__":
    _main() 