from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from hydra.core.config_store import ConfigStore

@dataclass
class DatasetConfig:
    name: str = "coco"
    root: str = ""
    train_set: str = "train2017"
    test_set: str = "val2017"
    num_joints: int = 17
    flip: bool = True
    rot_factor: float = 40.0
    scale_factor: float = 0.5
    target_heatmap: bool = True
    target_keypoint: bool = True
    num_joints_half_body: int = 8
    prob_half_body: float = 0.3

@dataclass
class BackboneConfig:
    type: str = "ViT"
    img_size: List[int] = field(default_factory=list)
    patch_size: int = 16
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    drop_path_rate: float = 0.1
    lr_decay_rate: float = 0.75

@dataclass
class KeypointHeadConfig:
    type: str = "TopdownHeatmapSimpleHead"
    in_channels: int = 768
    num_deconv_layers: int = 2
    num_deconv_filters: List[int] = field(default_factory=list)
    num_deconv_kernels: List[int] = field(default_factory=list)
    out_channels: int = 17

@dataclass
class ModelConfig:
    name: str = "vit"
    type: str = "base"
    pretrained: str = ""
    num_joints: int = 17
    image_size: List[int] = field(default_factory=list)
    heatmap_size: List[int] = field(default_factory=list)
    use_gpe: bool = False
    use_lpe: bool = False
    use_gap: bool = False

@dataclass
class LossConfig:
    hm_loss: str = "JointMSEloss"
    hm_loss_weight: float = 1.0
    kd_loss_weight: float = 1.0
    use_target_weight: bool = True

@dataclass
class KnowledgeDistillationConfig:
    enabled: bool = False
    alpha: float = 0.7
    loss_scale: float = 1.0

@dataclass
class FederatedConfig:
    method: str = "fedavg"
    enabled: bool = True
    num_clients: int = 4
    mu: float = 1.0

@dataclass
class TrainingConfig:
    lr: float = 0.001
    batch_size: int = 32
    begin_epoch: int = 0
    end_epoch: int = 211
    lr_step: List[int] = field(default_factory=list)
    lr_factor: float = 0.1
    optimizer: str = "adamW"
    shuffle: bool = True
    wd: float = 0.01

@dataclass
class TestConfig:
    batch_size: int = 32
    flip_test: bool = True
    use_gt_bbox: bool = False
    use_udp: bool = True
    oks_thre: float = 0.9
    in_vis_thre: float = 0.2

@dataclass
class EvaluationConfig:
    interval: int = 10
    metric: str = "mAP"
    save_best: str = "AP"

@dataclass
class ExperimentConfig:
    # 메타데이터
    exp_name: str = "default"
    output_dir: str = "output"
    log_dir: str = "log"
    workers: int = 4
    print_freq: int = 100
    seed: int = 42
    gpu: int = 0
    wandb: bool = False

# 간단한 등록만 수행
def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=ExperimentConfig)

if __name__ == "__main__":
    register_configs() 