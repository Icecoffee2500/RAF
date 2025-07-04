import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Dict, Any
from copy import deepcopy

from hpe.models import ViT, TopdownHeatmapSimpleHead, ViTPose
from hpe.utils.model_utils import get_vit_optimizer, get_loss
from hpe.federated.scheduler import MultistepWarmUpRestargets


class ModelManager:
    """모델 생성 및 관리 전담 클래스"""
    
    def __init__(self, 
                 model_cfg: DictConfig,
                 training_cfg: DictConfig,
                 loss_cfg: DictConfig,
                 device: torch.device,
                 init_model: nn.Module = None):
        """
        Args:
            model_cfg: 모델 설정
            training_cfg: 훈련 설정
            loss_cfg: 손실함수 설정
            device: 디바이스
            init_model: 초기 모델 (연합학습용)
        """
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.loss_cfg = loss_cfg
        self.device = device
        
        # 모델 생성 또는 복사
        if init_model is not None:
            self.model = deepcopy(init_model)
        else:
            self.model = self._create_model()
        
        self.model.to(device)
        
        # 옵티마이저, 스케줄러, 손실함수 생성
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        self.criterion = self._create_loss_functions()
    
    def _create_model(self) -> nn.Module:
        """모델 생성"""
        # Backbone 생성
        backbone = ViT(
            img_size=self.model_cfg.backbone.img_size,
            patch_size=self.model_cfg.backbone.patch_size,
            embed_dim=self.model_cfg.backbone.embed_dim,
            in_channels=3,
            num_heads=self.model_cfg.backbone.num_heads,
            depth=self.model_cfg.backbone.depth,
            qkv_bias=True,
            drop_path_rate=self.model_cfg.backbone.drop_path_rate,
            use_gpe=self.model_cfg.use_gpe,
            use_lpe=self.model_cfg.use_lpe,
            use_gap=self.model_cfg.use_gap,
        )
        
        # Head 생성
        keypoint_head = TopdownHeatmapSimpleHead(
            in_channels=self.model_cfg.keypoint_head.in_channels,
            num_deconv_layers=self.model_cfg.keypoint_head.num_deconv_layers,
            num_deconv_filters=self.model_cfg.keypoint_head.num_deconv_filters,
            num_deconv_kernels=self.model_cfg.keypoint_head.num_deconv_kernels,
            extra=dict(final_conv_kernel=1),
            out_channels=self.model_cfg.keypoint_head.out_channels,
        )
        
        # 전체 모델 조합
        model = ViTPose(backbone, keypoint_head)
        return model
    
    def _create_optimizer(self):
        """옵티마이저 생성"""
        # 임시로 legacy config 형식으로 변환하여 기존 함수 사용
        legacy_config = self._create_legacy_config()
        extra = {
            "backbone": {
                "lr_decay_rate": self.model_cfg.backbone.lr_decay_rate,
                "embed_dim": self.model_cfg.backbone.embed_dim,
                "depth": self.model_cfg.backbone.depth
            }
        }
        return get_vit_optimizer(legacy_config, self.model, extra)
    
    def _create_scheduler(self):
        """학습률 스케줄러 생성"""
        return MultistepWarmUpRestargets(
            self.optimizer, 
            milestones=self.training_cfg.lr_step, 
            gamma=self.training_cfg.lr_factor
        )
    
    def _create_loss_functions(self) -> Dict[str, Any]:
        """손실 함수들 생성"""
        legacy_config = self._create_legacy_config()
        return get_loss(legacy_config)
    
    def _create_legacy_config(self):
        """임시: 기존 함수들이 요구하는 legacy config 생성"""
        from easydict import EasyDict as edict
        
        config = edict()
        
        # TRAIN 설정
        config.TRAIN = edict()
        config.TRAIN.LR = self.training_cfg.lr
        config.TRAIN.OPTIMIZER = self.training_cfg.optimizer
        config.TRAIN.LR_STEP = self.training_cfg.lr_step
        config.TRAIN.LR_FACTOR = self.training_cfg.lr_factor
        if hasattr(self.training_cfg, 'wd'):
            config.TRAIN.WD = self.training_cfg.wd
        else:
            config.TRAIN.WD = 0.01
        
        # LOSS 설정
        config.LOSS = edict()
        config.LOSS.HM_LOSS = self.loss_cfg.hm_loss
        config.LOSS.HM_LOSS_WEIGHT = self.loss_cfg.hm_loss_weight
        config.LOSS.KD_LOSS = self.loss_cfg.kd.kd_loss
        config.LOSS.KD_LOSS_WEIGHT = self.loss_cfg.kd.kd_loss_weight
        config.LOSS.KD_LOSS_SCALE = self.loss_cfg.kd.loss_scale
        
        # MODEL 설정 
        config.MODEL = edict()
        config.MODEL.NAME = self.model_cfg.name
        config.MODEL.TYPE = self.model_cfg.type
        
        return config
    
    def get_model(self) -> nn.Module:
        """모델 반환"""
        return self.model
    
    def get_optimizer(self):
        """옵티마이저 반환"""
        return self.optimizer
    
    def get_scheduler(self):
        """스케줄러 반환"""
        return self.lr_scheduler
    
    def get_criterion(self) -> Dict[str, Any]:
        """손실함수들 반환"""
        return self.criterion
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """모델 파라미터 반환 (연합학습용)"""
        return self.model.state_dict()
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """모델 파라미터 설정 (연합학습용)"""
        self.model.load_state_dict(parameters)
    
    def train_mode(self):
        """모델을 훈련 모드로 설정"""
        self.model.train()
    
    def eval_mode(self):
        """모델을 평가 모드로 설정"""
        self.model.eval()
    
    def scheduler_step(self):
        """스케줄러 한 스텝 진행"""
        self.lr_scheduler.step()
    
    def update_learning_rate(self):
        """학습률 업데이트 (스케줄러 적용)"""
        lr_ = self.lr_scheduler.get_lr()
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = lr_[i]
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": f"{self.model_cfg.name}_{self.model_cfg.type}",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_embed_dim": self.model_cfg.backbone.embed_dim,
            "backbone_depth": self.model_cfg.backbone.depth,
            "num_heads": self.model_cfg.backbone.num_heads,
            "output_joints": self.model_cfg.keypoint_head.out_channels,
            "device": str(self.device)
        } 