import torch
from omegaconf import DictConfig
from typing import Dict, Any
from datetime import datetime

from hpe.utils.resolution_utils import is_multi_resolution
from ..core.data_manager import DataManager
from ..core.model_manager import ModelManager
from ..core.training_manager import TrainingManager
from ..core.evaluation_manager import EvaluationManager
from ..core.logging_manager import LoggingManager


class FLClient:
    """리팩토링된 연합학습 클라이언트 - 컴포넌트 기반 아키텍처"""
    
    def __init__(self,
                 client_id: int,
                 cfg: DictConfig,
                 device: torch.device,
                 init_model: torch.nn.Module = None):
        """
        Args:
            client_id: 클라이언트 ID
            cfg: Hydra 설정
            device: 디바이스
            init_model: 초기 모델 (연합학습용)
        """
        self.client_id = client_id
        self.cfg = cfg
        self.device = device
        
        # 각 컴포넌트 초기화
        self._initialize_components(init_model)
        
        # 정보 로깅
        self._log_initialization_info()
    
    def _initialize_components(self, init_model):
        """모든 컴포넌트 초기화"""
        # 1. 로깅 매니저 (가장 먼저)
        self.logging_manager = LoggingManager(
            logging_cfg=self.cfg.get('logging', {}),
            wandb_cfg=self.cfg.get('wandb', None),
            client_id=self.client_id
        )
        
        # 2. 데이터 매니저
        self.data_manager = DataManager(
            dataset_cfg=self.cfg.dataset,
            client_id=self.client_id,
            image_size=self.cfg.resolution.image_size[self.client_id],
            heatmap_size=self.cfg.resolution.heatmap_size[self.client_id],
            train_batch_size=self.cfg.training.batch_size,
            valid_batch_size=self.cfg.test.batch_size,
            samples_per_split=self.cfg.federated.samples_per_client,
            workers=self.cfg.workers
        )
        
        # 3. 모델 매니저
        self.model_manager = ModelManager(
            model_cfg=self.cfg.model,
            training_cfg=self.cfg.training,
            loss_cfg=self.cfg.loss,
            device=self.device,
            init_model=init_model
        )
        
        # 4. 훈련 매니저
        self.training_manager = TrainingManager(
            training_cfg=self.cfg.training,
            federated_cfg=self.cfg.federated,
            loss_cfg=self.cfg.loss,
            device=self.device
        )
        
        # 5. 평가 매니저
        self.evaluation_manager = EvaluationManager(
            dataset_cfg=self.cfg.dataset,
            test_cfg=self.cfg.get('test', {}),
            device=self.device
        )
    
    def _log_initialization_info(self):
        """초기화 정보 로깅"""
        # 모델 정보 로깅
        model_info = self.model_manager.get_model_info()
        self.logging_manager.log_model_info(model_info)
        
        # 데이터 정보 로깅
        data_info = self.data_manager.get_dataset_info()
        self.logging_manager.log_data_info(data_info)
        
        self.logging_manager.log_info("모든 컴포넌트 초기화 완료")
    
    def train_epoch(self, epoch: int, global_weights: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """한 에포크 훈련 실행"""
        logger = self.logging_manager.get_logger()
        
        # 해상도에 따라 훈련 방식 선택
        image_size = self.cfg.resolution.image_size[self.client_id]
        
        try:
            if is_multi_resolution(image_size):
                self.logging_manager.log_info(f"다중 해상도 KD 훈련 시작 - Epoch {epoch}")
                metrics = self.training_manager.train_multi_resolution(
                    self.model_manager,
                    self.data_manager,
                    epoch,
                    logger,
                    global_weights
                )
            else:
                self.logging_manager.log_info(f"단일 해상도 훈련 시작 - Epoch {epoch}")
                metrics = self.training_manager.train_single_resolution(
                    self.model_manager,
                    self.data_manager,
                    epoch,
                    logger,
                    global_weights
                )
            
            # 메트릭 로깅
            self.logging_manager.log_training_metrics(epoch, metrics)
            
            return metrics
            
        except Exception as e:
            self.logging_manager.log_error(f"Epoch {epoch} 훈련 중 오류", e)
            raise
    
    def evaluate(self, epoch: int = 0) -> float:
        """모델 평가 실행"""
        try:
            self.logging_manager.log_info(f"평가 시작 - Epoch {epoch}")
            
            performance = self.evaluation_manager.evaluate(
                self.model_manager,
                self.data_manager,
                self.logging_manager.get_output_dir(),
                self.logging_manager.get_wandb(),
                self.logging_manager.get_logger()
            )
            
            # 평가 메트릭 로깅
            eval_metrics = self.evaluation_manager.get_evaluation_metrics()
            self.logging_manager.log_evaluation_metrics(epoch, eval_metrics, performance)
            
            return performance
            
        except Exception as e:
            self.logging_manager.log_error(f"Epoch {epoch} 평가 중 오류", e)
            return 0.0
    
    def update_model_weights(self, new_weights: Dict[str, torch.Tensor]):
        """모델 가중치 업데이트 (연합학습용)"""
        try:
            self.model_manager.set_parameters(new_weights)
            self.logging_manager.log_info("모델 가중치 업데이트 완료")
        except Exception as e:
            self.logging_manager.log_error("모델 가중치 업데이트 실패", e)
            raise
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """모델 가중치 반환 (연합학습용)"""
        return self.model_manager.get_parameters()
    
    def scheduler_step(self):
        """스케줄러 스텝"""
        self.model_manager.scheduler_step()
        self.model_manager.update_learning_rate()
    
    def get_client_info(self) -> Dict[str, Any]:
        """클라이언트 정보 반환"""
        return {
            "client_id": self.client_id,
            "device": str(self.device),
            "dataset_info": self.data_manager.get_dataset_info(),
            "model_info": self.model_manager.get_model_info(),
            "training_metrics": self.training_manager.get_training_metrics()
        }
    
    def save_checkpoint(self, epoch: int, performance: float, save_dir: str):
        """체크포인트 저장"""
        try:
            checkpoint = {
                "epoch": epoch,
                "client_id": self.client_id,
                "model_state_dict": self.model_manager.get_parameters(),
                "optimizer_state_dict": self.model_manager.get_optimizer().state_dict(),
                "performance": performance,
                "config": self.cfg
            }
            
            checkpoint_path = f"{save_dir}/client_{self.client_id}_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            self.logging_manager.log_info(f"체크포인트 저장: {checkpoint_path}")
            
        except Exception as e:
            self.logging_manager.log_error("체크포인트 저장 실패", e)
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model_manager.set_parameters(checkpoint["model_state_dict"])
            self.model_manager.get_optimizer().load_state_dict(checkpoint["optimizer_state_dict"])
            
            self.logging_manager.log_info(f"체크포인트 로드: {checkpoint_path}")
            
            return checkpoint["epoch"], checkpoint["performance"]
            
        except Exception as e:
            self.logging_manager.log_error("체크포인트 로드 실패", e)
            raise
    
    def finalize(self):
        """클라이언트 종료 처리"""
        self.logging_manager.log_info("클라이언트 종료 처리 시작")
        self.logging_manager.finalize()
    
    # 기존 인터페이스와의 호환성을 위한 프로퍼티들
    @property
    def model(self):
        """기존 코드 호환성을 위한 모델 접근"""
        return self.model_manager.get_model()
    
    @property
    def optimizer(self):
        """기존 코드 호환성을 위한 옵티마이저 접근"""
        return self.model_manager.get_optimizer()
    
    @property
    def lr_scheduler(self):
        """기존 코드 호환성을 위한 스케줄러 접근"""
        return self.model_manager.get_scheduler()
    
    # 기존 메서드명 호환성
    def train_single_resolution(self, epoch: int):
        """기존 인터페이스 호환성"""
        return self.train_epoch(epoch)
    
    def train_multi_resolution(self, epoch: int):
        """기존 인터페이스 호환성"""
        return self.train_epoch(epoch) 