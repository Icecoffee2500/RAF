import logging
from typing import Dict, Any, Optional
from datetime import datetime
from omegaconf import DictConfig

from hpe.utils.logging import create_logger_sfl, ShellColors as sc


class LoggingManager:
    """로깅 관리 전담 클래스"""
    
    def __init__(self, 
                 logging_cfg: DictConfig,
                 wandb_cfg: Optional[DictConfig] = None,
                 client_id: int = 0):
        """
        Args:
            logging_cfg: 로깅 설정
            wandb_cfg: WandB 설정 (선택적)
            client_id: 클라이언트 ID
        """
        self.logging_cfg = logging_cfg
        self.wandb_cfg = wandb_cfg
        self.client_id = client_id
        
        # 기본 logger 설정
        self.logger: logging.Logger = logging.getLogger(f"FL_Client_{self.client_id}_init")
        self.final_output_dir = None
        
        # WandB 설정
        self.wdb = None
        
        # logger와 WandB 초기화
        self._setup_logger()
        self._setup_wandb()
    
    def _setup_logger(self):
        """기본 logger 설정"""
        try:
            # 기존 create_logger_sfl 함수 사용
            from configs.hpe.config import config  # 임시로 기존 config 사용
            
            # 임시 config 파일명 생성 (실제로는 설정에서 가져와야 함)
            cfg_name = f"client_{self.client_id}_config"
            res_arg = f"client_{self.client_id}"
            
            self.logger, self.final_output_dir = create_logger_sfl(
                config, cfg_name, f"train_gpu_{self.client_id}", arg=res_arg
            )
            
            self.logger.info(f"Logger 초기화 완료 - Client {self.client_id}")
            
        except Exception as e:
            # 기본 logger로 fallback
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(f"FL_Client_{self.client_id}")
            self.logger.error(f"Logger 설정 실패, 기본 logger 사용: {e}")
    
    def _setup_wandb(self):
        """WandB 설정"""
        if self.wandb_cfg is None:
            self.logger.info("WandB가 비활성화됨")
            return
        
        try:
            import wandb
            
            # 실험명 생성
            now = datetime.now()
            timestamp = now.strftime("%m%d_%H:%M")
            
            project_name = self.wandb_cfg.project if hasattr(self.wandb_cfg, 'project') else "RAF-v2"
            run_name = f"client_{self.client_id}_{timestamp}"
            
            if hasattr(self.wandb_cfg, 'run_name'):
                run_name = f"{self.wandb_cfg.run_name}_client_{self.client_id}_{timestamp}"
            
            # WandB 초기화
            self.wdb = wandb
            self.wdb.init(
                project=project_name,
                name=run_name,
                config=self._create_wandb_config(),
                mode=self.wandb_cfg.mode if hasattr(self.wandb_cfg, 'mode') else "online",
                tags=[f"client_{self.client_id}"]
            )
            
            self.logger.info(f"{sc.COLOR_GREEN}WandB 초기화 성공 - Project: {project_name}, Run: {run_name}{sc.ENDC}")
            
        except Exception as e:
            self.logger.warning(f"{sc.COLOR_RED}WandB 초기화 실패: {e}{sc.ENDC}")
            self.logger.info(f"{sc.COLOR_YELLOW}WandB 없이 실행을 계속합니다{sc.ENDC}")
            self.wdb = None
    
    def _create_wandb_config(self) -> Dict[str, Any]:
        """WandB용 설정 생성"""
        config_dict = {
            "client_id": self.client_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        # wandb_cfg에서 추가 설정 가져오기
        if hasattr(self.wandb_cfg, 'config'):
            config_dict.update(self.wandb_cfg.config)
        
        return config_dict
    
    def log_training_metrics(self, epoch: int, metrics: Dict[str, Any], client_prefix: bool = True):
        """훈련 메트릭 로깅"""
        prefix = f"[Client {self.client_id}] " if client_prefix else ""
        
        # 기본 logger
        self.logger.info(f"{prefix}Epoch {epoch} 훈련 완료:")
        self.logger.info(f"{prefix}  - Avg Loss: {metrics.get('avg_loss', 0):.4f}")
        self.logger.info(f"{prefix}  - Avg Accuracy: {metrics.get('avg_acc', 0):.3f}")
        if 'epoch_time' in metrics:
            self.logger.info(f"{prefix}  - Epoch Time: {metrics['epoch_time']}")
        
        # WandB 로깅
        if self.wdb:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_name = f"client_{self.client_id}_{key}" if client_prefix else key
                    wandb_metrics[metric_name] = value
            
            wandb_metrics['epoch'] = epoch
            self.wdb.log(wandb_metrics)
    
    def log_evaluation_metrics(self, epoch: int, metrics: Dict[str, Any], 
                             performance: float, client_prefix: bool = True):
        """평가 메트릭 로깅"""
        prefix = f"[Client {self.client_id}] " if client_prefix else ""
        
        # 기본 logger
        self.logger.info(f"{prefix}Epoch {epoch} 평가 완료:")
        self.logger.info(f"{prefix}  - Performance: {performance:.4f}")
        self.logger.info(f"{prefix}  - Avg Loss: {metrics.get('avg_loss', 0):.4f}")
        self.logger.info(f"{prefix}  - Avg Accuracy: {metrics.get('avg_acc', 0):.3f}")
        
        # WandB 로깅
        if self.wdb:
            wandb_metrics = {
                'epoch': epoch,
                f"client_{self.client_id}_performance" if client_prefix else "performance": performance,
            }
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_name = f"client_{self.client_id}_{key}" if client_prefix else key
                    wandb_metrics[metric_name] = value
            
            self.wdb.log(wandb_metrics)
    
    def log_federated_metrics(self, epoch: int, 
                            client_performances: Dict[int, float],
                            avg_performance: float):
        """연합학습 전체 메트릭 로깅"""
        self.logger.info(f"=== Epoch {epoch} 연합학습 결과 ===")
        
        for client_id, perf in client_performances.items():
            self.logger.info(f"  Client {client_id}: {perf:.4f}")
        
        self.logger.info(f"  평균 성능: {avg_performance:.4f}")
        
        if self.wdb:
            wandb_metrics = {
                'epoch': epoch,
                'federated_avg_performance': avg_performance
            }
            
            for client_id, perf in client_performances.items():
                wandb_metrics[f'client_{client_id}_final_performance'] = perf
            
            self.wdb.log(wandb_metrics)
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """모델 정보 로깅"""
        self.logger.info(f"=== Client {self.client_id} 모델 정보 ===")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        if self.wdb:
            self.wdb.config.update(model_info, allow_val_change=True)  # type: ignore[attr-defined]
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """데이터 정보 로깅"""
        self.logger.info(f"=== Client {self.client_id} 데이터 정보 ===")
        for key, value in data_info.items():
            self.logger.info(f"  {key}: {value}")
        
        if self.wdb:
            # 중복 키 업데이트 허용 (예: client_id 등)
            self.wdb.config.update(data_info, allow_val_change=True)  # type: ignore[attr-defined]
    
    def log_error(self, error_msg: str, exception: Exception = None):
        """에러 로깅"""
        self.logger.error(f"[Client {self.client_id}] {error_msg}")
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
    
    def log_info(self, message: str):
        """일반 정보 로깅"""
        self.logger.info(f"[Client {self.client_id}] {message}")
    
    def log_warning(self, message: str):
        """경고 로깅"""
        self.logger.warning(f"[Client {self.client_id}] {message}")
    
    def get_logger(self) -> logging.Logger:
        """Logger 객체 반환"""
        return self.logger
    
    def get_wandb(self):
        """WandB 객체 반환"""
        return self.wdb
    
    def get_output_dir(self) -> str:
        """출력 디렉토리 반환"""
        return self.final_output_dir if self.final_output_dir else "./output"
    
    def finalize(self):
        """로깅 종료 처리"""
        if self.wdb:
            try:
                self.wdb.finish()
                self.logger.info("WandB 로깅 종료")
            except Exception as e:
                self.logger.error(f"WandB 종료 중 오류: {e}") 