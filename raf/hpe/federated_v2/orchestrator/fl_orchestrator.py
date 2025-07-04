import torch
import os
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig
from datetime import datetime

from federated.server import FedServer
from ..client.fl_client import FLClient
from ..core.logging_manager import LoggingManager
from hpe.utils.checkpoint_utils import save_checkpoint


class FLOrchestrator:
    """연합학습 전체 프로세스 오케스트레이터"""
    
    def __init__(self, 
                 cfg: DictConfig,
                 global_model: torch.nn.Module,
                 device: torch.device):
        """
        Args:
            cfg: Hydra 설정
            global_model: 글로벌 모델
            device: 디바이스
        """
        self.cfg = cfg
        self.global_model = global_model
        self.device = device
        
        # 연합학습 서버
        self.fed_server = FedServer()
        
        # 전체 로깅 매니저 (서버용)
        self.logging_manager = LoggingManager(
            logging_cfg=cfg.get('logging', {}),
            wandb_cfg=cfg.get('wandb', None),
            client_id=-1  # 서버는 -1로 구분
        )
        
        # 클라이언트들
        self.clients: List[FLClient] = []
        self._create_clients()
        
        # 성능 추적
        self.best_avg_performance = 0.0
        self.performance_history = []
        
        self.logging_manager.log_info(f"연합학습 오케스트레이터 초기화 완료 - {len(self.clients)}개 클라이언트")
    
    def _create_clients(self):
        """클라이언트들 생성"""
        for client_id in range(self.cfg.federated.num_clients):
            try:
                client = FLClient(
                    client_id=client_id,
                    cfg=self.cfg,
                    device=self.device,
                    init_model=self.global_model
                )
                self.clients.append(client)
                self.logging_manager.log_info(f"Client {client_id} 생성 완료")
                
            except Exception as e:
                self.logging_manager.log_error(f"Client {client_id} 생성 실패", e)
                raise
    
    def run_federated_learning(self) -> Dict[str, Any]:
        """연합학습 전체 실행"""
        self.logging_manager.log_info("=== 연합학습 시작 ===")
        
        start_time = datetime.now()
        results = {
            "epoch_results": [],
            "best_performance": 0.0,
            "final_performance": 0.0,
            "total_time": None
        }
        
        try:
            for epoch in range(self.cfg.training.begin_epoch, self.cfg.training.end_epoch):
                epoch_start_time = datetime.now()
                
                # 에포크 시작 로깅
                self.logging_manager.log_info(f"\n{'='*50}")
                self.logging_manager.log_info(f"Epoch {epoch} 시작")
                self.logging_manager.log_info(f"{'='*50}")
                
                # 스케줄러 스텝
                self._scheduler_step()
                
                # 훈련 단계
                training_results = self._training_phase(epoch)
                
                # 집계 단계
                self._aggregation_phase(epoch)
                
                # 평가 단계
                evaluation_results = self._evaluation_phase(epoch)
                
                # 에포크 결과 정리
                epoch_result = {
                    "epoch": epoch,
                    "training": training_results,
                    "evaluation": evaluation_results,
                    "epoch_time": datetime.now() - epoch_start_time
                }
                results["epoch_results"].append(epoch_result)
                
                # 베스트 모델 체크
                avg_performance = evaluation_results["avg_performance"]
                if avg_performance > self.best_avg_performance:
                    self._save_best_model(epoch, avg_performance)
                    self.best_avg_performance = avg_performance
                    results["best_performance"] = avg_performance
                
                self.performance_history.append(avg_performance)
                
                # 에포크 종료 로깅
                epoch_time = datetime.now() - epoch_start_time
                self.logging_manager.log_info(f"Epoch {epoch} 완료 - 시간: {epoch_time}")
            
            # 최종 결과
            results["final_performance"] = self.performance_history[-1] if self.performance_history else 0.0
            results["total_time"] = datetime.now() - start_time
            
            self._save_final_model()
            self._log_final_results(results)
            
        except Exception as e:
            self.logging_manager.log_error("연합학습 실행 중 오류", e)
            raise
        
        finally:
            self._finalize_clients()
        
        return results
    
    def _scheduler_step(self):
        """모든 클라이언트의 스케줄러 스텝"""
        for client in self.clients:
            client.scheduler_step()
    
    def _training_phase(self, epoch: int) -> Dict[str, Any]:
        """훈련 단계"""
        self.logging_manager.log_info(f"Epoch {epoch} 훈련 단계 시작")
        
        client_metrics = {}
        total_loss = 0.0
        total_acc = 0.0
        
        for client in self.clients:
            try:
                # 글로벌 가중치 준비 (FedProx용)
                global_weights = self.global_model.state_dict() if self.cfg.federated.method == "fedprox" else None
                
                # 클라이언트 훈련
                metrics = client.train_epoch(epoch, global_weights)
                client_metrics[client.client_id] = metrics
                
                total_loss += metrics["avg_loss"]
                total_acc += metrics["avg_acc"]
                
            except Exception as e:
                self.logging_manager.log_error(f"Client {client.client_id} 훈련 중 오류", e)
                raise
        
        # 평균 메트릭 계산
        num_clients = len(self.clients)
        avg_metrics = {
            "avg_loss": total_loss / num_clients,
            "avg_accuracy": total_acc / num_clients,
            "client_metrics": client_metrics
        }
        
        self.logging_manager.log_info(f"훈련 완료 - 평균 Loss: {avg_metrics['avg_loss']:.4f}, 평균 Acc: {avg_metrics['avg_accuracy']:.3f}")
        
        return avg_metrics
    
    def _aggregation_phase(self, epoch: int):
        """집계 단계"""
        self.logging_manager.log_info("모델 가중치 집계 시작")
        
        # 클라이언트 가중치 수집
        client_weights = []
        for client in self.clients:
            weights = client.get_model_weights()
            client_weights.append(weights)
        
        # 연합 평균
        aggregated_weights = self.fed_server.aggregate(
            self.logging_manager.get_logger(), 
            client_weights
        )
        
        # 글로벌 모델 업데이트
        self.global_model.load_state_dict(aggregated_weights)
        
        # 클라이언트 모델들 업데이트
        for client in self.clients:
            client.update_model_weights(aggregated_weights)
        
        self.logging_manager.log_info("모델 가중치 집계 및 브로드캐스트 완료")
    
    def _evaluation_phase(self, epoch: int) -> Dict[str, Any]:
        """평가 단계"""
        if epoch % self.cfg.evaluation.interval != 0:
            return {"avg_performance": 0.0, "client_performances": {}}
        
        self.logging_manager.log_info(f"Epoch {epoch} 평가 단계 시작")
        
        client_performances = {}
        total_performance = 0.0
        
        # 글로벌 모델을 평가 모드로
        self.global_model.eval()
        
        for client in self.clients:
            try:
                performance = client.evaluate(epoch)
                client_performances[client.client_id] = performance
                total_performance += performance
                
                self.logging_manager.log_info(f"Client {client.client_id} 성능: {performance:.4f}")
                
            except Exception as e:
                self.logging_manager.log_error(f"Client {client.client_id} 평가 중 오류", e)
                client_performances[client.client_id] = 0.0
        
        # 평균 성능 계산
        avg_performance = total_performance / len(self.clients)
        
        # 연합학습 메트릭 로깅
        self.logging_manager.log_federated_metrics(epoch, client_performances, avg_performance)
        
        return {
            "avg_performance": avg_performance,
            "client_performances": client_performances
        }
    
    def _save_best_model(self, epoch: int, performance: float):
        """베스트 모델 저장"""
        self.logging_manager.log_info(f"새로운 베스트 성능: {self.best_avg_performance:.4f} -> {performance:.4f}")
        
        output_dir = self.logging_manager.get_output_dir()
        
        try:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_client": f"{self.cfg.model.name}_{self.cfg.model.type}",
                    "state_dict": self.global_model.state_dict(),
                    "perf": performance,
                    "optimizer": self.clients[0].optimizer.state_dict(),
                    "HM_LOSS": self.cfg.loss.hm_loss,
                },
                output_dir,
            )
            
            self.logging_manager.log_info(f"베스트 모델 저장 완료: {output_dir}")
            
        except Exception as e:
            self.logging_manager.log_error("베스트 모델 저장 실패", e)
    
    def _save_final_model(self):
        """최종 모델 저장"""
        output_dir = self.logging_manager.get_output_dir()
        final_model_path = os.path.join(output_dir, "final_global_model.pt")
        
        try:
            torch.save(self.global_model.state_dict(), final_model_path)
            self.logging_manager.log_info(f"최종 모델 저장: {final_model_path}")
        except Exception as e:
            self.logging_manager.log_error("최종 모델 저장 실패", e)
    
    def _log_final_results(self, results: Dict[str, Any]):
        """최종 결과 로깅"""
        self.logging_manager.log_info("\n" + "="*60)
        self.logging_manager.log_info("연합학습 완료 - 최종 결과")
        self.logging_manager.log_info("="*60)
        self.logging_manager.log_info(f"베스트 성능: {results['best_performance']:.4f}")
        self.logging_manager.log_info(f"최종 성능: {results['final_performance']:.4f}")
        self.logging_manager.log_info(f"총 소요시간: {results['total_time']}")
        self.logging_manager.log_info(f"총 에포크: {len(results['epoch_results'])}")
        
        # WandB에 최종 결과 로깅
        if self.logging_manager.get_wandb():
            self.logging_manager.get_wandb().log({
                "final_best_performance": results['best_performance'],
                "final_last_performance": results['final_performance'],
                "total_epochs": len(results['epoch_results'])
            })
    
    def _finalize_clients(self):
        """모든 클라이언트 종료 처리"""
        for client in self.clients:
            try:
                client.finalize()
            except Exception as e:
                self.logging_manager.log_error(f"Client {client.client_id} 종료 처리 실패", e)
        
        self.logging_manager.finalize()
    
    def get_client(self, client_id: int) -> Optional[FLClient]:
        """특정 클라이언트 반환"""
        if 0 <= client_id < len(self.clients):
            return self.clients[client_id]
        return None
    
    def get_global_model(self) -> torch.nn.Module:
        """글로벌 모델 반환"""
        return self.global_model
    
    def get_performance_history(self) -> List[float]:
        """성능 히스토리 반환"""
        return self.performance_history.copy() 