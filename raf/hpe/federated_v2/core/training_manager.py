import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad
from omegaconf import DictConfig
from typing import Dict, Any, List, Tuple
from datetime import datetime

from hpe.utils.training_utils import AverageMeter, gpu_timer
from hpe.utils.evaluate import accuracy
from hpe.utils.resolution_utils import is_multi_resolution
from hpe.federated.loss_fns import JointsMSELoss #JointsKLDLoss


class TrainingManager:
    """훈련 로직 전담 클래스"""
    
    def __init__(self, 
                 training_cfg: DictConfig,
                 federated_cfg: DictConfig,
                 loss_cfg: DictConfig,
                 device: torch.device):
        """
        Args:
            training_cfg: 훈련 설정
            federated_cfg: 연합학습 설정
            loss_cfg: 손실함수 설정
            device: 디바이스
        """
        self.training_cfg = training_cfg
        self.federated_cfg = federated_cfg
        self.loss_cfg = loss_cfg
        self.device = device
        
        # 메트릭 추적용
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        
        # Knowledge Distillation용 손실함수
        self.criterion_kd = JointsMSELoss().to(device)
    
    def train_single_resolution(self, 
                              model_manager,
                              data_manager,
                              epoch: int,
                              logger,
                              global_weights: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """단일 해상도 훈련"""
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        
        model = model_manager.get_model()
        optimizer = model_manager.get_optimizer()
        criterion = model_manager.get_criterion()
        train_loader = data_manager.get_train_loader()
        
        model.train()
        
        # FedProx 설정
        mu = 0.0
        if self.federated_cfg.method == "fedprox" and global_weights is not None:
            mu = self.federated_cfg.mu
        
        epoch_start_time = datetime.now()
        batch_num = len(train_loader)
        
        for batch_idx, (img, heatmap, heatmap_weight, meta) in enumerate(train_loader):
            etime = gpu_timer(
                lambda: self._train_step_single(
                    model, optimizer, criterion,
                    img, heatmap, heatmap_weight,
                    global_weights=global_weights,
                    mu=mu
                )
            )
            batch_time.update(etime)
            
            # 로깅
            if batch_idx % self.training_cfg.print_freq == 0:
                self._log_training_progress(
                    epoch, batch_idx, batch_num, 
                    batch_time, epoch_start_time, logger
                )
        
        return {
            "avg_loss": self.losses.avg,
            "avg_acc": self.acc.avg,
            "epoch_time": datetime.now() - epoch_start_time
        }
    
    def train_multi_resolution(self,
                             model_manager,
                             data_manager, 
                             epoch: int,
                             logger,
                             global_weights: Dict[str, torch.Tensor] = None) -> Dict[str, Any]:
        """다중 해상도 + Knowledge Distillation 훈련"""
        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        
        model = model_manager.get_model()
        optimizer = model_manager.get_optimizer()
        criterion = model_manager.get_criterion()
        train_loader = data_manager.get_train_loader()
        
        model.train()
        
        # FedProx 설정
        mu = 0.0
        if self.federated_cfg.method == "fedprox" and global_weights is not None:
            mu = self.federated_cfg.mu
        
        epoch_start_time = datetime.now()
        batch_num = len(train_loader)
        
        for batch_idx, (imgs, heatmaps, heatmap_weights, meta) in enumerate(train_loader):
            etime = gpu_timer(
                lambda: self._train_step_multi(
                    model, optimizer, criterion,
                    imgs, heatmaps, heatmap_weights,
                    global_weights=global_weights,
                    mu=mu
                )
            )
            batch_time.update(etime)
            
            # 로깅
            if batch_idx % self.training_cfg.print_freq == 0:
                self._log_training_progress(
                    epoch, batch_idx, batch_num,
                    batch_time, epoch_start_time, logger
                )
        
        return {
            "avg_loss": self.losses.avg,
            "avg_acc": self.acc.avg,
            "epoch_time": datetime.now() - epoch_start_time
        }
    
    def _train_step_single(self, model, optimizer, criterion,
                          img, heatmap, heatmap_weight,
                          global_weights=None, mu=0.0):
        """단일 해상도 훈련 스텝"""
        # 데이터 GPU로 이동
        img = img.to(self.device)
        heatmap = heatmap.to(self.device)
        heatmap_weight = heatmap_weight.to(self.device)
        
        # Forward pass
        output = model(img)
        
        # 손실 계산
        loss = self._calculate_loss(criterion, output, heatmap, heatmap_weight)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        self._clip_gradients(model.parameters())
        
        # FedProx regularization
        if global_weights is not None and mu > 0.0:
            self._apply_fedprox_regularization(model, global_weights, mu)
        
        # Optimizer step
        optimizer.step()
        
        # 정확도 계산
        _, avg_acc, cnt, _ = accuracy(
            output.detach().cpu().numpy(), 
            heatmap.detach().cpu().numpy()
        )
        
        # 메트릭 업데이트
        self.acc.update(avg_acc, cnt)
        self.losses.update(loss.item(), img.size(0))
    
    def _train_step_multi(self, model, optimizer, criterion,
                         imgs, heatmaps, heatmap_weights,
                         global_weights=None, mu=0.0):
        """다중 해상도 + KD 훈련 스텝"""
        teacher_output = None
        total_avg_acc = 0.0
        total_cnt = 0
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 데이터 GPU로 이동
        imgs = [img.to(self.device) for img in imgs]
        heatmaps = [heatmap.to(self.device) for heatmap in heatmaps]
        heatmap_weights = [heatmap_weight.to(self.device) for heatmap_weight in heatmap_weights]
        
        for idx, img in enumerate(imgs):
            output = model(img)
            heatmap = heatmaps[idx]
            heatmap_weight = heatmap_weights[idx]
            
            # Ground truth 손실
            loss_gt = self._calculate_loss(criterion, output, heatmap, heatmap_weight)
            
            # Knowledge Distillation 손실
            if teacher_output is not None and self.loss_cfg.kd.enabled:
                output_interpolated = F.interpolate(
                    output, 
                    size=(teacher_output.shape[2], teacher_output.shape[3]), 
                    mode='nearest'
                )
                
                loss_kd = self._calculate_loss(
                    criterion, output_interpolated, 
                    teacher_output.detach(), heatmap_weight
                )
                
                # alpha = self.federated_cfg.kd.alpha
                # loss = loss_gt * alpha + loss_kd * (1 - alpha)
                loss = loss_gt * self.loss_cfg.hm_loss_weight + loss_kd * self.loss_cfg.kd.kd_loss_weight
            else:
                loss = loss_gt
            
            # Teacher output 업데이트
            teacher_output = output
            
            # 정확도 계산
            _, avg_acc, cnt, _ = accuracy(
                output.detach().cpu().numpy(),
                heatmap.detach().cpu().numpy()
            )
            
            # 누적
            total_loss = total_loss + loss
            total_avg_acc += avg_acc
            total_cnt += cnt
        
        # 평균 계산
        loss_scale = self.loss_cfg.kd.loss_scale if hasattr(self.loss_cfg.kd, 'loss_scale') else 1.0
        total_avg_loss = total_loss * loss_scale
        total_avg_acc = total_avg_acc / len(imgs)
        
        # Backward pass
        optimizer.zero_grad()
        total_avg_loss.backward()
        
        # Gradient clipping
        self._clip_gradients(model.parameters())
        
        # FedProx regularization
        if global_weights is not None and mu > 0.0:
            self._apply_fedprox_regularization(model, global_weights, mu)
        
        # Optimizer step
        optimizer.step()
        
        # 메트릭 업데이트
        self.acc.update(total_avg_acc, total_cnt)
        self.losses.update(total_avg_loss.item(), imgs[0].size(0))
    
    def _calculate_loss(self, criterion, pred_heatmap, gt_heatmap, hm_weight):
        """손실 계산"""
        loss = 0
        for k, v in criterion.items():
            if k == "heatmap":
                l = v(pred_heatmap, gt_heatmap, hm_weight)
                loss += l * self.loss_cfg.hm_loss_weight
        return loss
    
    def _clip_gradients(self, params):
        """그래디언트 클리핑"""
        grad_clip = {"max_norm": 0.003, "norm_type": 2}
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **grad_clip)
    
    def _apply_fedprox_regularization(self, model, global_weights, mu):
        """FedProx 정규화 적용"""
        for name, local_param in model.named_parameters():
            if name not in global_weights:
                continue
            
            global_param = global_weights[name]
            if local_param.shape != global_param.shape:
                continue
            
            if local_param.grad is None:
                local_param.grad = torch.zeros_like(local_param)
            
            with torch.no_grad():
                delta = mu * (global_param.detach() - local_param.detach())
                local_param.grad.add_(delta)
    
    def _log_training_progress(self, epoch, batch_idx, batch_num, 
                             batch_time, epoch_start_time, logger):
        """훈련 진행상황 로깅"""
        msg = f"\tEpoch[{epoch}][{batch_idx}/{batch_num}] "
        msg += f"| Batch Time {batch_time.avg:.3f}(s) "
        msg += f"| Avg Loss {self.losses.avg:.4f} "
        msg += f"| Avg Accuracy {self.acc.avg:.3f} "
        
        elapsed_time = str((datetime.now()-epoch_start_time) / (batch_idx+1) * (batch_num-batch_idx-1)).split('.')[0].split(':')
        msg += f"| ETA {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
        
        logger.info(msg)
    
    def get_training_metrics(self) -> Dict[str, float]:
        """훈련 메트릭 반환"""
        return {
            "avg_loss": self.losses.avg,
            "avg_accuracy": self.acc.avg,
            "num_samples": self.losses.count
        } 