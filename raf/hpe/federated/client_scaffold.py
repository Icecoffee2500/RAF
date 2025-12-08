# Works as a local trainer
from typing import OrderedDict
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad
from datetime import datetime
import time
import logging
import numpy as np
from copy import deepcopy

from configs.hpe.config import get_model_name
from hpe.utils.training_utils import AverageMeter, gpu_timer
from hpe.federated.scheduler import MultistepWarmUpRestargets
from hpe.dataset.utils.builder import build_train_val_dataloader, build_split_dataset
from hpe.utils.evaluate import accuracy
from hpe.dataset.coco import COCODataset as coco
from hpe.dataset.mpii import MPIIDataset as mpii
from hpe.utils.model_utils import get_vit_optimizer, get_loss
from hpe.utils.logging import ShellColors as sc
from hpe.utils.post_processing import get_final_preds
from hpe.federated.loss_fns import JointsKLDLoss
from hpe.federated.client import FLClient

class FLClientScaffold(FLClient):
    def __init__(
        self,
        client_id,
        config,
        device,
        init_model,
        extra,
        wdb,
        logger,
        im_size,
        hm_size,
        batch_size,
        is_proxy=False,
        samples_per_split=0,
    ):
        super().__init__(client_id, config, device, init_model, extra, wdb, logger, im_size, hm_size, batch_size, is_proxy, samples_per_split)

        self.c_local: list[torch.Tensor] = []
        self.c_diff = []
        # self.local_lr = self.lr_scheduler.get_lr()
        self.local_lr = 1

    def train_single_resolution(self, epoch, global_params: OrderedDict[str, torch.Tensor], c_global):
        self.model.load_state_dict(global_params, strict=False)
        self.model.train()

        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        
        # SCAFFOLD 적용 ------------------------------
        if self.c_local == []:
            self.c_diff = c_global
        else:
            # c_diff는 aggregate된 c_global과 아직 학습하지 않은 c_local의 차이. (c_local은 이전 round의 c_plus)
            self.c_diff = []
            for c_l, c_g in zip(self.c_local, c_global):
                self.c_diff.append(c_g - c_l)
        # -------------------------------------------
        
        epoch_start_time = datetime.now()
        batch_num = len(self.train_loader)

        update_count = 0
        
        for batch_idx, (img, heatmap, heatmap_weight, meta) in enumerate(self.train_loader):
            update_count += 1
            etime = gpu_timer(
                lambda: self._train_step_single(img, heatmap, heatmap_weight)
            )
            batch_time.update(etime)
            
            # logging
            self._log_while_training(
                idx=self.client_id,
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=self.logger,
                train_batch_idx=batch_idx,
                train_batch_size=batch_num,
                total_batch_time=batch_time,
            )
        
        # SCAFFOLD 적용 (y_delta, c_plus, c_delta 계산) ------------------------------
        with torch.no_grad():
            trainable_parameters = filter(
                lambda p: p.requires_grad, global_params.values() # global_params 중에서 requires_grad가 true인 파라미터만 필터링.
            )

            if self.c_local == []: # (처음에만) c_local을 초기화.
                # self.c_local = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]
                self.c_local = [torch.zeros_like(param, device=self.device) for param in self.model.state_dict().values()]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            for param_l, param_g in zip(self.model.state_dict().values(), global_params.values()):
                y_delta.append(param_l - param_g)
            
            # compute c_plus # Option II version
            coef = 1 / (update_count * self.local_lr)
            for c_l, c_g, diff in zip(self.c_local, c_global, y_delta):
                c_plus.append(c_l - c_g - coef * diff)

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local):
                c_delta.append(c_p - c_l)

            self.c_local = c_plus

        # # y_delta가 tensor라면
        # print("y_delta[0] shape:", y_delta[0].size())
        # print("y_delta[0] dtype:", y_delta[0].dtype)
        # print("y_delta[0] elements:", y_delta[0].numel())

        # # c_delta가 tensor라면
        # print("c_delta[0] shape:", c_delta[0].size())
        # print("c_delta[0] dtype:", c_delta[0].dtype)
        # print("c_delta[0] elements:", c_delta[0].numel())
        
        res = (y_delta, self.dataset_length, c_delta)

        return res
    
    def _train_step_single(self, img, heatmap, heatmap_weight, **proximal):

        # torch.cuda.reset_peak_memory_stats(self.device)
        # memory_start = time.perf_counter()
        # forward propagation
        img, heatmap, heatmap_weight = img.to(self.device), heatmap.to(self.device), heatmap_weight.to(self.device)
        output = self.model(img)
        
        # calculate privacy loss
        loss = self.cal_loss(
            self.config,
            self.criterion,
            output,
            heatmap,
            heatmap_weight,
        )
        
        # backward propagation
        self.optimizer.zero_grad()
        loss.backward()

        # SCAFFOLD 적용 (Local update) ----------------
        with torch.no_grad():
            # for param, c_d in zip(self.model.parameters(), self.c_diff):
            for param, c_d in zip(self.model.state_dict().values(), self.c_diff):
                if not param.requires_grad:
                    continue
                param.grad.add_(c_d.detach())
        # -------------------------------------------
        
        # calculate gradient norm
        grad_norm = self.clip_grads(self.model.parameters())
        
        # step optimizer
        self.optimizer.step()

        # torch.cuda.synchronize(self.device)
        # end = time.perf_counter()

        # peak_alloc = torch.cuda.max_memory_allocated(self.device)
        # current_alloc = torch.cuda.memory_allocated(self.device)
        # reserved = torch.cuda.memory_reserved(self.device)

        # print(f"step time: {end-memory_start:.3f}s")
        # print(f"GPU peak allocated: {peak_alloc/1024**2:.2f} MB")
        # print(f"GPU currently allocated: {current_alloc/1024**2:.2f} MB")
        # print(f"GPU reserved (cached): {reserved/1024**2:.2f} MB")
        # print(torch.cuda.memory_summary(device=self.device, abbreviated=True))
        
        # calculate accuracy
        _, avg_acc, cnt, pred = accuracy(
            output.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
        ) # cnt는 acc가 0이상인 것의 개수
        
        # record accuracy
        self.acc.update(avg_acc, cnt)
        self.losses.update(loss.item(), img.size(0))
    
    def train_multi_resolution(self, epoch, global_params: OrderedDict[str, torch.Tensor], c_global):
        self.model.load_state_dict(global_params, strict=False)
        self.model.train()

        batch_time = AverageMeter()
        self.losses.reset()
        self.acc.reset()
        
        # SCAFFOLD 적용 ------------------------------
        if self.c_local == []:
            self.c_diff = c_global
        else:
            # c_diff는 aggregate된 c_global과 아직 학습하지 않은 c_local의 차이. (c_local은 이전 round의 c_plus)
            self.c_diff = []
            for c_l, c_g in zip(self.c_local, c_global):
                self.c_diff.append(c_g - c_l)
        # -------------------------------------------
        
        epoch_start_time = datetime.now()
        batch_num = len(self.train_loader)

        update_count = 0
        
        for batch_idx, (imgs, heatmaps, heatmap_weights, meta) in enumerate(self.train_loader):
            update_count += 1
            etime = gpu_timer(
                lambda: self._train_step_multi(imgs, heatmaps, heatmap_weights)
            )
            batch_time.update(etime)
            
            # logging
            self._log_while_training(
                idx=self.client_id,
                epoch=epoch,
                epoch_start_time=epoch_start_time,
                logger=self.logger,
                train_batch_idx=batch_idx,
                train_batch_size=batch_num,
                total_batch_time=batch_time,
            )
        
        # SCAFFOLD 적용 (y_delta, c_plus, c_delta 계산) ------------------------------
        with torch.no_grad():
            trainable_parameters = filter(
                lambda p: p.requires_grad, global_params.values() # global_params 중에서 requires_grad가 true인 파라미터만 필터링.
            )

            if self.c_local == []: # (처음에만) c_local을 초기화.
                # self.c_local = [torch.zeros_like(param, device=self.device) for param in self.model.parameters()]
                self.c_local = [torch.zeros_like(param, device=self.device) for param in self.model.state_dict().values()]

            y_delta = []
            c_plus = []
            c_delta = []

            # compute y_delta (difference of model before and after training)
            for param_l, param_g in zip(self.model.state_dict().values(), global_params.values()):
                y_delta.append(param_l - param_g)
            
            # compute c_plus # Option II version
            coef = 1 / (update_count * self.local_lr)
            for c_l, c_g, diff in zip(self.c_local, c_global, y_delta):
                c_plus.append(c_l - c_g - coef * diff)

            # compute c_delta
            for c_p, c_l in zip(c_plus, self.c_local):
                c_delta.append(c_p - c_l)

            self.c_local = c_plus
        
        res = (y_delta, self.dataset_length, c_delta)

        return res
    
    def _train_step_multi(self, imgs, heatmaps, heatmap_weights, **proximal):
        # forward propagation
        teacher_output = None
        total_avg_acc = 0.0
        total_cnt = 0
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        imgs = [img.to(self.device) for img in imgs]
        heatmaps = [heatmap.to(self.device) for heatmap in heatmaps]
        heatmap_weights = [heatmap_weight.to(self.device) for heatmap_weight in heatmap_weights]
        
        for idx, img in enumerate(imgs):
            output = self.model(img)
            heatmap = heatmaps[idx]
            heatmap_weight = heatmap_weights[idx]
            
            # calculate gt loss
            # heatmap loss
            loss_gt = self.cal_loss(
                self.config,
                self.criterion, # MSE
                output,
                heatmap,
                heatmap_weight,
            )
            
            # knolwedge distillation loss
            if teacher_output is not None:
                output_interpolated = F.interpolate(output, size=(teacher_output.shape[2], teacher_output.shape[3]), mode='nearest')
                
                # knowledge distillation loss (MSE)
                loss_kd = self.cal_loss(
                    self.config,
                    self.criterion,
                    output_interpolated,
                    teacher_output.detach(),
                    heatmap_weight,
                )
                
                alpha = self.config.KD_ALPHA
                loss = loss_gt * alpha + loss_kd * (1 - alpha)
                # print(f"[{img.shape[2]}x{img.shape[3]}] KD Loss Used")
                # loss = loss_gt + loss_kd
            else:
                loss = loss_gt
            
            teacher_output = output
            
            # calculate accuracy
            _, avg_acc, cnt, pred = accuracy(
                output.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
            ) # cnt는 acc가 0이상인 것의 개수
        
            # sum loss, acc, cnt
            total_loss = total_loss + loss
            total_avg_acc = total_avg_acc + avg_acc
            total_cnt = total_cnt + cnt
        
        # averaging loss, acc
        # total_avg_loss = total_loss / len(imgs)
        # total_avg_loss = total_loss
        loss_scale = self.config.LOSS_SCALE
        
        total_avg_loss = total_loss * loss_scale
        total_avg_acc = total_avg_acc / len(imgs)
        
        # backward propagation
        self.optimizer.zero_grad()
        total_avg_loss.backward()

        # SCAFFOLD 적용 (Local update) ----------------
        with torch.no_grad():
            # for param, c_d in zip(self.model.parameters(), self.c_diff):
            for param, c_d in zip(self.model.state_dict().values(), self.c_diff):
                if not param.requires_grad:
                    continue
                param.grad.add_(c_d.detach())
        # -------------------------------------------
        
        # step optimizer
        self.optimizer.step()
        
        # record accuracy
        self.acc.update(total_avg_acc, total_cnt)
        self.losses.update(total_avg_loss.item(), imgs[0].size(0))
    
    
