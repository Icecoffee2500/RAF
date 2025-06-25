import sys, os
home_dir = os.path.dirname(os.path.abspath(f"{__file__}/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)
import torch
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from lib.utils.utils import get_vit_optimizer, get_vit_optimizer_general, get_vit_server_optimizer
from lib.core.scheduler import MultistepWarmUpRestargets

from lib.models.backbones.vit_server import ViT_server
from lib.utils.utils import load_checkpoint
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.models.vit_pose_sfl import ViTPoseSFL
from lib.core.evaluate import accuracy
from lib.utils.average_meter import AverageMeter
from lib.core.inference import get_final_preds
from lib.utils.utils import get_loss

from torch.nn.utils import clip_grad
import numpy as np
import logging
from datetime import datetime
import torch.nn.functional as F
from lib.utils.utils import ShellColors as sc
from lib.models.losses.mse_loss import JointsKLDLoss
from lib.utils.utils import clone_parameters
from collections import OrderedDict


class Server:
    def __init__(
        self,
        extra,
        config,
        gpu,
        device,
        pretrained_path=None,
        ckpt_server_path=None) -> None:
        
        self.config = config
        self.criterion = get_loss(config)
        # self.kd_use=True if self.config.KD_USE else False
        
        self.backbone = ViT_server(
            embed_dim=extra["backbone"]["embed_dim"],
            depth=extra["backbone"]["depth"],
            num_heads=extra["backbone"]["num_heads"],
            qkv_bias=True,
            drop_path_rate=extra["backbone"]["drop_path_rate"],
            use_gpe=config.MODEL.USE_GPE,
            use_lpe=config.MODEL.USE_LPE,
        )
        
        self.deconv_head = TopdownHeatmapSimpleHead(
            in_channels=extra["keypoint_head"]["in_channels"],
            num_deconv_layers=extra["keypoint_head"]["num_deconv_layers"],
            num_deconv_filters=extra["keypoint_head"]["num_deconv_filters"],
            num_deconv_kernels=extra["keypoint_head"]["num_deconv_kernels"],
            extra=dict(final_conv_kernel=1),
            out_channels=config.MODEL.NUM_JOINTS,
        )
        
        self.logger = logging.getLogger(__name__)
        self.gpu = gpu
        self.device = device
        
        # weight initialization
        if pretrained_path is not None and "mae" in pretrained_path:
            load_checkpoint(self.backbone, pretrained_path)
        
        if config.MODEL.KD_TARGET is not None:
            self.model = ViTPoseSFL(self.backbone, self.deconv_head, config.MODEL.KD_TARGET)
        else:
            self.model = ViTPoseSFL(self.backbone, self.deconv_head)
        
        if ckpt_server_path is not None:
            server_state_dict = torch.load(ckpt_server_path, map_location='cpu')
            self.model.load_state_dict(server_state_dict)
        
        self.model.to(self.device)
        # self.model = DDP(
        #     self.model, device_ids=[self.gpu], find_unused_parameters=False, broadcast_buffers=False
        # )
        
        # self.optimizer_server = get_vit_optimizer_general(config, self.model, extra)
        self.optimizer_server = get_vit_server_optimizer(config, self.model, extra, client_backbone_block_num=1)
        
        self.lr_scheduler_server = MultistepWarmUpRestargets(
            self.optimizer_server, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR
        )
        
        # TODO: Temp codes!!
        self.criterion_kd = JointsKLDLoss().to(device)
        
        self.server_params_dict: OrderedDict[str : torch.Tensor] = None
        # self.server_params_dict = OrderedDict(
        #     self.model.state_dict(keep_vars=True)
        # )
        self.trainable_server_params: list[torch.Tensor] = None
    
    def set_parameters(
        self,
        model_params: OrderedDict[str, torch.Tensor],
    ):
        self.trainable_server_params = list(
            filter(lambda p: p.requires_grad, model_params.values())
        )
        # print(f"trainable_server_params: {self.trainable_server_params}")
    
    def train(self, activations, gt_heatmaps, heatmap_weights, kd_use=False):
        for activation in activations:
            activation.retain_grad()
    
        # switch to train mode
        self.model.train()
        
        teacher_output = None
        loss_buffer = []
        acc_buffer = []
        gradient_client_buffer = []
        
        for idx, activation in enumerate(activations):
            output, _ = self.model(activation)

            # calculate privacy loss
            loss = self.get_total_loss(
                self.config,
                self.criterion,
                output,
                gt_heatmaps[idx],
                heatmap_weights[idx],
            )
            
            # for Knowledge Distillation
            if kd_use == True:
                if teacher_output is None:
                    teacher_output = output
                    
                # teacher와 현재 output의 shape이 다르면 output의 shape을 teacher에 맞춰 줌.
                if teacher_output.shape != output.shape:
                    output_interpolated = F.interpolate(output, size=(teacher_output.shape[2], teacher_output.shape[3]), mode='nearest')
                    # knowledge distillation loss
                    loss_kd = self.get_total_loss(
                        self.config,
                        self.criterion,
                        output_interpolated,
                        teacher_output.detach(),
                        heatmap_weights[idx],
                    )
                    loss = loss + loss_kd
                
                teacher_output = output
                
            # loss에 loss_kd 추가
            loss_buffer.append(loss)
            
            # calculate accuracy    
            _, avg_acc, cnt, pred = accuracy(
                output.detach().cpu().numpy(), gt_heatmaps[idx].detach().cpu().numpy()
            ) # cnt는 acc가 0이상인 것의 개수
            
            # record accuracy
            acc_buffer.append((avg_acc, cnt))
            
        
        self.optimizer_server.zero_grad()
        #--------backward prop--------------
        for idx, loss in enumerate(loss_buffer):
            loss.backward(retain_graph=True)
            # loss.backward()
            # grad_norm = self.clip_grads(self.model.parameters())
            gradient_client = activations[idx].grad.clone().detach()
            gradient_client_buffer.append(gradient_client)

        self.optimizer_server.step()
        return gradient_client_buffer, loss_buffer, acc_buffer
    
    def train_proxy(self, activations, gt_heatmaps, heatmap_weights, kd_use=False, mu=1.0, momentum=None, pseudo_activations=None):  
        for activation in activations:
            activation.retain_grad()
    
        # switch to train mode
        self.model.train()
        
        # total_loss = 0.0
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        teacher_output = None
        total_avg_acc = 0.0
        total_cnt = 0
        gradient_client_buffer = []
        
        #---------forward prop-------------
        for idx, activation in enumerate(activations):
            # activation.retain_grad()
            output, _ = self.model(activation)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # pseudo heatmap
            if pseudo_activations is not None:
                with torch.no_grad():
                    pseudo_heatmap, _ = self.model(pseudo_activations[idx])

            # calculate privacy loss
            loss = self.get_total_loss(
                self.config,
                self.criterion,
                output,
                # pseudo_heatmap.detach(),
                gt_heatmaps[idx],
                heatmap_weights[idx],
            )
            
            # for Knowledge Distillation
            if kd_use == True:
                if teacher_output is None:
                    teacher_output = output
                    
                # teacher와 현재 output의 shape이 다르면 output의 shape을 teacher에 맞춰 줌.
                if teacher_output.shape != output.shape:
                    output_interpolated = F.interpolate(output, size=(teacher_output.shape[2], teacher_output.shape[3]), mode='nearest')
                    # knowledge distillation loss
                    loss_kd = self.get_total_loss(
                        self.config,
                        self.criterion,
                        output_interpolated,
                        teacher_output.detach(),
                        heatmap_weights[idx],
                    )
                    # Online Knowledge Distillation for Efficient Pose Estimation # kd loss is kl div loss!!
                    # https://github.com/zhengli97/OKDHP/tree/master
                    # loss_kd = self.criterion_kd(output_interpolated, teacher_output.detach(), heatmap_weights[idx])
                    
                    alpha = self.config.KD_ALPHA
                    # print(f"client {idx} loss: {loss} / loss_kl_div: {loss_kd}")
                    # beta = 10
                    # beta = 100
                    # beta = 1000
                    # beta = 5000
                    loss = loss * alpha + loss_kd * (1 - alpha)
                    # loss = loss * alpha + loss_kd * beta
                    # loss = loss_kd
                    # loss = loss + loss_kd
                
                teacher_output = output
                
            total_loss = total_loss + loss
            
            # calculate accuracy    
            _, avg_acc, cnt, pred = accuracy(
                output.detach().cpu().numpy(), gt_heatmaps[idx].detach().cpu().numpy()
            ) # cnt는 acc가 0이상인 것의 개수
            
            # record accuracy
            total_avg_acc = total_avg_acc + avg_acc
            total_cnt = total_cnt + cnt
        
        total_loss = total_loss / len(activations)
        total_avg_acc = total_avg_acc / len(activations)
        
        self.optimizer_server.zero_grad()
        #--------backward prop--------------
        # if total_loss.item() != 0.0:
        for idx, activation in enumerate(activations):
            total_loss.backward(retain_graph=True)
            # for proximal alternative to gt loss
            # for w, w_g in zip(self.model.parameters(), self.trainable_server_params):
            #     # w.grad.data += mu * (w_g.data - w.data) # 잘못됨
            #     w.grad.data += mu * (w.data - w_g.data) # 이게 맞음.
            if activation.grad is None:
                activation.grad = torch.zeros_like(activation)
            gradient_client = activation.grad.clone().detach()
            gradient_client_buffer.append(gradient_client)
        # # server는 proxy-kd loss를 업데이트하지 않음.
        # self.optimizer_server.zero_grad()
        self.optimizer_server.step()
        # EMA update
        # if momentum is not None:
        #     with torch.no_grad():
        #         for param, param_g in zip(self.model.parameters(), self.trainable_server_params):
        #             param.data.mul_(1. - momentum).add_((momentum) * param_g.detach().data)
        
        return gradient_client_buffer, total_loss, total_avg_acc, total_cnt
    
    def train_proxy_takd(self, activation_pair, gt_heatmap_pair, heatmap_weight_pair, kd_use=False):
        if kd_use:
            activation_pair[1].retain_grad()
        else:
            activation_pair.retain_grad()
    
        # switch to train mode
        self.model.train()
        
        #---------forward prop-------------
        if kd_use:
            output_teacher, _ = self.model(activation_pair[0])
            output_student, _ = self.model(activation_pair[1])
        else:
            output, _ = self.model(activation_pair)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # calculate gt loss
        loss = self.get_total_loss(
            self.config,
            self.criterion,
            output_student if kd_use else output,
            gt_heatmap_pair[1] if kd_use else gt_heatmap_pair,
            heatmap_weight_pair[1] if kd_use else heatmap_weight_pair,
        )
        
        # for Knowledge Distillation
        if kd_use:
            # teacher와 현재 output의 shape이 다르면 output의 shape을 teacher에 맞춰 줌.
            output_student_interpolated = F.interpolate(output_student, size=(output_teacher.shape[2], output_teacher.shape[3]), mode='nearest')
            # knowledge distillation loss
            loss_kd = self.get_total_loss(
                self.config,
                self.criterion,
                output_student_interpolated,
                output_teacher.detach(),
                heatmap_weight_pair[1],
            )
            
            alpha = self.config.KD_ALPHA
            loss = loss * alpha + loss_kd * (1 - alpha)
            # loss = loss + loss_kd
            
        # calculate accuracy    
        if kd_use:
            _, avg_acc, cnt, pred = accuracy(
                output_student.detach().cpu().numpy(), gt_heatmap_pair[1].detach().cpu().numpy()
            ) # cnt는 acc가 0이상인 것의 개수
        else:
            _, avg_acc, cnt, pred = accuracy(
                output.detach().cpu().numpy(), gt_heatmap_pair.detach().cpu().numpy()
            ) # cnt는 acc가 0이상인 것의 개수
        
        #--------backward prop--------------
        self.optimizer_server.zero_grad()
        loss.backward(retain_graph=True)
        if kd_use:
            gradient_client = activation_pair[1].grad.clone().detach()
        else:
            gradient_client = activation_pair.grad.clone().detach()
        
        # 이 optimizer 부분이 좀 걸리긴 함...
        self.optimizer_server.step()
        
        return gradient_client, loss, avg_acc, cnt
    
    def forward_and_backward(self, activation, gt_heatmap, heatmap_weight):  
        # self.lr_scheduler_server.step()
        activation.retain_grad()
    
        # switch to train mode
        self.model.train()
        
        #---------forward prop-------------
        output, _ = self.model(activation)

        # calculate privacy loss
        loss = self.get_total_loss(
            self.config,
            self.criterion,
            output,
            gt_heatmap,
            heatmap_weight,
        )
        
        # calculate accuracy    
        _, avg_acc, cnt, pred = accuracy(
            output.detach().cpu().numpy(), gt_heatmap.detach().cpu().numpy()
        ) # cnt는 acc가 0이상인 것의 개수
        
        self.optimizer_server.zero_grad()
        #--------backward prop--------------
        loss.backward(retain_graph=True)
        # loss.backward()
        grad_norm = self.clip_grads(self.model.parameters())
        gradient_client = activation.grad.clone().detach()
        
        self.optimizer_server.step()
        return gradient_client, grad_norm, loss, avg_acc, cnt
    
    
    def evaluate(
        self, valid_loader_len, activation, activation_flipped, gt_heatmap,
        heatmap_weight, meta, batch_idx, epoch_start_time, end,
        batch_time: AverageMeter, losses: AverageMeter, acc: AverageMeter, img_idx, all_preds, all_boxes, image_path, bbox_ids, device=None
    ):
        # switch to eval mode
        self.model.eval()
        
        #---------forward prop-------------
        output_og, output_kd = self.model(activation)
        
        if self.config.TEST.FLIP_TEST:
            features_flipped, output_kd_flipped = self.backbone(activation_flipped)
            output_flipped_heatmap = self.deconv_head.inference_model(
                features_flipped, meta["flip_pairs"]
            )
            output_augmented = (
                output_og + torch.from_numpy(output_flipped_heatmap.copy()).cuda()
            ) * 0.5
        
        gt_heatmap = gt_heatmap.cuda(non_blocking=True)
        heatmap_weight = heatmap_weight.cuda(non_blocking=True)
        
        num_images = activation.size(0) # 이 batch의 img 개수
        
        # calculte loss
        # if not self.config.TEST.USE_GT_BBOX:
        #     loss = 0.0
        # else:
        #     loss = self.criterion(output_augmented, gt_heatmap, heatmap_weight)
        #     losses.update(loss.item(), num_images)
        
        # loss = self.criterion(output_augmented, gt_heatmap, heatmap_weight)
        loss = self.get_total_loss(
            self.config,
            self.criterion,
            output_augmented,
            gt_heatmap,
            heatmap_weight
        )
        losses.update(loss.item(), num_images)
        
        # calculate accuracy
        _, avg_acc, cnt, pred = accuracy(
            output_augmented.detach().cpu().numpy(), gt_heatmap.detach().cpu().numpy()
        ) # cnt는 acc가 0이상인 것의 개수
        
        acc.update(avg_acc, cnt)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        c = meta["center"].numpy()
        s = meta["scale"].numpy()
        score = meta["score"].numpy()
        preds, maxvals = get_final_preds(
            self.config, output_augmented.clone().cpu().numpy(), c, s, 11)
        
        all_preds[img_idx : img_idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[img_idx : img_idx + num_images, :, 2:3] = maxvals

        # double check this all_boxes parts
        all_boxes[img_idx : img_idx + num_images, 0:2] = c[:, 0:2]
        all_boxes[img_idx : img_idx + num_images, 2:4] = s[:, 0:2]
        all_boxes[img_idx : img_idx + num_images, 4] = np.prod(s * 200, 1) # area
        all_boxes[img_idx : img_idx + num_images, 5] = score
        image_path.extend(meta["image"])
        bbox_ids.extend(meta["bbox_id"])

        img_idx += num_images

        if batch_idx % self.config.PRINT_FREQ == 0 and self.gpu == 0:
            msg = f"Test[{batch_idx}/{valid_loader_len}] "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Time{sc.ENDC} {batch_time.avg:.3f}(s) "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Batch Speed{sc.ENDC} {activation.size(0)/batch_time.avg:.1f}(samples/s) "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Loss{sc.ENDC} {losses.avg:.4f} "
            msg += f"| {sc.COLOR_LIGHT_BLUE}Avg Accuracy{sc.ENDC} {acc.avg:.3f} "
            
            elapsed_time = str((datetime.now()-epoch_start_time) / (batch_idx+1) * (valid_loader_len-batch_idx-1)).split('.')[0].split(':')
            msg += f"| {sc.COLOR_LIGHT_BLUE}ETA{sc.ENDC} {elapsed_time[0]}h {elapsed_time[1]}m {elapsed_time[2]}s |"
            
            self.logger.info(msg)

            # prefix = "{}_{}".format(os.path.join(output_dir, "val"), batch_idx)
            # save_debug_images(config, input, meta, heatmap, pred * 4, output_heatmap, prefix)
        return img_idx, losses, acc
    
    def get_total_loss(
        self, config, criterion, output, heatmap, heatmap_weight
    ):
        loss = self.cal_loss(
            config, criterion, output, heatmap, heatmap_weight)
        return loss
    
    def cal_loss(
        self,
        config,
        criterion: dict,
        pred_heatmap,
        gt_heatmap,
        hm_weight,
    ):
        loss = 0
        for k, v in criterion.items():
            if k == "heatmap":
                l = v(pred_heatmap, gt_heatmap, hm_weight)
                loss += l * config.LOSS.HM_LOSS_WEIGHT
        return loss
    
    def clip_grads(self, params):
        if params := list(
            filter(lambda p: p.requires_grad and p.grad is not None, params)
        ):
            grad_clip = {"max_norm": 0.003, "norm_type": 2}
            return clip_grad.clip_grad_norm_(params, **grad_clip)