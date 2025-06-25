import sys, os
home_dir = os.path.dirname(os.path.abspath(f"{__file__}/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)
import torch
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from lib.utils.utils import get_vit_optimizer, get_vit_optimizer_general
from lib.core.scheduler import MultistepWarmUpRestargets

from lib.models.backbones.vit_server import ViT_server
from lib.utils.utils import load_checkpoint
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.models.vit_pose_sfl import ViTPoseSFL
from lib.core.evaluate import accuracy
from lib.utils.average_meter import AverageMeter
from lib.core.inference import get_final_preds


from torch.nn.utils import clip_grad
import numpy as np
import logging
from datetime import datetime
import torch.nn.functional as F


class Server():
    def __init__(self, extra, config, args, checkpoint_path, gpu) -> None:
        self.config = config
        
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
        
        # weight initialization
        if "mae" in checkpoint_path:
            load_checkpoint(self.backbone, checkpoint_path)
        
        if config.MODEL.KD_TARGET is not None:
            self.model = ViTPoseSFL(self.backbone, self.deconv_head, config.MODEL.KD_TARGET)
        else:
            self.model = ViTPoseSFL(self.backbone, self.deconv_head)
        
        self.model.cuda(self.gpu)
        self.model = DDP(
            self.model, device_ids=[self.gpu], find_unused_parameters=False, broadcast_buffers=False
        )
        
        self.optimizer_server = get_vit_optimizer_general(config, self.model, extra)
    
    def train(
        self, activations, Hps, gt_heatmaps, heatmap_weights, batch_idx,
        wdb, criterion, losses: AverageMeter, acc: AverageMeter, device=None, is_proxy=False
    ):
        if is_proxy:
            activations_privacy, activations_proxy = activations
            Hps_privacy, Hps_proxy = Hps
            gt_heatmaps_privacy, gt_heatmaps_proxy = gt_heatmaps
            heatmap_weights_privacy, heatmap_weights_proxy = heatmap_weights
            
            for activation in activations_proxy:
                activation.retain_grad()
        else:
            activations_privacy = activations
            Hps_privacy = Hps
            gt_heatmaps_privacy = gt_heatmaps
            heatmap_weights_privacy = heatmap_weights
        
        for activation in activations_privacy:
            activation.retain_grad()
    
        # switch to train mode
        self.model.train()
        
        torch.autograd.set_detect_anomaly(True)
        # loss = 0.0
        teacher_output = None
        outputs = []
        avg_acc = 0
        cnt = 0
        loss_buffer = []
        loss_kd_pr_buffer = []
        loss_total_buffer = []
        gradient_client_buffer = []
        grad_norm_buff = []
        
        #---------forward prop-------------
        for idx, activation in enumerate(activations_privacy):
            # activation.retain_grad()
            output, _ = self.model(activation, Hps_privacy[idx])
            outputs.append(output)

            # calculate privacy loss
            loss = self.get_loss(
                    batch_idx, self.config, criterion, output, gt_heatmaps_privacy[idx], heatmap_weights_privacy[idx], wdb)
            loss_buffer.append(loss)

        if is_proxy == True:
            for idx, activation in enumerate(activations_proxy):
                output_pr, _ = self.model(activation, Hps_proxy[idx])

                if teacher_output is None:
                    teacher_output = output_pr

                # proxy heatmap loss
                loss_hm_pr = self.get_loss(
                    batch_idx, self.config, criterion, output_pr, gt_heatmaps_proxy[idx], heatmap_weights_proxy[idx], wdb)
                # loss_hm_pr = 0
                
                # teacher와 현재 output_pr의 shape이 다르면 output_pr의 shape을 teacher에 맞춰 줌.
                if teacher_output.shape != output_pr.shape:
                    output_pr_interpolated = F.interpolate(output_pr, size=(teacher_output.shape[2], teacher_output.shape[3]), mode='nearest')
                else:
                    output_pr_interpolated = output_pr

                # proxy knowledge distillation loss
                loss_kd_pr = self.get_loss(
                    batch_idx, self.config, criterion, output_pr_interpolated, teacher_output, heatmap_weights_proxy[idx], wdb)

                # loss_pr_buffer에 loss_hm + loss_kd 저장
                loss_kd_pr_buffer.append(loss_hm_pr + loss_kd_pr)
                # loss_kd_pr_buffer.append(loss_kd_pr)

                teacher_output = output_pr
        
        # calculate accuracy    
        _, avg_acc, cnt, pred = accuracy(
            outputs[0].detach().cpu().numpy(), gt_heatmaps_privacy[0].detach().cpu().numpy()
        ) # cnt는 acc가 0이상인 것의 개수
        
        #--------backward prop--------------
        # self.optimizer_server.zero_grad()

        for idx, loss in enumerate(loss_buffer):
            if is_proxy:
                # loss_total_buffer.append(loss + loss_kd_pr_buffer[idx]) # kd만 적용할 때
                loss_total_buffer.append((loss + loss_kd_pr_buffer[idx]) / 2) # gt 적용할 때
            else:
                loss_total_buffer.append(loss)
        
        for idx, loss in enumerate(loss_total_buffer):
            loss.backward(retain_graph=True)
            # loss.backward()
            grad_norm = self.clip_grads(self.model.parameters())
            
            gradient_client = activations_privacy[idx].grad.clone().detach()
            gradient_client_buffer.append(gradient_client)
            grad_norm_buff.append(grad_norm)
        
        self.optimizer_server.step()
        
        # record accuracy
        acc.update(avg_acc, cnt)
        
        # measure accuracy and record loss
        losses.update(loss.item(), activation.size(0))
        
        return gradient_client_buffer, grad_norm_buff
    
    def evaluate(
        self, valid_loader_len, activation, activation_flipped, Hp, Hp_flipped, gt_heatmap,
        heatmap_weight, meta, batch_idx, criterion, epoch_start_time, end,
        batch_time: AverageMeter, losses: AverageMeter, acc: AverageMeter, img_idx, all_preds, all_boxes, image_path, bbox_ids, device=None
    ):
        # switch to eval mode
        self.model.eval()
        
        #---------forward prop-------------
        output_og, output_kd = self.model(activation, Hp)
        
        if self.config.TEST.FLIP_TEST:
            features_flipped, output_kd_flipped = self.backbone(activation_flipped, Hp_flipped)
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
        if not self.config.TEST.USE_GT_BBOX:
            loss = 0.0
        else:
            loss = criterion(output_augmented, gt_heatmap, heatmap_weight)
            losses.update(loss.item(), num_images)
        
        # calculate accuracy
        _, avg_acc, cnt, pred = accuracy(
            output_augmented.detach().cpu().numpy(), gt_heatmap.detach().cpu().numpy()
        ) # cnt는 acc가 0이상인 것의 개수
        
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
            msg += f"Time[{batch_time.val:.3f}/{batch_time.avg:.3f}] "
            msg += f"Loss[{losses.val:.4f}/{losses.avg:.4f}] "
            msg += f"Accuracy[{acc.val:.3f}/{acc.avg:.3f}] "
            msg += f"ETA[{str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (valid_loader_len - batch_idx - 1)).split('.')[0]}]"
            self.logger.info(msg)

            # prefix = "{}_{}".format(os.path.join(output_dir, "val"), batch_idx)
            # save_debug_images(config, input, meta, heatmap, pred * 4, output_heatmap, prefix)
        return img_idx
    
    def get_loss(
        self, batch_idx, config, criterion, output, heatmap, heatmap_weight, wdb
    ):
        loss = self.cal_loss(
            config, criterion, output, heatmap, heatmap_weight, count=batch_idx, wdb=wdb)
        return loss
    
    def cal_loss(
        self,
        config,
        criterion: dict,
        pred_heatmap,
        gt_heatmap,
        hm_weight,
        count=None,
        wdb=None,
    ):
        loss = 0
        for k, v in criterion.items():
            if k == "heatmap":
                l = v(pred_heatmap, gt_heatmap, hm_weight, count, wdb)
                loss += l * config.LOSS.HM_LOSS_WEIGHT
        return loss
    
    def clip_grads(self, params):
        if params := list(
            filter(lambda p: p.requires_grad and p.grad is not None, params)
        ):
            grad_clip = {"max_norm": 0.003, "norm_type": 2}
            return clip_grad.clip_grad_norm_(params, **grad_clip)