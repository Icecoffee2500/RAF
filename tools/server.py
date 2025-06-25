import sys, os
home_dir = os.path.dirname(os.path.abspath(f"{__file__}/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)
import torch
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from lib.utils.utils import get_vit_optimizer
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
            out_channels=17,
        )
        
        self.logger = logging.getLogger(__name__)
        self.gpu = gpu
        
        
        if config.MODEL.KD_TARGET is not None:
            self.model = ViTPoseSFL(self.backbone, self.deconv_head, config.MODEL.KD_TARGET)
        else:
            self.model = ViTPoseSFL(self.backbone, self.deconv_head)
        
        # weight initialization
        if "mae" in checkpoint_path:
            load_checkpoint(self.model.backbone, checkpoint_path)
        else:
            self.model.custom_init_weights(self.model, checkpoint_path)
        
        self.model.cuda(args.gpu)
        self.model = DDP(
            self.model, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False
        )
        
        self.optimizer_server = get_vit_optimizer(config, self.model, extra)
        
        lr_scheduler_server = MultistepWarmUpRestargets(
            self.optimizer_server, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR
        )
    
    def train(
        self, activation, Hp, gt_heatmap, heatmap_weight, batch_idx,
        wdb, criterion, losses: AverageMeter, acc: AverageMeter, device=None
    ):
    
        # switch to train mode
        self.model.train()
        
        torch.autograd.set_detect_anomaly(True)
        loss = 0.0
        teacher_output = None
        outputs = []
        avg_acc = 0
        cnt = 0
        
        gt_heatmap = gt_heatmap.cuda(non_blocking=True, device=device)
        heatmap_weight = heatmap_weight.cuda(non_blocking=True, device=device)
        
        #---------forward prop-------------
        output, output_kd = self.model(activation, Hp)
        # print(f"output shape => {output.shape}")
        
        # calculate loss
        loss = self.get_loss(
            batch_idx, self.config, criterion, output, gt_heatmap, heatmap_weight, wdb)
        
        # calculate accuracy
        _, avg_acc, cnt, pred = accuracy(
            output.detach().cpu().numpy(), gt_heatmap.detach().cpu().numpy()
        ) # cnt는 acc가 0이상인 것의 개수
        
        #--------backward prop--------------
        self.optimizer_server.zero_grad()
        loss.backward()
        grad_norm = self.clip_grads(self.model.parameters())
        gradient_client = activation.grad.clone().detach()
        self.optimizer_server.step()
        
        # record accuracy
        acc.update(avg_acc, cnt)
        
        # measure accuracy and record loss
        losses.update(loss.item(), activation.size(0))
        
        return gradient_client, grad_norm
    
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