# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
import numpy as np
import torch

from torch.nn.functional import softplus
from torch.nn.utils import clip_grad
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
from datetime import datetime

from hpe.core.config import get_model_name
from hpe.core.evaluate import accuracy
from hpe.core.inference import get_final_preds
from hpe.utils.vis import save_debug_images
from hpe.core.inference import post_dark_udp
from hpe.core.inference import get_max_preds

import torch.nn.functional as F


logger = logging.getLogger(__name__)


def get_loss_and_uncertainty(batch_idx, config, criterion, output, heatmap, heatmap_target, target_joints, target_joints_vis, output_kd, teacher_output, wdb):
    uncertainty = None # (batch,2,64,48)
    uncertainty_map = None

    if config.LOSS.UNCERTAINTY:
        pred_keys, uncertainty, unc_norm_heatmap, output = output
        uncertainty_map = softplus(uncertainty) # shape과 range도 변하지 않지만 non-linearity 추가. (모든 값이 부드러운 양수가 됨)

        if uncertainty.shape[-2:] == output.shape[-2:]:
            if config.LOSS.USE_INDEXING:
                # kp_ = np.round(target.detach().cpu().numpy() / 4)
                # keypoint max version
                if not config.MODEL.USE_EXP_KP:
                    kp_, _ = get_max_preds(output.detach().cpu().numpy()) # kp => (B, 17, 2) # 2는 x, y좌표
                    kp_ = post_dark_udp(kp_, output.detach().cpu().numpy(), kernel=11)
                    pred_keys = torch.from_numpy(kp_).cuda()
                    kp_ = np.round(kp_)
                else:
                    kp_ = np.round(pred_keys.detach().cpu().numpy())

                x = np.clip(kp_[:, :, 0], 0, config.MODEL.HEATMAP_SIZE[0] - 1)
                y = np.clip(kp_[:, :, 1], 0, config.MODEL.HEATMAP_SIZE[1] - 1)

                # Uncertainty Map has 1 channel
                sigma_x = torch.diagonal(
                    uncertainty_map[:, 0, y, x], dim1=0, dim2=1
                ).permute(1, 0) # (48, 64)?

                uncertainty = torch.cat(
                    [sigma_x.unsqueeze(-1), sigma_x.unsqueeze(-1)], dim=-1
                ) # (48, 64, 2)

        if config.LOSS.HM_LOSS != "":
            loss = cal_loss(
                config,
                criterion,
                output,
                heatmap,
                heatmap_target,
                pred_keys,
                target_joints,
                target_joints_vis,
                uncertainty,
                output_kd=output_kd,
                teacher_output=teacher_output,
                count=batch_idx,
                wdb=wdb,
            )
        else:
            loss = cal_loss(
                config,
                criterion,
                0,
                0,
                heatmap_target,
                pred_keys,
                target_joints,
                target_joints_vis,
                uncertainty,
                output_kd=output_kd,
                teacher_output=teacher_output,
                count=batch_idx,
                wdb=wdb,
            )
    else:
        loss = cal_loss(config, criterion, output, heatmap, heatmap_target, output_kd=output_kd, teacher_output=teacher_output, count=batch_idx, wdb=wdb)
    return output, loss, uncertainty, uncertainty_map

def train(
    config, train_loader, model, criterion, optimizer, epoch, output_dir, device=None, wdb=None, warmup_scheduler=None,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    
    if isinstance(config.MODEL.IMAGE_SIZE[0], (list, np.ndarray)):
        res_num = len(config.MODEL.IMAGE_SIZE)
    else:
        res_num = 1

    # switch to train mode
    model.train()

    # freeze할 module이 있으면 그 내부에 batch norm을 freeze 시킴!
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            result = [True if i in name else False for i in config.MODEL.FREEZE_NAME]
            if any(result):
                # print(name)
                module.eval()

    end = time.time()
    epoch_start_time = datetime.now()

    # AMP setting
    if config.MODEL.USE_AMP:
        torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True)

        logger.info("Using Automatic Mixed Precision (AMP) training...")
        # Create a GradScaler object for FP16 training
        scaler = GradScaler()

    for batch_idx, (inputs, target_joints, target_joints_vis, heatmaps, heatmap_target, meta) in enumerate(train_loader):
        
        torch.autograd.set_detect_anomaly(True)
        loss = 0.0
        teacher_output = None
        outputs = []
        avg_acc = 0
        cnt = 0
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        # 밑에서 for문 적용을 위해서 single resolution이면 list를 덧씌워줌.
        if res_num == 1:
            inputs = [inputs]
            target_joints = [target_joints]
            heatmaps = [heatmaps]
            
        # if isinstance(target_joints, list):
        target_joints = [target_joint.cuda(non_blocking=True) for target_joint in target_joints]
        try:
            heatmaps = [heatmap.cuda(non_blocking=True, device=device) for heatmap in heatmaps]
        except AttributeError as e:
            print(f"AttributeError, heatmaps type => {type(heatmaps)}, heatmaps[0][0] shape => {heatmaps[0][0].shape}")
            raise e
        
        target_joints_vis = target_joints_vis.cuda(non_blocking=True)
        heatmap_target = heatmap_target.cuda(non_blocking=True, device=device)
        
        # multi resolution setting
        for idx in range(res_num):
            # 각 list에서 해당 값들만 빼오기.
            input = inputs[idx]
            target_joint = target_joints[idx]
            heatmap = heatmaps[idx]
            
            # compute output
            if config.MODEL.USE_AMP:
                with autocast():
                    # output = model(input)
                    output, output_kd = model(input)
                    
                    # 첫 resolution은 distillation이 적용이 안되게끔 하기.
                    if teacher_output is None:
                        teacher_output = output_kd
                    
                    # TODO 깔끔하게
                    output, loss_curr, uncertainty, uncertainty_map = get_loss_and_uncertainty(batch_idx, config, criterion, output, heatmap, heatmap_target, target_joint, target_joints_vis, output_kd, teacher_output, wdb)
                    loss = loss + loss_curr
                    teacher_output = output_kd.detach()
                
                # loss가 resolution의 모든 loss를 다 더했을 때만 scaler update
                if idx == res_num - 1:
                    loss = loss / res_num
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grads(model.parameters())

                    if warmup_scheduler is not None:
                        if warmup_scheduler.count < config.TRAIN.WARMUP_ITERS:
                            # warmup_scheduler.step()
                            warmup_scheduler.count += 1
                            scaler.step(warmup_scheduler)
                        else:
                            scaler.step(warmup_scheduler)
                    else:
                        scaler.step(optimizer)

                    scaler.update()
            else:
                # output = model(input)
                output, output_kd = model(input)
                # 첫 resolution은 distillation이 적용이 안되게끔 하기.
                if teacher_output is None:
                    teacher_output = output_kd
                
                if teacher_output.shape != output_kd.shape:
                    output_kd_resized = F.interpolate(output_kd, size=(teacher_output.shape[2], teacher_output.shape[3]), mode='nearest')
                else:
                    output_kd_resized = output_kd
                
                output, loss_curr, uncertainty, uncertainty_map = get_loss_and_uncertainty(batch_idx, config, criterion, output, heatmap, heatmap_target, target_joint, target_joints_vis, output_kd_resized, teacher_output, wdb)
                
                loss = loss + loss_curr
                teacher_output = output_kd.detach()
                
                # loss가 resolution의 모든 loss를 다 더했을 때만 scaler update
                if idx == res_num - 1:
                    loss = loss / res_num
                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = clip_grads(model.parameters())
                    if warmup_scheduler is not None:
                        if warmup_scheduler.count < config.TRAIN.WARMUP_ITERS:
                            warmup_scheduler.step()
                            warmup_scheduler.count += 1
                        else:
                            optimizer.step()
                    else:
                        optimizer.step()
        
            _, avg_acc_curr, cnt_curr, pred_curr = accuracy( # cnt는 acc가 0이상인 것의 개수
                output.detach().cpu().numpy(), heatmap.detach().cpu().numpy()
            )
            avg_acc = avg_acc + avg_acc_curr
            cnt = cnt + cnt_curr
            if idx == 0:
                pred = pred_curr
            outputs.append(output)
        
        avg_acc = avg_acc / res_num
        acc.update(avg_acc, cnt)
        
        # measure accuracy and record loss
        losses.update(loss.item(), inputs[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config.PRINT_FREQ == 0:
            msg = f"Epoch[{epoch}][{batch_idx}/{len(train_loader)}] "
            msg += f"Time[{batch_time.val:.3f}/{batch_time.avg:.3f}] "
            msg += f"Speed[{inputs[0].size(0)*res_num/batch_time.val:.1f} samples/s] "
            msg += f"Data[{data_time.val:.3f}/{data_time.avg:.3f}] "
            msg += f"Loss[{losses.val:.4f}/{losses.avg:.4f}] "
            msg += f"Grad Norm[{grad_norm:.4f}] "
            msg += f"Accuracy[{acc.val:.3f}/{acc.avg:.3f}] "
            
            if config.LOSS.UNCERTAINTY:
                msg += f"Mean Uncertainty[{torch.mean(uncertainty).item():.4f}] "
            
            if warmup_scheduler is not None:
                msg += f"Warmup LR[{warmup_scheduler.get_lr()[-1]:4f}] "

            msg += f"ETA[{str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (len(train_loader) - batch_idx - 1)).split('.')[0]}] "
            logger.info(msg)
            prefix = "{}_{}".format(os.path.join(output_dir, "train"), batch_idx)
            save_debug_images(config, inputs[0], meta, target_joints[0], pred * 4, outputs[0], prefix)
            if wdb:
                wdb.log({"Avg loss": losses.avg})
                wdb.log({"Accuracy": acc.avg})

    if wdb and config.LOSS.UNCERTAINTY:
        if epoch % 2 == 0:
            wdb.log({"uncertainty_map": to_wandb_img(wdb, uncertainty_map, True)})
            wdb.log({"Img": to_wandb_img(wdb, inputs.permute(0, 2, 3, 1), True)})
    
    logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")

def validate(
    config,
    val_loader,
    val_dataset,
    model,
    criterion,
    output_dir,
    backbone=None,
    keypoint_head=None,
    wdb=None,
):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    num_samples = len(val_dataset) # validation dataset의 전체 object 개수
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32) # 모든 object의 keypoint를 담을 변수
    all_boxes = np.zeros((num_samples, 6)) # 모든 object의 bounding box를 담을 변수
    image_path = [] # image의 파일 경로
    bbox_ids = [] # bounding box의 id (object마다 bounding box의 번호가 부여됨.)
    idx = 0 # batch마다 idx를 업데이트 하면서 이미지의 순서를 알려줌.
    filenames = [] # filename, pose tracking에서 쓰는거라 여기서는 상관 x
    imgnums = [] # imgnum, pose tracking에서 쓰는거라 여기서는 상관 x
    
    if isinstance(config.MODEL.IMAGE_SIZE[0], (list, np.ndarray)):
        # print(f"image size type => {type(config.MODEL.IMAGE_SIZE[0])}")
        res_num = len(config.MODEL.IMAGE_SIZE)
    else:
        res_num = 1
    
    with torch.no_grad():
        end = time.time()
        epoch_start_time = datetime.now()
        for batch_idx, (inputs, targets, target_joints_vis, heatmaps, heatmap_target, meta) in enumerate(
            val_loader
        ):
            # 기존의 convention을 해치지 않기 위해서 다음과 같이 데이터 초기화
            if res_num == 1:
                input = inputs
                target = targets
                heatmap = heatmaps
            else:
                input = inputs[0]
                target = targets[0]
                heatmap = heatmaps[0]
            
            output_heatmap, output_kd = model.forward(input)
            if config.LOSS.UNCERTAINTY:
                pred_keys, uncertainty, output_norm_heatmap, output_heatmap = output_heatmap
                
            # Changed
            if config.TEST.FLIP_TEST:
                img_flipped = input.flip(3).cuda()
                features_flipped, output_kd_flipped = backbone(img_flipped)
                output_flipped_heatmap = keypoint_head.inference_model(
                    features_flipped, meta["flip_pairs"]
                )
                output_heatmap = (
                    output_heatmap + torch.from_numpy(output_flipped_heatmap.copy()).cuda()
                ) * 0.5

            heatmap = heatmap.cuda(non_blocking=True)
            heatmap_target = heatmap_target.cuda(non_blocking=True)

            num_images = input.size(0) # 이 batch의 input 개수
            
            if not config.TEST.USE_GT_BBOX:
                loss = 0.0
            else:
                # measure record loss
                loss = criterion(output_heatmap, heatmap, heatmap_target)
                losses.update(loss.item(), num_images)


            # measure accuracy
            _, avg_acc, cnt, pred = accuracy(output_heatmap.cpu().numpy(), heatmap.cpu().numpy())
            acc.update(avg_acc, cnt)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta["center"].numpy()
            s = meta["scale"].numpy()
            score = meta["score"].numpy()
            preds, maxvals = get_final_preds(config, output_heatmap.clone().cpu().numpy(), c, s, 11)
            
            all_preds[idx : idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx : idx + num_images, :, 2:3] = maxvals

            # double check this all_boxes parts
            all_boxes[idx : idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx : idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx : idx + num_images, 4] = np.prod(s * 200, 1) # area
            all_boxes[idx : idx + num_images, 5] = score
            image_path.extend(meta["image"]) # no problem
            bbox_ids.extend(meta["bbox_id"]) # no problem

            if config.DATASET.DATASET == "posetrack":
                filenames.extend(meta["filename"])
                imgnums.extend(meta["imgnum"].numpy())

            idx += num_images

            if batch_idx % config.PRINT_FREQ == 0:
                msg = f"Test[{batch_idx}/{len(val_loader)}] "
                msg += f"Time[{batch_time.val:.3f}/{batch_time.avg:.3f}] "
                msg += f"Loss[{losses.val:.4f}/{losses.avg:.4f}] "
                msg += f"Accuracy[{acc.val:.3f}/{acc.avg:.3f}] "
                msg += f"ETA[{str((datetime.now() - epoch_start_time) / (batch_idx + 1) * (len(val_loader) - batch_idx - 1)).split('.')[0]}]"
                logger.info(msg)

                prefix = "{}_{}".format(os.path.join(output_dir, "val"), batch_idx)
                save_debug_images(config, input, meta, heatmap, pred * 4, output_heatmap, prefix)
        
        perf_indicator = 0
        if config.gpu == 0:
            name_values, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path, bbox_ids,
            )

            if wdb:
                wdb.log({"performance (AP)": perf_indicator})
                wdb.log({"loss_valid": losses.avg})
                wdb.log({"acc_valid": acc.avg})

            
            _, full_arch_name = get_model_name(config)
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, full_arch_name)
            else:
                _print_name_value(name_values, full_arch_name)
            logger.info(f"This epoch takes {datetime.now() - epoch_start_time}")
    return perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info("| Arch " + " ".join(["| {}".format(name) for name in names]) + " |")
    logger.info("|---" * (num_values + 1) + "|")
    logger.info(
        "| "
        + full_arch_name
        + "\n"
        + " "
        + " ".join(["| {:.3f}".format(value) for value in values])
        + " |"
    )


def cal_loss(
    config,
    criterion: dict,
    pred_heatmap,
    gt_heatmap,
    hm_weight,
    pred_key=None,
    target_keys=None,
    target_keypoints_weight=None,
    uncertainty=None,
    output_kd=None,
    teacher_output=None,
    count=None,
    wdb=None,
):
    loss = 0
    for k, v in criterion.items():
        if k == "heatmap":
            l = v(pred_heatmap, gt_heatmap, hm_weight, count, wdb)
            loss += l * config.LOSS.HM_LOSS_WEIGHT
        elif k == "keypoint":
            l = v(pred_key, target_keys / 4, target_keypoints_weight, uncertainty, count, wdb)
            loss += l * config.LOSS.KP_LOSS_WEIGHT
        elif k == "distillation":
            l = v(output_kd, teacher_output, hm_weight)
            loss += l * config.LOSS.KD_LOSS_WEIGHT
    return loss


def clip_grads(params):
    grad_clip = {"max_norm": 0.003, "norm_type": 2}
    params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return clip_grad.clip_grad_norm_(params, **grad_clip)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def to_wandb_img(wdb, data, clamp=False):
    data = data[:8]
    if clamp:
        data = data.clamp(min=0, max=1)
    return [wdb.Image(img) for img in np.split(data.detach().cpu().numpy(), data.size(0))]
