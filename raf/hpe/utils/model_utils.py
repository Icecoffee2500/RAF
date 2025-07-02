# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import torch
import torch.optim as optim
import torch.nn as nn

from hpe.federated.loss_fns import JointsMSELoss
from hpe.utils.logging import ShellColors


def get_loss(cfg, device=None):
    loss = {}

    # Heatmap Loss
    if cfg.LOSS.HM_LOSS == "JointMSEloss":
        print(f"{ShellColors.COLOR_GREEN}Use JointMSE Loss{ShellColors.ENDC}")
        loss["heatmap"] = JointsMSELoss(use_target_weight=True)
    else:
        raise NotImplementedError("You have to check your HM loss name !!")

    # Uncertainty Loss 기능 제거됨 (사용하지 않음)
    
    #TODO: Distillation Loss
    if cfg.LOSS.KD_LOSS == "DistillationLoss":
        print(f"{ShellColors.COLOR_GREEN}Use Distillation Loss{ShellColors.ENDC}")
        # loss["distillation"] = DistillationLoss(distillation_type=cfg.LOSS.KD_TYPE, tau=cfg.LOSS.KD_TAU)
    elif cfg.LOSS.KD_LOSS == "JointMSEloss":
        print(f"{ShellColors.COLOR_GREEN}Use Distillation Loss - HM MSE{ShellColors.ENDC}")
        loss["distillation"] = JointsMSELoss(use_target_weight=True)
    else:
        print(f"{ShellColors.WARNING}Do not use Distillation Loss{ShellColors.ENDC}")
    
    if device:
        for k, v in loss.items():
            loss[k] = loss[k].to(device=device)

    return loss


def get_vit_optimizer(cfg, model, extra):
    # large_group_list 구성하기.
    # large_group_list = [['backbone.pos_embed',
    #                     'backbone.patch_embed.proj.weight',
    #                     'backbone.patch_embed.proj.bias',
    #                     'backbone.global_pos_embed.embed_layer.weight',
    #                     'backbone.global_pos_embed.embed_layer.bias'],
    #                     ]
    large_group_list = [['backbone.pos_embed',
                        'backbone.patch_embed.proj.weight',
                        'backbone.patch_embed.proj.bias',],
                        ]
    for i in range(extra['backbone']['depth']):
        large_group_list.append([name for name, param in model.named_parameters() if f'blocks.{i}.' in name])

    large_group_list.append([name for name, param in model.named_parameters() if 'keypoint_head' in name])

    # no_decay_group 관련 parameters
    no_decay_name = ['pos_embed', 'norm', 'bias']

    # lr_each_group 구성하기. (얘는 무조건 vit의 크기에 따라 달라짐.)
    lr_each_group = [cfg.TRAIN.LR * extra['backbone']['lr_decay_rate']**i for i in range(extra['backbone']['depth']+2)]
    lr_each_group.insert(0,cfg.TRAIN.LR)
    lr_each_group = lr_each_group[::-1]
    
    # param_group을 정의 - decay / no decay를 나누기 위함.
    param_group = []
    for i, group in enumerate(large_group_list):
        no_decay_group = []
        decay_group = []
        for name in group:
            if any([1 for n in no_decay_name if n in name]):
                no_decay_group.append(name)
            else:
                decay_group.append(name)
        if no_decay_group != []:
            param_group.append({'params': [param for name, param in model.named_parameters() if name in no_decay_group], 
                                'lr': lr_each_group[i], 
                                'weight_decay': 0.0, 
                                'param_names': no_decay_group, 
                                'lr_scale': extra['backbone']['lr_decay_rate']**(len(large_group_list)-i), 
                })
        if decay_group != []: 
            param_group.append({'params': [param for name, param in model.named_parameters() if name in decay_group], 
                                'lr': lr_each_group[i], 
                                'weight_decay': 0.1, 
                                'param_names': decay_group, 
                                'lr_scale': extra['backbone']['lr_decay_rate']**(len(large_group_list)-i), 
                })
            
    # freeze를 위한 ...
    freeze_list = []
    if cfg.MODEL.FREEZE_NAME or cfg.MODEL.DIFF_NAME:
        for name, param in model.named_parameters():
            result_freeze = [True if i in name else False for i in cfg.MODEL.FREEZE_NAME]
            if any(result_freeze):
                param.requires_grad = False
                freeze_list.append(name)
    print(f"freeze list => ")
    for name in freeze_list:
        print(name)
    
    optimizer = optim.AdamW(param_group, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    return optimizer


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias) 