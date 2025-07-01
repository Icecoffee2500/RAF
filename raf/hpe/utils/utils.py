# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import re
import copy
import logging
import time
import warnings
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from torch import distributed as dist
from distutils.version import LooseVersion

from pathlib import Path
from collections import OrderedDict
from typing import Union

import argparse
from configs.hpe.config import update_config
from hpe.federated.loss_fns import JointsMSELoss, JointsKLDLoss, DistillationLoss

TORCH_VERSION = torch.__version__

class ShellColors:
    COLOR_NC = "\033[0m"  # No Color
    COLOR_BLACK = "\033[0;30m"
    COLOR_GRAY = "\033[1;30m"
    COLOR_RED = "\033[0;31m"
    COLOR_LIGHT_RED = "\033[1;31m"
    COLOR_GREEN = "\033[0;32m"
    COLOR_LIGHT_GREEN = "\033[1;32m"
    COLOR_BROWN = "\033[0;33m"
    COLOR_YELLOW = "\033[1;33m"
    COLOR_BLUE = "\033[0;34m"
    COLOR_LIGHT_BLUE = "\033[1;34m"
    COLOR_PURPLE = "\033[0;35m"
    COLOR_LIGHT_PURPLE = "\033[1;35m"
    COLOR_CYAN = "\033[0;36m"
    COLOR_LIGHT_CYAN = "\033[1;36m"
    COLOR_LIGHT_GRAY = "\033[0;37m"
    COLOR_WHITE = "\033[1;37m"

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    MAGENTA = "\033[95m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def create_logger(cfg, cfg_name, phase="train"):
    root_output_dir = Path(cfg.OUTPUT_DIR) # output

    # set up logger
    if not root_output_dir.exists():
        print(f"=> creating {root_output_dir}")
        root_output_dir.mkdir()
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    dataset = (
        cfg.DATASET.DATASET + "_" + cfg.DATASET.HYBRID_JOINTS_TYPE
        if cfg.DATASET.HYBRID_JOINTS_TYPE
        else cfg.DATASET.DATASET
    )
    dataset = dataset.replace(":", "_")
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split(".")[0] # config file (yaml) 이름
    cfg_name = cfg_name + "_" + time_str

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print("=> creating {}".format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = f"{cfg_name}_{time_str}_{phase}.log"
    final_log_file = final_output_dir / log_file
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger, str(final_output_dir)

def create_logger_sfl(cfg, cfg_name, phase="train", arg=None):
    root_output_dir = Path(cfg.OUTPUT_DIR)

    # set up logger
    if not root_output_dir.exists():
        print(f"=> creating {root_output_dir}")
        root_output_dir.mkdir()
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    dataset = f"{cfg.DATASET_SETS[0].DATASET}_{cfg.DATASET_SETS[1].DATASET}_{cfg.DATASET_SETS[2].DATASET}"
    dataset = dataset.replace(":", "_")
    model = f"{cfg.MODEL.NAME}-{cfg.MODEL.TYPE}"
    cfg_name = os.path.basename(cfg_name).split(".")[0]
    cfg_name = f"{cfg_name}_{time_str}"
    if arg is not None:
        cfg_name += f"_{arg}"

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print(f"=> creating {final_output_dir}")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = f"{cfg_name}_{time_str}_{phase}.log"
    final_log_file = final_output_dir / log_file
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    return logger, str(final_output_dir)


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


def save_checkpoint(states, output_dir, filename="checkpoint.pth.tar"):
    torch.save(states, os.path.join(output_dir, filename)) # resume를 위해서 모든 설정 저장.
    
    # best model state dict 저장
    if "client_state_dict" in states and "server_state_dict" in states:
        # torch.save(states["client_state_dict"], os.path.join(output_dir, f"model_client_{states['client_idx']}_best.pt"))
        torch.save(states["client_state_dict"], os.path.join(output_dir, f"model_client_best.pt"))
        torch.save(states["server_state_dict"], os.path.join(output_dir, "model_server_best.pt"))
    elif "state_dict" in states:
        torch.save(states["state_dict"], os.path.join(output_dir, "model_best.pth"))


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str or pathlib.Path): target directory

    Returns:
    A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def resize(
    input,
    size=None,
    align_corners=None,
    warning=True,
):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)


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


def build_norm_layer(type, feature_num):
    if type['type'] == 'BN':
        return nn.BatchNorm2d(feature_num)
    elif type['type'] == 'SyncBN':
        return nn.SyncBatchNorm(feature_num)
    elif type['type'] == 'LN':
        return nn.LayerNorm(feature_num, eps=type['eps'])
    
    
def custom_init_weights(model, checkpoint_path):
    print("checkpoint path : ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    try:
        c_keys = list(checkpoint["state_dict"].keys())
        c_sd = checkpoint["state_dict"]
    except:
        c_sd = checkpoint
        c_keys = list(checkpoint.keys())

    m_sd = model.state_dict()
    m_keys = list(model.state_dict().keys())

    for i in range(len(m_keys)):
        try:
            if c_sd[c_keys[i]].shape != m_sd[m_keys[i]].shape:
                print("Please verify once again!! >>", end=" ")
                print(c_keys[i], m_keys[i])
            if c_sd[c_keys[i]].shape == m_sd[m_keys[i]].shape:
                m_sd[m_keys[i]] = c_sd[c_keys[i]]
        except IndexError:
            print("index is over :", m_keys[i])

    print("Succesfully init weights..")
    model.load_state_dict(m_sd, strict=False)
    return model


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

def get_dist_info():
    if LooseVersion(TORCH_VERSION) < LooseVersion('1.0'):
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def set_random_seed(seed):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _load_checkpoint(filename, map_location=None):

    if not osp.isfile(filename):
        raise IOError(f"{filename} is not a checkpoint file")
    # checkpoint = torch.load(filename, map_location=map_location)
    checkpoint = torch.load(filename, map_location=map_location, weights_only=True)

    return checkpoint

def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=""):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if "num_batches_tracked" not in key]

    if unexpected_keys:
        err_msg.append(
            "unexpected key in source " f'state_dict: {", ".join(unexpected_keys)}\n'
        )
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n'
        )

def load_checkpoint(
    model,
    filename,
    map_location="cpu",
    strict=False,
    logger=None,
    patch_padding="pad",
    part_features=None,
):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        patch_padding (str): 'pad' or 'bilinear' or 'bicubic', used for interpolate patch embed from 14x14 to 16x16

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # for MoBY, load model of online branch
    # encoder로 시작하는 파라미터들은 앞에 encoder 떼어버림
    if sorted(list(state_dict.keys()))[0].startswith("encoder"):
        state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }

    if ("patch_embed.proj.weight" in state_dict) and ("patch_embed.proj.weight" in model.state_dict()):
        proj_weight = state_dict["patch_embed.proj.weight"]
        orig_size = proj_weight.shape[2:]
        current_size = model.patch_embed.proj.weight.shape[2:]
        if orig_size != current_size:
            padding_size = current_size[0] - orig_size[0]
            padding_l = padding_size // 2
            padding_r = padding_size - padding_l
            if "pad" in patch_padding:
                proj_weight = torch.nn.functional.pad(
                    proj_weight, (padding_l, padding_r, padding_l, padding_r)
                )
            elif "bilinear" in patch_padding:
                proj_weight = torch.nn.functional.interpolate(
                    proj_weight, size=current_size, mode="bilinear", align_corners=False
                )
            elif "bicubic" in patch_padding:
                proj_weight = torch.nn.functional.interpolate(
                    proj_weight, size=current_size, mode="bicubic", align_corners=False
                )
            state_dict["patch_embed.proj.weight"] = proj_weight

    if ("pos_embed" in state_dict) and ("pos_embed" in model.state_dict()):
        pos_embed_checkpoint = state_dict["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        H, W = model.patch_embed.patch_shape
        n_patches = model.patch_embed.n_patches
        num_extra_tokens = model.pos_embed.shape[-2] - n_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size
        ).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(H, W), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dict["pos_embed"] = new_pos_embed

    new_state_dict = copy.deepcopy(state_dict)
    
    # MoE를 위한 수정 (expert 빼고 공통 fc만 가져옴 (아마도?))
    if part_features is not None:
        current_keys = list(model.state_dict().keys())
        for key in current_keys:
            if "mlp.experts" in key:
                source_key = re.sub(r"experts.\d+.", "fc2.", key)
                new_state_dict[key] = state_dict[source_key][-part_features:]
            elif "fc2" in key:
                new_state_dict[key] = state_dict[key][:-part_features]

    # load state_dict
    load_state_dict(model, new_state_dict, strict, logger)
    
    print("mae success!!")
    return checkpoint

def show_info(gpu, args, config):
    print(f"{'='*20} Info {'='*20}")
    print(f"{ShellColors.COLOR_GREEN}CURRENT GPU: {ShellColors.ENDC}{gpu}")
    print(f"{ShellColors.COLOR_GREEN}CONFIG: {ShellColors.ENDC}{args.cfg}")
    print(f"{ShellColors.COLOR_GREEN}PRETRAINED: {ShellColors.ENDC}{args.pretrained}")
    print(f"{ShellColors.COLOR_GREEN}CKPT CLIENT: {ShellColors.ENDC}{args.ckpt_client}")
    print(f"{ShellColors.COLOR_GREEN}CKPT SERVER: {ShellColors.ENDC}{args.ckpt_server}")
    print(f"{ShellColors.COLOR_GREEN}SEED: {ShellColors.ENDC}{args.seed}")
    # print(f"{ShellColors.COLOR_GREEN}TRAIN BATCH SIZE: {ShellColors.ENDC}{config.TRAIN.BATCH_SIZE}")
    print(f"{ShellColors.COLOR_GREEN}GENRERAL CLIENTS TRAIN BATCH SIZE: {ShellColors.ENDC}{args.gnc_train_bs}")
    print(f"{ShellColors.COLOR_GREEN}PROXY CLIENTS TRAIN BATCH SIZE: {ShellColors.ENDC}{args.prc_train_bs}")
    print(f"{ShellColors.COLOR_GREEN}TEST BATCH SIZE: {ShellColors.ENDC}{config.TEST.BATCH_SIZE}")
    print(f"{ShellColors.COLOR_CYAN}USE AMP: {ShellColors.ENDC}{config.MODEL.USE_AMP}")
    print(f"{ShellColors.COLOR_CYAN}USE UDP: {ShellColors.ENDC}{config.TEST.USE_UDP}")
    print(f"{ShellColors.COLOR_CYAN}USE FLIP: {ShellColors.ENDC}{config.TEST.FLIP_TEST}")
    print(f"{ShellColors.COLOR_CYAN}USE GT BBOX: {ShellColors.ENDC}{config.TEST.USE_GT_BBOX}")
    # Uncertainty 기능 제거됨
    print(f"{ShellColors.COLOR_CYAN}USE WARMUP: {ShellColors.ENDC}{args.warmup}")
    print(f"{ShellColors.COLOR_CYAN}USE WANDB: {ShellColors.ENDC}{args.wandb}")
    print(f"{ShellColors.COLOR_CYAN}Augmentation: {ShellColors.ENDC}{args.data_aug}")
    print(f"{ShellColors.COLOR_CYAN}USE SAME_POS: {ShellColors.ENDC}{args.same_pos}")
    print(f"{ShellColors.COLOR_CYAN}USE CLEAN_HIGH: {ShellColors.ENDC}{args.clean_high}")
    print(f"{ShellColors.COLOR_CYAN}AGGREGATE METHOD: {ShellColors.ENDC}{args.fed}")
    print(f"{ShellColors.COLOR_CYAN}FEDAVG: {ShellColors.ENDC}{args.fed == 'fedavg'}")
    print(f"{ShellColors.COLOR_CYAN}FEDPROX: {ShellColors.ENDC}{args.fed == 'fedprox'}")
    print(f"{ShellColors.COLOR_CYAN}USE KD_USE: {ShellColors.ENDC}{args.kd_use}")
    print(f"{ShellColors.COLOR_CYAN}NUMBER OF GENERAL CLIENT SPLITS: {ShellColors.ENDC}{args.gnc_split_num}")
    print(f"{ShellColors.COLOR_CYAN}NUMBER OF PROXY CLIENT SPLITS: {ShellColors.ENDC}{args.prc_split_num}")
    print(f"{ShellColors.COLOR_CYAN}CASCADED ARCHITECTURE: {ShellColors.ENDC}{args.cascade}")
    print(f"{ShellColors.COLOR_CYAN}KD ALPHA: {ShellColors.ENDC}{args.kd_alpha}")
    print(f"{ShellColors.COLOR_CYAN}PRC USE: {ShellColors.ENDC}{args.prc_use}")
    # print(f"{ShellColors.COLOR_CYAN}NUM USER: {ShellColors.ENDC}{args.num_users}")
    print(f"{ShellColors.COLOR_CYAN}GENERAL CLIENT NUM: {ShellColors.ENDC}{args.gnc_num}")
    print(f"{ShellColors.COLOR_CYAN}PROXY CLIENT NUM: {ShellColors.ENDC}{args.prc_num}")
    print(f"{ShellColors.COLOR_CYAN}LOSS SCALE: {ShellColors.ENDC}{args.loss_scale}")
    print(f"{ShellColors.COLOR_CYAN}GENERAL CLIENT RESOLUTIONS: {ShellColors.ENDC}{args.gnc_res}")
    print(f"{'='*46}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="")
    parser.add_argument("--pretrained", help="checkpoint name", required=True, type=str)
    parser.add_argument("--ckpt_client", default=None, help="checkpoint name", type=str)
    parser.add_argument("--ckpt_server", default=None, help="checkpoint name", type=str)
    parser.add_argument("--wandb", help="use wandb", action="store_true")
    parser.add_argument("--warmup", help="use warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--data_aug", default="", help="data augmentation", type=str)
    parser.add_argument("--same_pos", action="store_true", help="same_pos")
    parser.add_argument("--clean_high", action="store_true", help="clean_high")
    parser.add_argument("--fed", type=str, default="fedavg", help="aggregate method")
    parser.add_argument("--kd_use", action="store_true", help="knowledge distillation")
    parser.add_argument("--gnc_split_num", type=int, default=1, help="number of split")
    parser.add_argument("--prc_split_num", type=int, default=1, help="number of split")
    parser.add_argument("--cascade", action="store_true", default=False, help="Cascaded Aggregation")
    parser.add_argument("--kd_alpha", type=float, default=1.0, help="alpha wight for proxy gt loss")
    parser.add_argument("--prc_use", action="store_true", default=False, help="Is there proxy-client?")
    parser.add_argument("--prc_num", type=int, help="Proxy Clients number")
    parser.add_argument("--gnc_num", type=int, help="General Clients number")
    parser.add_argument("--num_users", type=int, default=3, help="number of users")
    parser.add_argument("--gnc_train_bs", type=int, default=32, help="General Clients Train Batch Size")
    parser.add_argument("--prc_train_bs", type=int, default=32, help="Proxy Clients Train Batch Size")
    parser.add_argument("--loss_scale", type=int, default=1, help="loss scale")
    parser.add_argument(
        "--gnc_res",
        type=str,
        default="",
        nargs='+',
        choices=['max_high', 'sup_high', 'high', 'mid', 'low', 'sup_low'],
        help="Resolution of clients (high / mid / low)"
    )
    parser.add_argument(
        "--test_res",
        type=int,
        nargs='+',
        help="Test Resolution - (256 192) (192 144)"
    )
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)
    parser.add_argument("--lr", default=0.1, help="")
    parser.add_argument("--resume", default=None, help="")
    parser.add_argument("--batch_size", type=int, default=768, help="")
    parser.add_argument("--num_workers", type=int, default=4, help="")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    args = parser.parse_args()

    return args