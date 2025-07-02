# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import os
import warnings
import argparse
import torch
from configs.hpe.config import update_config
from hpe.utils.logging import ShellColors


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