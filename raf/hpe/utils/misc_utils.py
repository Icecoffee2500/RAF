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
    print(f"{ShellColors.COLOR_GREEN}SEED: {ShellColors.ENDC}{args.seed}")
    print(f"{ShellColors.COLOR_GREEN}TRAIN BATCH SIZE: {ShellColors.ENDC}{args.train_bs} (per client)")
    print(f"{ShellColors.COLOR_GREEN}TEST BATCH SIZE: {ShellColors.ENDC}{config.TEST.BATCH_SIZE}")
    print(f"{ShellColors.COLOR_CYAN}USE AMP: {ShellColors.ENDC}{config.MODEL.USE_AMP}")
    print(f"{ShellColors.COLOR_CYAN}USE UDP: {ShellColors.ENDC}{config.TEST.USE_UDP}")
    print(f"{ShellColors.COLOR_CYAN}USE FLIP: {ShellColors.ENDC}{config.TEST.FLIP_TEST}")
    print(f"{ShellColors.COLOR_CYAN}USE GT BBOX: {ShellColors.ENDC}{config.TEST.USE_GT_BBOX}")
    print(f"{ShellColors.COLOR_CYAN}USE WANDB: {ShellColors.ENDC}{args.wandb}")
    print(f"{ShellColors.COLOR_CYAN}AGGREGATE METHOD: {ShellColors.ENDC}{args.fed}")
    # print(f"{ShellColors.COLOR_CYAN}FEDAVG: {ShellColors.ENDC}{args.fed == 'fedavg'}")
    # print(f"{ShellColors.COLOR_CYAN}FEDPROX: {ShellColors.ENDC}{args.fed == 'fedprox'}")
    print(f"{ShellColors.COLOR_CYAN}USE KD_USE: {ShellColors.ENDC}{args.kd_use}")
    print(f"{ShellColors.COLOR_CYAN}NUMBER OF CLIENT SPLITS: {ShellColors.ENDC}{args.samples_per_client}")
    print(f"{ShellColors.COLOR_CYAN}KD ALPHA: {ShellColors.ENDC}{args.kd_alpha}")
    print(f"{ShellColors.COLOR_CYAN}CLIENT NUM: {ShellColors.ENDC}{args.client_num}")
    print(f"{ShellColors.COLOR_CYAN}LOSS SCALE: {ShellColors.ENDC}{args.loss_scale}")
    print(f"{ShellColors.COLOR_CYAN}CLIENT RESOLUTIONS: {ShellColors.ENDC}{args.client_res}")
    print(f"{'='*46}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="")
    parser.add_argument("--pretrained", help="checkpoint name", required=True, type=str)
    parser.add_argument("--wandb", help="use wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--fed", type=str, default="fedavg", help="aggregate method")
    parser.add_argument("--kd_use", action="store_true", help="knowledge distillation")
    parser.add_argument("--samples_per_client", type=int, default=1, help="number of split")
    parser.add_argument("--kd_alpha", type=float, default=1.0, help="alpha wight for proxy gt loss")
    parser.add_argument("--client_num", type=int, help="Client number")
    parser.add_argument("--train_bs", type=int, default=32, help="Train Batch Size")
    parser.add_argument("--loss_scale", type=int, default=1, help="loss scale")
    parser.add_argument("--dyn_alpha", type=float, default=1e-6, help="Dynamic Regularization alpha Coefficient")
    parser.add_argument("--mu_con", type=float, default=1.0, help="MOON - mu")
    parser.add_argument(
        "--client_res",
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
    parser.add_argument("--num_workers", type=int, default=4, help="")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    args = parser.parse_args()

    return args 