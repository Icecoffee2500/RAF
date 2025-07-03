# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

import os
import logging
import time
from pathlib import Path


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


def create_logger_sfl(cfg, cfg_name, phase="train", arg=None):
    root_output_dir = Path(cfg.OUTPUT_DIR)

    # set up logger
    if not root_output_dir.exists():
        print(f"=> creating {root_output_dir}")
        root_output_dir.mkdir()
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    # dataset = f"{cfg.DATASET_SETS[0].DATASET}_{cfg.DATASET_SETS[1].DATASET}_{cfg.DATASET_SETS[2].DATASET}"
    dataset = cfg.DATASET.DATASET
    # dataset = dataset.replace(":", "_")
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