import sys
import os
import numpy as np
from pathlib import Path
import importlib

# 현재 파일의 위치에서 프로젝트 루트까지의 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # experiments -> raf -> project_root

# raf를 top-level package로 사용하기 위해 project_root만 추가
sys.path.insert(0, str(project_root))

import torch
from typing import Any

from hpe.utils.logging_utils import ShellColors as sc
from configs.hpe.config import config
from configs.hpe.config import get_model_name
from hpe.utils.logging_utils import create_logger_sfl
from hpe.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from hpe.utils.random_utils import init_random_seed, set_random_seed
from hpe.utils.misc_utils import show_info, parse_args
from hpe.utils.resolution_utils import setup_single_client_resolution, is_multi_resolution
from federated.server import FedServer
from hpe.train.client import FLClient
from hpe.models.vit import ViT
from hpe.models.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead
from hpe.models.vit_pose import ViTPose

# Type hint for config to allow dynamic attribute access
config: Any = config

def main(args):
    wdb = None
    if args.wandb:
        import wandb
        from datetime import datetime
        
        now = datetime.now()
        today = now.strftime("%m%d_%H:%M")
        
        name = f"{args.gnc_res[0]}-kd_use-{args.kd_use}_{args.gnc_split_num*1000}_bs{args.gnc_train_bs}_alpha={args.kd_alpha}-loss_sacle-{args.loss_scale}"
        name += f"_{today}"
        wdb = wandb
        wdb.init(
            config=config,
            project="CL Train",
            name = name,
        )
    
    config.KD_USE = True if args.kd_use else False
    config.KD_ALPHA = args.kd_alpha
    
    # Setup resolution configuration using utility function
    config.MODEL.IMAGE_SIZE, config.MODEL.HEATMAP_SIZE = setup_single_client_resolution(
        args.gnc_res[0], args.kd_use
    )
    print(f"image size: {config.MODEL.IMAGE_SIZE}")
    print(f"heatmap size: {config.MODEL.HEATMAP_SIZE}")
    
    
    show_info(0, args, config)
    
    print("------------- config image size ---------------------")
    print(config.MODEL.IMAGE_SIZE)
    print("------------- config heatmap size ---------------------")
    print(config.MODEL.HEATMAP_SIZE)

    # 동적 import 수행 (config 파일명에서 모듈명 추출)
    config_module_name = None
    if "small" in args.cfg:
        config_module_name = "configs.hpe.extra.vit_small_config"
    elif "large" in args.cfg:
        config_module_name = "configs.hpe.extra.vit_large_config"
    elif "huge" in args.cfg:
        config_module_name = "configs.hpe.extra.vit_huge_config"
    
    if config_module_name is None:
        raise FileNotFoundError(f"Check config file name: {args.cfg}")
    
    # 동적 import 수행
    try:
        config_module = importlib.import_module(config_module_name)
        extra = config_module.extra
    except ImportError as e:
        raise ImportError(f"Failed to import {config_module_name}: {e}")
    
    # Type hint for extra to allow dynamic attribute access
    extra: Any = extra

    res_arg = ""
    for res in args.gnc_res:
        res_arg += f"{res}_"
    res_arg += f"{args.kd_alpha}"
    res_arg += "_all_kd" if args.kd_use else "_no_kd"
    print(res_arg)
    logger, final_output_dir = create_logger_sfl(config, args.cfg, f"train_{args.gpu}", arg=res_arg)
    
    seed = init_random_seed(args.seed)
    logger.info(f"Set random seed to {seed}")
    set_random_seed(seed)
    
    pretrained_path = project_root / args.pretrained
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    perf_indicator = 0.0

    if config.MODEL.FREEZE_NAME:
        print("Freeze Group : ", config.MODEL.FREEZE_NAME)
    if config.MODEL.DIFF_NAME:
        print("Diff Group : ", config.MODEL.DIFF_NAME)
    
    # For Federated Learning -------------------
    
    ## Backbone - ViT
    backbone = ViT(
        img_size=extra["backbone"]["img_size"],
        patch_size=extra["backbone"]["patch_size"],
        embed_dim=extra["backbone"]["embed_dim"],
        in_channels=3,
        num_heads=extra["backbone"]["num_heads"],
        depth=extra["backbone"]["depth"],
        qkv_bias=True,
        drop_path_rate=extra["backbone"]["drop_path_rate"],
        use_gpe=config.MODEL.USE_GPE,
        use_lpe=config.MODEL.USE_LPE,
        use_gap=config.MODEL.USE_GAP,
    )
 
    ## HEAD - Heatmap Simple Head
    deconv_head = TopdownHeatmapSimpleHead(
        in_channels=extra["keypoint_head"]["in_channels"],
        num_deconv_layers=extra["keypoint_head"]["num_deconv_layers"],
        num_deconv_filters=extra["keypoint_head"]["num_deconv_filters"],
        num_deconv_kernels=extra["keypoint_head"]["num_deconv_kernels"],
        extra=dict(final_conv_kernel=1),
        out_channels=config.MODEL.NUM_JOINTS,
    )
    
    global_fl_model = ViTPose(backbone, deconv_head)
    
    # fl client model weight initialization
    if "mae" in str(pretrained_path):
        load_checkpoint(global_fl_model, pretrained_path)
    
    # global fl model -> gpu
    global_fl_model.to(device)
    
    cl_client = FLClient(
        client_id=0, # dataset의 index
        config=config,
        device=device,
        init_model=global_fl_model,  # 이전과 동일하게 init_model 전달
        extra=extra,
        wdb=wdb,
        logger=logger,
        im_size=config.MODEL.IMAGE_SIZE,
        hm_size=config.MODEL.HEATMAP_SIZE,
        batch_size=args.gnc_train_bs,
        is_proxy=False,
        samples_per_split=args.gnc_split_num,
    )
    
    perf_buf = [0.0]
    
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        init_time = datetime.now()
        
        # scheduler step
        cl_client.lr_scheduler.step()

        # -------- Train --------------------------------------------------------
        if is_multi_resolution(config.MODEL.IMAGE_SIZE):
            print(f"\n>>> [{args.gnc_res[0]}] Client Multi-res (KD) Centralized Training")
            cl_client.train_multi_resolution(epoch)
        else:
            print(f"\n>>> [{args.gnc_res[0]}] Client Single Centralized Training")
            cl_client.train_single_resolution(epoch)
        
        epoch_e_time = datetime.now() - init_time
        logger.info(f"This epoch takes {epoch_e_time}\n")
        # ----------------------------------------------------------------------
        lr_ = cl_client.lr_scheduler.get_lr()
        for i, g in enumerate(cl_client.optimizer.param_groups):
            g["lr"] = lr_[i]

        # -------- Test --------------------------------------------------------
        # evaluate on validation set
        if epoch % config.EVALUATION.INTERVAL == 0:
            global_model_state_file = os.path.join(
                final_output_dir,
                f"{config.MODEL.NAME}_{config.MODEL.TYPE}_global_global_{config.LOSS.HM_LOSS}_{epoch}.pt",
            )
            logger.info(f"saving final global model state to {global_model_state_file}")
            torch.save(cl_client.model.state_dict(), global_model_state_file)
            
            print(f"{sc.COLOR_LIGHT_PURPLE}---------------------------------------------------------------------------------------------{sc.ENDC}")
            print(f"--------------- Client {sc.COLOR_LIGHT_PURPLE}Evaluating{sc.ENDC} ---------------------------------------------")
            print(f"{sc.COLOR_LIGHT_PURPLE}---------------------------------------------------------------------------------------------{sc.ENDC}")
            
            # evaluate performance of each clients
            perf_indicator = cl_client.evaluate(
                final_output_dir=final_output_dir,
                wdb=wdb,
            )
            
            # avg perf가 가장 높은 성능이 나왔을 때만 model save
            if perf_indicator > max(perf_buf):
                logger.info(f"Epoch [{epoch}] best avg performance detected: {max(perf_buf):.4f} => {perf_indicator:.4f}")
                perf_buf.append(round(float(perf_indicator), 4))
                logger.info(f"Current Best Performances: {perf_buf}")
                
                # logging
                logger.info(f"=> saving best client checkpoint to {final_output_dir}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_client": get_model_name(config),
                        "state_dict": cl_client.model.state_dict(),
                        "perf": perf_indicator,
                        "optimizer": cl_client.optimizer.state_dict(),
                        "HM_LOSS": config.LOSS.HM_LOSS,
                    },
                    final_output_dir,
                )
            print(f"{sc.COLOR_LIGHT_PURPLE}---------------------------------------------------------------------------------------------{sc.ENDC}")
            
    # saving model state file    
    final_global_model_state_file = os.path.join(final_output_dir, "final_state_global.pt")
    logger.info(f"saving final client model state to {final_global_model_state_file}")
    torch.save(cl_client.model.state_dict(), final_global_model_state_file)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
