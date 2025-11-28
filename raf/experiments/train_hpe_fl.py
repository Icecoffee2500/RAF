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

from hpe.utils.logging import ShellColors as sc
from configs.hpe.config import config
from configs.hpe.config import get_model_name

# Type hint for config to allow dynamic attribute access
config: Any = config
from hpe.utils.logging import create_logger_sfl
from hpe.utils.checkpoint_utils import save_checkpoint, load_checkpoint, save_checkpoint_fedbn
from hpe.utils.random_utils import init_random_seed, set_random_seed
from hpe.utils.misc_utils import show_info, parse_args
from hpe.utils.resolution_utils import setup_client_resolutions, is_multi_resolution
from federated.server import FedServer
from hpe.federated.client import FLClient
from hpe.models import ViT
from hpe.models import TopdownHeatmapSimpleHead
from hpe.models import ViTPose

def main(args):
    wdb = None
    if args.wandb:
        try:
            import wandb
            from datetime import datetime
            
            now = datetime.now()
            today = now.strftime("%m%d_%H:%M")
            
            name = f"aggr-{args.fed}_loss_sacle-{args.loss_scale}_G{args.client_num}_{args.samples_per_client}_bs{args.train_bs}_alpha={args.kd_alpha}"
            name += "_res"
            for res in args.client_res:
                name += f"_{res}"
            name += f"_{today}"
            
            # wandb 초기화 시도
            wdb = wandb
            
            # config를 dict로 변환하여 wandb에 안전하게 전달
            try:
                # EasyDict를 일반 dict로 변환
                wandb_config = dict(config) if hasattr(config, '__dict__') else {}
            except:
                # 변환 실패 시 주요 설정만 수동으로 전달
                wandb_config = {
                    "model_name": config.MODEL.NAME if hasattr(config, 'MODEL') else "unknown",
                    "batch_size": args.train_bs,
                    "fed_method": args.fed,
                    "kd_alpha": args.kd_alpha,
                    "loss_scale": args.loss_scale,
                }
            
            wdb.init(
                project="RAF-refactoring",
                name=name,
                config=wandb_config,
                mode="online",  # "offline"로 변경하면 로컬에서만 로깅
            )
            print(f"{sc.COLOR_GREEN}WandB 초기화 성공{sc.ENDC}")
            
        except Exception as e:
            print(f"{sc.COLOR_RED}WandB 초기화 실패: {e}{sc.ENDC}")
            print(f"{sc.COLOR_YELLOW}WandB 없이 실행을 계속합니다{sc.ENDC}")
            wdb = None
    
    # update config
    config.FED.FEDAVG = True if args.fed == "fedavg" else False
    config.FED.FEDPROX = True if args.fed == "fedprox" else False
    config.KD_USE = True if args.kd_use else False
    config.KD_ALPHA = args.kd_alpha
    config.LOSS_SCALE = args.loss_scale
    
    if args.client_res:
        config.MODEL.IMAGE_SIZE, config.MODEL.HEATMAP_SIZE = setup_client_resolutions(args.client_res, args.kd_use)
        print(f"image size: {config.MODEL.IMAGE_SIZE}")
        print(f"heatmap size: {config.MODEL.HEATMAP_SIZE}")
    
    
    show_info(0, args, config)
    
    print("------------- config image size ---------------------")
    print(config.MODEL.IMAGE_SIZE)
    print("------------- config heatmap size ---------------------")
    print(config.MODEL.HEATMAP_SIZE)

    # 동적 import를 위한 config 파일 매핑 (uncertainty config 제거됨)
    config_mapping = {
        "small": "configs.hpe.extra.vit_small_config",
        "large": "configs.hpe.extra.vit_large_config", 
        "huge": "configs.hpe.extra.vit_huge_config"
    }
    
    # args.cfg에서 적절한 config 모듈 찾기
    config_module_name = None
    for key in config_mapping:
        if key in args.cfg:
            config_module_name = config_mapping[key]
            break
    
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

    # res_arg = f"pr-split{args.prc_split_num}_pr-bs{args.prc_train_bs}"
    res_arg = ""
    for res in args.client_res:
        res_arg += f"{res}_"
    res_arg += f"{args.kd_alpha}"
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

    # keys = sorted(global_fl_model.state_dict().keys())
    # prev = []
    # for key in keys:
    #     parts = key.split('.')
    #     for i, p in enumerate(parts):
    #         if i >= len(prev) or prev[i] != p:
    #             print("  " * i + f"- {p}")
    #     prev = parts
    # return
    
    fl_clients = []
    for idx in range(args.client_num):
        fl_clients.append(
            FLClient(
                client_id=idx, # dataset의 index
                config=config,
                device=device,
                init_model=global_fl_model,  # 이전과 동일하게 init_model 전달
                extra=extra,
                wdb=wdb,
                logger=logger,
                im_size=config.MODEL.IMAGE_SIZE[idx],
                hm_size=config.MODEL.HEATMAP_SIZE[idx],
                batch_size=args.train_bs,
                is_proxy=False,
                samples_per_split=args.samples_per_client,
            )
        )
    
    # Fed Server for aggregating model weights
    fed_server = FedServer(args.fed)
    
    avg_perf_buf = [0.0]
    
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        init_time = datetime.now()
        
        # scheduler step
        for client in fl_clients:
            client.lr_scheduler.step()

        # Train ----------------------------------------------------------------
        client_weights = []
        for idx, client in enumerate(fl_clients):
            # print(f"\n>>> General Client [{idx}] Training")
            if is_multi_resolution(config.MODEL.IMAGE_SIZE[idx]):
                print(f"\n>>> General Client [{idx}]-[{args.client_res[idx]}] Multi-res (KD) Federated Learning Training")
                client.train_multi_resolution(epoch)
            else:
                print(f"\n>>> General Client [{idx}]-[{args.client_res[idx]}] Single-res (No-KD) Federated Learning")
                client.train_single_resolution(epoch)
            
            client_weights.append(client.model.state_dict())
        
        # aggregate weights
        logger.info(">>> load Fed-Averaged weight to the client model ...")
        w_glob_client = fed_server.aggregate(logger, client_weights)
        
        # Broadcast weight to each clients
        logger.info(">>> load Fed-Averaged weight to the each client model ...")
        if args.fed == "fedbn":
            print("FedBN Broadcasting ...")
            for client in fl_clients:
                for key in w_glob_client.keys():
                    if 'running' not in key:
                        client.model.state_dict()[key].data.copy_(w_glob_client[key])
                    # else:
                    #     print(f"ignore parameter: {key}")
        else:
            for client in fl_clients:
                client.model.load_state_dict(w_glob_client)
        
        # load weight to global fl model
        global_fl_model.load_state_dict(w_glob_client)
        
        # # Set global model to eval mode for consistent evaluation
        # global_fl_model.eval()
        
        epoch_e_time = datetime.now() - init_time
        logger.info(f"This epoch takes {epoch_e_time}\n")
        # ----------------------------------------------------------------------
        for client in fl_clients:
            lr_ = client.lr_scheduler.get_lr()
            for i, g in enumerate(client.optimizer.param_groups):
                g["lr"] = lr_[i]

        # -------- Test --------------------------------------------------------
        # evaluate on validation set
        if epoch % config.EVALUATION.INTERVAL == 0:
            global_model_state_file = os.path.join(
                final_output_dir,
                f"{config.MODEL.NAME}_{config.MODEL.TYPE}_global_global_{config.LOSS.HM_LOSS}_{epoch}.pt",
            )
            logger.info(f"saving final global model state to {global_model_state_file}")
            torch.save(global_fl_model.state_dict(), global_model_state_file)
            
            curr_avg_perf = 0 # this is average performance
            
            for client_idx, client in enumerate(fl_clients):
                print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
                print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_PURPLE}Evaluating{sc.ENDC} ------------")
                print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
                
                # evaluate performance of each clients (자체 모델 사용으로 개선)
                perf_indicator = client.evaluate(
                    final_output_dir=final_output_dir,
                    wdb=wdb,
                )
                # perf_indicator = client.evaluate(
                #     final_output_dir=final_output_dir,
                #     backbone=backbone,
                #     keypoint_head=deconv_head,
                #     wdb=wdb,
                # )
                curr_avg_perf = curr_avg_perf + perf_indicator / len(fl_clients) # this is average performance
            
            # avg perf가 가장 높은 성능이 나왔을 때만 model save
            if curr_avg_perf > max(avg_perf_buf):
                logger.info(f"Epoch [{epoch}] best avg performance detected: {max(avg_perf_buf):.4f} => {curr_avg_perf:.4f}")
                avg_perf_buf.append(round(float(curr_avg_perf), 4))
                logger.info(f"Current Best Average Performances: {avg_perf_buf}")
                
                # logging
                logger.info(f"=> saving best client checkpoint to {final_output_dir}")
                if args.fed == 'fedbn':
                    save_checkpoint_fedbn(
                        {
                            "epoch": epoch + 1,
                            "model_client": get_model_name(config),
                            "high_state_dict": fl_clients[0].model.state_dict(),
                            "mid_state_dict": fl_clients[1].model.state_dict(),
                            "low_state_dict": fl_clients[2].model.state_dict(),
                            "perf": curr_avg_perf,
                            "optimizer": fl_clients[0].optimizer.state_dict(),
                            "HM_LOSS": config.LOSS.HM_LOSS,
                        },
                        final_output_dir,
                    )
                else:
                    save_checkpoint(
                        {
                            "epoch": epoch + 1,
                            "model_client": get_model_name(config),
                            "state_dict": global_fl_model.state_dict(),
                            "perf": curr_avg_perf,
                            "optimizer": fl_clients[0].optimizer.state_dict(),
                            "HM_LOSS": config.LOSS.HM_LOSS,
                        },
                        final_output_dir,
                    )
            
    # saving model state file    
    final_global_model_state_file = os.path.join(final_output_dir, "final_state_global.pt")
    logger.info(f"saving final client model state to {final_global_model_state_file}")
    torch.save(global_fl_model.state_dict(), final_global_model_state_file)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
