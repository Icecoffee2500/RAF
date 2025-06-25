import sys
import os
import argparse
import torchvision.transforms as transforms
from timm.data.random_erasing import RandomErasing
import copy

home_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from lib.models.backbones.vit import ViT
from lib.models.backbones.vit_client import ViT_client
from lib.models.backbones.vit_server import ViT_server
from lib.models.heads import TopdownHeatmapSimpleHead
from lib.dataset.coco import COCODataset

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import get_model_name
from lib.core.scheduler import MultistepWarmUpRestargets
# from lib.core.function import train, validate 
from lib.core.function_sfl import train, validate 

from lib.utils.utils import get_optimizer, get_vit_optimizer, get_vit_optimizer_general
from lib.utils.utils import get_loss
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger_sfl
from lib.utils.utils import ShellColors as sc
from lib.utils.utils import init_random_seed, set_random_seed
from lib.utils.utils import load_checkpoint
import numpy as np
from TrainScheduler import TrainScheduler

from client import Client
from server_train_proxy import Server

def show_info(gpu, args, config):
    print(f"{'='*20} Info {'='*20}")
    print(f"{sc.COLOR_GREEN}CURRENT GPU: {sc.ENDC}{gpu}")
    print(f"{sc.COLOR_GREEN}CONFIG: {sc.ENDC}{args.cfg}")
    print(f"{sc.COLOR_GREEN}CKPT CLIENT: {sc.ENDC}{args.ckpt_client}")
    print(f"{sc.COLOR_GREEN}CKPT SERVER: {sc.ENDC}{args.ckpt_server}")
    print(f"{sc.COLOR_GREEN}SEED: {sc.ENDC}{args.seed}")
    print(f"{sc.COLOR_GREEN}TRAIN BATCH SIZE: {sc.ENDC}{config.TRAIN.BATCH_SIZE}")
    print(f"{sc.COLOR_GREEN}TEST BATCH SIZE: {sc.ENDC}{config.TEST.BATCH_SIZE}")
    print(f"{sc.COLOR_CYAN}USE AMP: {sc.ENDC}{config.MODEL.USE_AMP}")
    print(f"{sc.COLOR_CYAN}USE UDP: {sc.ENDC}{config.TEST.USE_UDP}")
    print(f"{sc.COLOR_CYAN}USE FLIP: {sc.ENDC}{config.TEST.FLIP_TEST}")
    print(f"{sc.COLOR_CYAN}USE GT BBOX: {sc.ENDC}{config.TEST.USE_GT_BBOX}")
    print(f"{sc.COLOR_CYAN}USE UNCERTAINTY: {sc.ENDC}{config.LOSS.UNCERTAINTY}")
    print(f"{sc.COLOR_CYAN}USE WARMUP: {sc.ENDC}{args.warmup}")
    print(f"{sc.COLOR_CYAN}USE WANDB: {sc.ENDC}{args.wandb}")
    print(f"{'='*46}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="")
    parser.add_argument("--pretrained", default=None, help="checkpoint name", type=str)
    parser.add_argument("--ckpt-client", default=None, help="checkpoint name", type=str)
    parser.add_argument("--ckpt-server", default=None, help="checkpoint name", type=str)
    parser.add_argument("--wandb", help="use wandb", action="store_true")
    parser.add_argument("--warmup", help="use warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)
    parser.add_argument("--lr", default=0.1, help="")
    parser.add_argument("--resume", default=None, help="")
    parser.add_argument("--batch_size", type=int, default=768, help="")
    parser.add_argument("--num_workers", type=int, default=4, help="")
    parser.add_argument("--gpus", type=int, nargs="+", default=None, help="gpu numbers", required=True)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    # parser.add_argument("--dist-url", default="tcp://127.0.0.1:3457", type=str, help="")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:3456", type=str, help="")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="")
    parser.add_argument("--rank", default=0, type=int, help="")
    parser.add_argument("--world_size", default=1, type=int, help="")
    parser.add_argument("--distributed", action="store_true", help="")
    args = parser.parse_args()

    return args

# Federated averaging: FedAvg
def FedAvg(weights):
    w_avg = copy.deepcopy(weights[0]) # weight_averaged 초기화
    for k in w_avg.keys():
        for i in range(1, len(weights)):
            w_avg[k] += weights[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights))
    return w_avg

args = parse_args()
gpus = ",".join([str(id) for id in args.gpus])
print("gpus: ", gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def main():
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):
    wdb = None
    if args.wandb and gpu == 0:
        import wandb
        wdb = wandb
        wdb.init(
            config=config,
            # project="HPE_Validation_240701_384x288_sc_LPE_GPE",
            project="Test_241112_SFL_ViTPose-S_mr_mpii",
            # name=f"{config.MODEL.NAME}_{config.MODEL.TYPE}_{config.DATASET.DATASET}_{config.LOSS.HM_LOSS}_{config.LOSS.UNC_LOSS}_{config.TRAIN.END_EPOCH}",
            name=f"sflv2_{config.MODEL.NAME}_{config.MODEL.TYPE}_bs{config.TRAIN.BATCH_SIZE}/{config.TEST.BATCH_SIZE}_lr{config.TRAIN.LR}",
        )
        
    show_info(gpu, args, config)
    args.gpu = gpu

    if "small" in args.cfg:
        from lib.models.extra.vit_small_uncertainty_config import extra
    elif "large" in args.cfg:
        from lib.models.extra.vit_large_uncertainty_config import extra
    elif "huge" in args.cfg:
        from lib.models.extra.vit_huge_uncertainty_config import extra
    else:
        raise FileNotFoundError(f"Check config file name!!")

    ngpus_per_node = torch.cuda.device_count()
    print(f"Use GPU: {args.gpu} for training")
    print("ngpus_per_ndoe : ", ngpus_per_node)
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    logger, final_output_dir = create_logger_sfl(config, args.cfg, f"train_{args.gpu}")
    if args.seed is not None:
        seed = init_random_seed(args.seed)
        logger.info(f"Set random seed to {seed}")
        set_random_seed(seed)

    ## Global Client Model
    global_model_client = ViT_client(
        img_size=extra["backbone"]["img_size"],
        patch_size=extra["backbone"]["patch_size"],
        embed_dim=extra["backbone"]["embed_dim"],
        in_channels=3,
        depth=extra["backbone"]["depth"],
        num_heads=extra["backbone"]["num_heads"],
        qkv_bias=True,
        drop_path_rate=extra["backbone"]["drop_path_rate"],
        use_gpe=config.MODEL.USE_GPE,
        use_lpe=config.MODEL.USE_LPE,
    )
    
    if args.ckpt_client is not None:
        ckpt_client_path = os.path.join(home_dir, args.ckpt_client)
        client_state_dict = torch.load(ckpt_client_path, map_location='cpu')
        global_model_client.load_state_dict(client_state_dict)
    
    global_model_client.cuda(args.gpu)
    global_model_client = DDP(
        global_model_client, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False
    )
        
    torch.cuda.set_device(args.gpu)
    config.gpu = gpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    criterion = get_loss(config)

    if config.MODEL.FREEZE_NAME:
        print("Freeze Group : ", config.MODEL.FREEZE_NAME)
    if config.MODEL.DIFF_NAME:
        print("Diff Group : ", config.MODEL.DIFF_NAME)
        
    num_users = 3
    client_list = []
    
    # Client 3개 초기화
    for idx in range(num_users):
        client = Client(
            idx=idx,
            config=config,
            gpu=gpu,
            device=device,
            init_model=global_model_client,
        )
        client_list.append(client)
    
    if args.ckpt_server is not None:
        ckpt_server_path = os.path.join(home_dir, args.ckpt_server)
    else:
        ckpt_server_path = None
    
    # Server 초기화
    server = Server(extra, config, args, gpu, ckpt_server_path=ckpt_server_path)
    
            
    for client_idx, client in enumerate(client_list):
        if gpu == 0:
            print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
            print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_PURPLE}Evaluating{sc.ENDC} ------------")
            print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
        perf_indicator = client.evaluate(
            server=server,
            final_output_dir=final_output_dir,
            wdb=wdb,
            criterion=criterion
        )


if __name__ == "__main__":
    main()
