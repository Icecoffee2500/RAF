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

from lib.utils.utils import get_optimizer, get_vit_optimizer
from lib.utils.utils import get_loss
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger_sfl
from lib.utils.utils import ShellColors as sc
from lib.utils.utils import init_random_seed, set_random_seed
from lib.utils.utils import load_checkpoint
import numpy as np

from client import Client
from server import Server

def show_info(gpu, args, config):
    print(f"{'='*20} Info {'='*20}")
    print(f"{sc.COLOR_GREEN}CURRENT GPU: {sc.ENDC}{gpu}")
    print(f"{sc.COLOR_GREEN}CONFIG: {sc.ENDC}{args.cfg}")
    print(f"{sc.COLOR_GREEN}WEIGHT: {sc.ENDC}{args.weight}")
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
    parser.add_argument("--weight", help="checkpoint name", required=True, type=str)
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
            project="241112_SFL_ViTPose-S_mr_mpii",
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
    
    checkpoint_path = os.path.join(home_dir, args.weight)
    # weight initialization
    if "mae" in checkpoint_path:
        load_checkpoint(global_model_client, checkpoint_path)
    
    global_model_client.cuda(args.gpu)
    global_model_client = DDP(
        global_model_client, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False
    )

    checkpoint_path = os.path.join(home_dir, args.weight)
        
    torch.cuda.set_device(args.gpu)
    config.gpu = gpu

    #TODO: client -> train() 으로 옮기기
    criterion = get_loss(config)
    optimizer_client = get_vit_optimizer(config, global_model_client, extra)

    # Print optimizer
    # print(f"{sc.OKBLUE} Optimizer : {optimizer} {sc.ENDC}")
    # print()

    lr_scheduler_client = MultistepWarmUpRestargets(
        optimizer_client, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR
    )
    
    warmup_scheduler = None

    best_perf = 0.0
    perf_indicator = 0.0
    best_model = False
    step_scale_count = 0 
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
            optimizer=optimizer_client,
            gpu=gpu,
            init_model=global_model_client
        )
        client_list.append(client)
    
    # Server 초기화
    server = Server(extra, config, args, checkpoint_path, gpu)
    
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        w_locals_client = []
        
        for client_idx, client in enumerate(client_list):
            
            if gpu == 0:
                print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
                print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_BLUE}Training{sc.ENDC} ------------")
                print(f"{sc.COLOR_LIGHT_BLUE}------------------------------------------------------------{sc.ENDC}")
            
            lr_scheduler_client.step()
            
            # train하고 backward된 weight 겟
            w_client = client.train(
                net=copy.deepcopy(global_model_client),
                server=server,
                epoch=epoch,
                final_output_dir=final_output_dir,
                wdb=wdb,
                args=args,
                extra=extra,
                checkpoint_path=checkpoint_path,
                criterion=criterion,
            )
            # w_client = copy.deepcopy(global_model_client.state_dict())
            
            w_locals_client.append(copy.deepcopy(w_client))
            
            lr_ = lr_scheduler_client.get_lr()
            for i, g in enumerate(optimizer_client.param_groups):
                g["lr"] = lr_[i]

            # evaluate on validation set
            if epoch % config.EVALUATION.INTERVAL == 0:
                model_state_file = os.path.join(
                    final_output_dir,
                    f"{config.MODEL.NAME}_{config.LOSS.HM_LOSS}_{epoch}.pth.tar",
                )
                logger.info(f"saving final model state to {model_state_file}")
                torch.save(global_model_client.module.state_dict(), model_state_file)

                perf_indicator = 0.
                
                if gpu == 0:
                    print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
                    print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_PURPLE}Evaluating{sc.ENDC} ------------")
                    print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
                perf_indicator = client.evaluate(
                    net=global_model_client,
                    server=server,
                    final_output_dir=final_output_dir,
                    wdb=wdb,
                    criterion=criterion
                )
            
            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            if best_model:
                logger.info(f"=> saving checkpoint to {final_output_dir}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_client": get_model_name(config),
                        "client_state_dict": global_model_client.state_dict(),
                        "server_state_dict": server.model.state_dict(),
                        "perf": perf_indicator,
                        "optimizer": optimizer_client.state_dict(),
                        "HM_LOSS": config.LOSS.HM_LOSS,
                        "unc_loss": config.LOSS.UNC_LOSS,
                    },
                    best_model,
                    final_output_dir,
                )
            
        # After serving all clients for its local epochs------------
        # Federation process at Client-Side------------------------
        if gpu == 0:
            print(f"{sc.COLOR_RED}------------------------------------------------------------{sc.ENDC}")
            print(f"{sc.COLOR_RED}------ Fed Server: Federation process at Client-Side -------{sc.ENDC}")
            print(f"{sc.COLOR_RED}------------------------------------------------------------{sc.ENDC}")
        w_glob_client = FedAvg(w_locals_client) # 각 client에서 update된 weight를 받아서 FedAvg로 합쳐줌.
        if gpu == 0:
            logger.info("Federation Process Done!")
        
        # Update client-side global model
        if gpu == 0:
            logger.info("load Fed-Averaged weight to the global client model ...")
        global_model_client.load_state_dict(w_glob_client)

    final_model_state_file = os.path.join(final_output_dir, "final_state.pth.tar")
    logger.info(f"saving final model state to {final_model_state_file}")
    torch.save(global_model_client.module.state_dict(), final_model_state_file)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
