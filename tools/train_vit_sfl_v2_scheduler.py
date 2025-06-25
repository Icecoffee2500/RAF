import sys
import os
import argparse

home_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.utils.utils import ShellColors as sc
from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import get_model_name

def show_info(gpu, args, config):
    print(f"{'='*20} Info {'='*20}")
    print(f"{sc.COLOR_GREEN}CURRENT GPU: {sc.ENDC}{gpu}")
    print(f"{sc.COLOR_GREEN}CONFIG: {sc.ENDC}{args.cfg}")
    print(f"{sc.COLOR_GREEN}PRETRAINED: {sc.ENDC}{args.pretrained}")
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
    print(f"{sc.COLOR_CYAN}USE CUTMIX: {sc.ENDC}{args.cutmix}")
    print(f"{sc.COLOR_CYAN}USE CUTOUT: {sc.ENDC}{args.cutout}")
    print(f"{sc.COLOR_CYAN}USE SAME_POS: {sc.ENDC}{args.same_pos}")
    print(f"{sc.COLOR_CYAN}USE CLEAN_HIGH: {sc.ENDC}{args.clean_high}")
    print(f"{sc.COLOR_CYAN}AGGREGATE METHOD: {sc.ENDC}{args.fed}")
    print(f"{sc.COLOR_CYAN}FEDAVG: {sc.ENDC}{args.fed == 'fedavg'}")
    print(f"{sc.COLOR_CYAN}FEDPROX: {sc.ENDC}{args.fed == 'fedprox'}")
    print(f"{sc.COLOR_CYAN}USE KD_USE: {sc.ENDC}{args.kd_use}")
    print(f"{sc.COLOR_CYAN}NUMBER OF SPLITS: {sc.ENDC}{args.split_num}")
    print(f"{'='*46}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    parser.add_argument("--cfg", help="")
    parser.add_argument("--pretrained", help="checkpoint name", required=True, type=str)
    parser.add_argument("--ckpt_client", default=None, help="checkpoint name", type=str)
    parser.add_argument("--ckpt_server", default=None, help="checkpoint name", type=str)
    parser.add_argument("--wandb", help="use wandb", action="store_true")
    parser.add_argument("--warmup", help="use warmup", action="store_true")
    # parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--cutmix", action="store_true", help="cutmix augmentation")
    parser.add_argument("--cutout", action="store_true", help="cutout augmentation")
    parser.add_argument("--same_pos", action="store_true", help="same_pos")
    parser.add_argument("--clean_high", action="store_true", help="clean_high")
    parser.add_argument("--fed", type=str, default="fedavg", help="aggregate method")
    parser.add_argument("--kd_use", action="store_true", help="knowledge distillation")
    parser.add_argument("--split_num", type=int, default=1, help="number of split")
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

args = parse_args()
gpus = ",".join([str(id) for id in args.gpus])
print("gpus: ", gpus)
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def main():
    # mp.set_start_method('spawn', force=True)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    print(f"ngpus per node: {ngpus_per_node}")
    print(f"args.world_size: {args.world_size}")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    
def main_worker(gpu, ngpus_per_node, args):
    # Import required modules here
    from lib.core.scheduler import MultistepWarmUpRestargets

    from lib.utils.utils import get_vit_optimizer_general
    from lib.utils.utils import get_loss
    from lib.utils.utils import save_checkpoint
    from lib.utils.utils import create_logger_sfl
    
    from lib.utils.utils import init_random_seed, set_random_seed
    from lib.utils.utils import load_checkpoint

    from lib.models.backbones.vit_client import ViT_client
    from client import Client
    # from TrainScheduler import TrainScheduler
    from TrainScheduler_train_proxy import TrainScheduler
    # from server_for_proxy_integration import Server
    from server_train_proxy import Server
    
    print("---------------------------- Hello World!!! ---------------------------------------")
    wdb = None
    if args.wandb and gpu == 0:
        import wandb
        from datetime import datetime
        
        now = datetime.now()
        today = now.strftime("%m%d_%H:%M")
        name = f"{args.split_num * 1000}"
        if args.cutmix:
            name += f"_CUTMIX"
        elif args.cutmix:
            name += f"_CUTOUT"
        else:
            name += "_NO_AUGs"
        if args.cutmix or args.cutout:
            name += "_SAME_POS" if args.same_pos else "_DIFF_POS"
            name += "_CLEAN_HIGH" if args.clean_high else ""
        name += f"_{args.fed}"
        name += f"_KD_USE" if args.kd_use else ""
        name += f"_{today}"
        wdb = wandb
        wdb.init(
            config=config,
            # project="241112_SFL_ViTPose-S_mr_mpii",
            # project="250110_SFL_ViTPose-S_MR_MPII",
            project="250201_SFL_ViTPose-S_MR_MPII_data_num_changing",
            name = name,
            # name=f"tarin_proxy_split_train_sflv2_{config.MODEL.NAME}_{config.MODEL.TYPE}_bs{config.TRAIN.BATCH_SIZE}/{config.TEST.BATCH_SIZE}_lr{config.TRAIN.LR}",
            # name=f"train_kd_full_mpii_sflv2_{config.MODEL.NAME}_{config.MODEL.TYPE}_bs{config.TRAIN.BATCH_SIZE}/{config.TEST.BATCH_SIZE}_lr{config.TRAIN.LR}",
        )
    
    config.DATASET.CUTMIX = True if args.cutmix else False
    config.DATASET.CUTOUT = True if args.cutout else False
    config.DATASET.SAME_POS = True if args.same_pos else False
    config.DATASET.CLEAN_HIGH = True if args.clean_high else False
    config.FED.FEDAVG = True if args.fed == "fedavg" else False
    config.FED.FEDPROX = True if args.fed == "fedprox" else False
    config.KD_USE = True if args.kd_use else False
    config.DATASET.NUMBER_OF_SPLITS = args.split_num
    
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
    # if args.seed is not None:
    #     seed = init_random_seed(args.seed)
    #     logger.info(f"Set random seed to {seed}")
    #     set_random_seed(seed)
    
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
    
    pretrained_path = os.path.join(home_dir, args.pretrained)
    
    # client model weight initialization
    if "mae" in pretrained_path:
        load_checkpoint(global_model_client, pretrained_path)
    
    torch.cuda.set_device(args.gpu)
    config.gpu = gpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    

    #TODO: client -> train() 으로 옮기기
    criterion = get_loss(config)
    optimizer_client = get_vit_optimizer_general(config, global_model_client, extra)

    lr_scheduler = MultistepWarmUpRestargets(
        optimizer_client, milestones=config.TRAIN.LR_STEP, gamma=config.TRAIN.LR_FACTOR
    )

    best_perf = 0.0
    perf_indicator = 0.0
    best_model = False

    if config.MODEL.FREEZE_NAME:
        print("Freeze Group : ", config.MODEL.FREEZE_NAME)
    if config.MODEL.DIFF_NAME:
        print("Diff Group : ", config.MODEL.DIFF_NAME)
        
    # global client model -> gpu
    global_model_client.to(device)
    
    # global clinet model -> ddp
    global_model_client = DDP(
        global_model_client, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False
    )
    
    num_users = 3
    client_list = []
    
    # Client 3개 생성
    for idx in range(num_users):
        client = Client(
            idx=idx,
            config=config,
            optimizer=optimizer_client,
            gpu=gpu,
            device=device,
            init_model=global_model_client,
        )
        client_list.append(client)
    
    # Server 생성
    server = Server(extra, config, args, gpu, pretrained_path=pretrained_path)
    
    train_scheduler = TrainScheduler(
        config=config,
        wdb=wdb,
        clients=client_list,
        server=server,
        criterion=criterion,
        gpu=args.gpu
    )
    
    best_perf_buf = [0.0 for _ in range(num_users)]
    avg_perf_buf = [0.0]
    
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # -------- Train ---------------
        w_glob_client = train_scheduler.train(device, logger, epoch)
        global_model_client.load_state_dict(w_glob_client)

        #TODO: 제대로 확인 필요    
        lr_ = lr_scheduler.get_lr()
        for i, g in enumerate(optimizer_client.param_groups):
            g["lr"] = lr_[i]

        # -------- Test ---------------
        # evaluate on validation set
        if epoch % config.EVALUATION.INTERVAL == 0:
            client_model_state_file = os.path.join(
                final_output_dir,
                f"{config.MODEL.NAME}_{config.MODEL.TYPE}_client_global_{config.LOSS.HM_LOSS}_{epoch}.pt",
            )
            server_model_state_file = os.path.join(
                final_output_dir,
                f"{config.MODEL.NAME}_{config.MODEL.TYPE}_server_{config.LOSS.HM_LOSS}_{epoch}.pt",
            )
            logger.info(f"saving final client model state to {client_model_state_file}")
            logger.info(f"saving final server model state to {server_model_state_file}")
            torch.save(global_model_client.module.state_dict(), client_model_state_file)
            torch.save(server.model.module.state_dict(), server_model_state_file)
            
            curr_avg_perf = 0
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
                curr_avg_perf = curr_avg_perf + perf_indicator / len(client_list)
                
                # 현재 client의 model 성능이 현재 client의 이전 best model 성능보다 높으면 best 갱신
                # if perf_indicator > best_perf:
                # if perf_indicator > best_perf_buf[client_idx]:
                #     logger.info(f"Epoch [{epoch}] Client {client_idx} best performance detected: {best_perf_buf[client_idx]:.4f} => {perf_indicator:.4f}")
                #     best_perf_buf[client_idx] = perf_indicator
                #     logger.info(f"Current best performance list: {best_perf_buf[0]:.4f} / {best_perf_buf[1]:.4f} / {best_perf_buf[2]:.4f}")
                    
                #     # logging
                #     logger.info(f"=> saving best client [{client_idx}] checkpoint to {final_output_dir}")
                #     save_checkpoint(
                #         {
                #             "epoch": epoch + 1,
                #             "model_client": get_model_name(config),
                #             "client_state_dict": global_model_client.module.state_dict(), # 확인 필요
                #             "server_state_dict": server.model.module.state_dict(), # 확인 필요
                #             "perf": perf_indicator,
                #             "optimizer": optimizer_client.state_dict(), # 확인 필요
                #             "HM_LOSS": config.LOSS.HM_LOSS,
                #             "unc_loss": config.LOSS.UNC_LOSS,
                #             "client_idx": client_idx
                #         },
                #         final_output_dir,
                #     )
            
            # avg perf가 가장 높은 성능이 나왔을 때만 model save
            if curr_avg_perf > max(avg_perf_buf):
                logger.info(f"Epoch [{epoch}] best avg performance detected: {max(avg_perf_buf):.4f} => {perf_indicator:.4f}")
                avg_perf_buf.append(round(float(curr_avg_perf), 4))
                logger.info(f"Current Best Average Performances: {avg_perf_buf}")
                
                # logging
                logger.info(f"=> saving best client checkpoint to {final_output_dir}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_client": get_model_name(config),
                        "client_state_dict": global_model_client.module.state_dict(), # 확인 필요
                        "server_state_dict": server.model.module.state_dict(), # 확인 필요
                        "perf": perf_indicator,
                        "optimizer": optimizer_client.state_dict(), # 확인 필요
                        "HM_LOSS": config.LOSS.HM_LOSS,
                        "unc_loss": config.LOSS.UNC_LOSS,
                        "client_idx": client_idx
                    },
                    final_output_dir,
                )
            
        
    final_model_client_state_file = os.path.join(final_output_dir, "final_state_client.pt")
    final_model_server_state_file = os.path.join(final_output_dir, "final_state_server.pt")
    logger.info(f"saving final client model state to {final_model_client_state_file}")
    logger.info(f"saving final server model state to {final_model_server_state_file}")
    torch.save(global_model_client.module.state_dict(), final_model_client_state_file)
    torch.save(server.model.module.state_dict(), final_model_client_state_file)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
