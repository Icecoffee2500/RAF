import sys
import os

home_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

import torch

from lib.utils.utils import ShellColors as sc
from lib.core.config import config
from lib.core.config import get_model_name
from lib.utils.utils import (
    save_checkpoint,
    create_logger_sfl,
    init_random_seed,
    set_random_seed,
    load_checkpoint,
    show_info,
    parse_args
)
from lib.models.backbones.vit_client import ViT_client
from client import Client
# from TrainScheduler_train_proxy import TrainScheduler
# from TrainScheduler_train_proxy_future import TrainScheduler
from TrainScheduler_train_proxy_future import Trainer
from server_train_proxy import Server
from tools.fed_server import FedServer
from lib.utils.utils import clone_parameters
from collections import OrderedDict
    
def main(args):
    wdb = None
    if args.wandb:
        import wandb
        from datetime import datetime
        
        now = datetime.now()
        today = now.strftime("%m%d_%H:%M")
        name = f"{args.split_num * 1000}"
        name += "_disjoint"
        name += f"_cascaded" if args.cascade else ""
        if args.data_aug is not None:
            name += f"_{args.data_aug}"
            name += "_same_pos" if args.same_pos else "_diff_pos"
            name += "_clean_high" if args.clean_high else ""
        else:
            name += "_no_augs"
        name += f"_{args.fed}"
        # name += f"_kd_use" if args.kd_use else ""
        # name += f"_proxy_gt+kd" if args.kd_use else ""
        name += "_only_kd" if args.kd_use else ""
        name += f"_alpha={args.kd_alpha}"
        name += f"_{today}"
        wdb = wandb
        wdb.init(
            config=config,
            # project=f"SFL_ViTPose-S_MR_MPII_single_gpu_{args.split_num * 1000}",
            project="testtesttest",
            name = name,
        )
    
    config.DATASET.AUGMENTATION = args.data_aug
    config.DATASET.SAME_POS = True if args.same_pos else False
    config.DATASET.CLEAN_HIGH = True if args.clean_high else False
    config.FED.FEDAVG = True if args.fed == "fedavg" else False
    config.FED.FEDPROX = True if args.fed == "fedprox" else False
    config.KD_USE = True if args.kd_use else False
    config.DATASET.NUMBER_OF_SPLITS = args.split_num
    config.KD_ALPHA = args.kd_alpha
    
    show_info(0, args, config)

    if "small" in args.cfg:
        from lib.models.extra.vit_small_uncertainty_config import extra
    elif "large" in args.cfg:
        from lib.models.extra.vit_large_uncertainty_config import extra
    elif "huge" in args.cfg:
        from lib.models.extra.vit_huge_uncertainty_config import extra
    else:
        raise FileNotFoundError(f"Check config file name!!")

    logger, final_output_dir = create_logger_sfl(config, args.cfg, f"train_{args.gpu}")
    
    seed = init_random_seed(args.seed)
    logger.info(f"Set random seed to {seed}")
    set_random_seed(seed)
    
    client_backbone_block_num = 1
    
    ## Global Client Model
    global_model_client = ViT_client(
        img_size=extra["backbone"]["img_size"],
        patch_size=extra["backbone"]["patch_size"],
        embed_dim=extra["backbone"]["embed_dim"],
        in_channels=3,
        # depth=extra["backbone"]["depth"],
        depth=client_backbone_block_num,
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
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    best_perf = 0.0
    perf_indicator = 0.0
    best_model = False

    if config.MODEL.FREEZE_NAME:
        print("Freeze Group : ", config.MODEL.FREEZE_NAME)
    if config.MODEL.DIFF_NAME:
        print("Diff Group : ", config.MODEL.DIFF_NAME)
        
    # global client model -> gpu
    global_model_client.to(device)
    
    num_users = 3
    # num_users = 4
    client_list = []
    
    # Client 3개 생성
    for idx in range(num_users):
        client = Client(
            idx=idx,
            config=config,
            gpu=0,
            device=device,
            init_model=global_model_client,
            extra=extra,
            client_backbone_block_num=client_backbone_block_num
        )
        client_list.append(client)
    
    proxy_client = Client(
        idx=10,
        # idx=4,
        config=config,
        gpu=0,
        device=device,
        init_model=global_model_client,
        extra=extra,
        client_backbone_block_num=client_backbone_block_num,
        is_proxy=True
    )
    
    # Server 생성
    server = Server(
        extra=extra,
        config=config,
        gpu=0,
        device=device,
        pretrained_path=pretrained_path
    )
    fed_server = FedServer()
    
    trainer = Trainer(
        config=config,
        wdb=wdb,
        clients=client_list,
        server=server,
        gpu=0,
        proxy_client=proxy_client,
        glob_client_model=global_model_client
    )
    
    best_perf_buf = [0.0 for _ in range(num_users)]
    avg_perf_buf = [0.0]
    
    # -- momentum schedule
    num_epochs = config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH
    ema = [0.996, 1.0]
    # ema = [0.99, 1.0]
    # ema = [0.9, 1.0]
    ipe = len(proxy_client.train_loader)
    momentum_scheduler = (
        ema[0] + i * (ema[1]-ema[0]) / (ipe * num_epochs)
        for i in range(int(ipe * num_epochs) + 1)
    )
    
    init_time = datetime.now()
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        epoch_e_time = datetime.now() - init_time
        init_time = datetime.now()
        
        # scheduler step
        for client in client_list:
            client.lr_scheduler_client.step()
        proxy_client.lr_scheduler_client.step()
        server.lr_scheduler_server.step()

        # -------- Train ---------------
        # ------------------------------------------------------------------------------------------------------------------------
        if args.cascade:
            # train disjoint clients
            trainer.train_each_client(logger, epoch)
            # aggregate disjoint clients
            client_weights = [client.model.state_dict() for client in client_list]
            w_glob_client = fed_server.aggregate(logger, client_weights)
            
            # if epoch > 40: # warm start after 40 epochs
            # load global client model weight to proxy client model
            logger.info(">>> load Fed-Averaged weight to the proxy client model ...")
            proxy_client.model.load_state_dict(w_glob_client)
            
            # save global model params
            # global_client_model_params = clone_parameters(OrderedDict(w_glob_client))
            # trainer.set_parameters(global_client_model_params)
            
            # train proxy clients
            # trainer.train_proxy_kd(device, logger, epoch)
            # trainer.train_split_agent(logger, epoch)
            trainer.train_proxy_takd(device, logger, epoch)
            # trainer.train_proxy_kd(device, logger, epoch, mu_=1.0)
            # trainer.train_proxy_kd(device, logger, epoch, mu_=0.1)
            # trainer.train_proxy_kd(device, logger, epoch, mu_=0.01)
            # trainer.train_proxy_kd(device, logger, epoch, momentum_scheduler=momentum_scheduler)
            
            # load proxy model weight to all clients
            w_glob_client = proxy_client.model.state_dict()
            logger.info(">>> load proxy client weight to the all clients' model ...")
            
            for client in client_list:
                client.model.load_state_dict(w_glob_client)
            
            # global client model for MOCO
            trainer.glob_client_model.load_state_dict(w_glob_client)
            
            global_model_client.load_state_dict(w_glob_client)
        else:
            # trainer.train_full_shared(device, logger, epoch)
            trainer.train_each_client(logger, epoch)
            # trainer.train_proxy_kd(device, logger, epoch)
            trainer.train_proxy_takd(device, logger, epoch)
            
            # aggregate weights
            w_client_proxy = proxy_client.model.state_dict()
            client_weights = [client.model.state_dict() for client in client_list]
            client_weights.append(w_client_proxy)
            w_glob_client = fed_server.aggregate(logger, client_weights)
            # w_glob_client = fed_server.aggregate(logger, [w_glob_client, w_client_proxy])
            
            # m = next(momentum_scheduler)
            # for key in w_glob_client.keys():
            #     w_glob_client[key].data.mul_(m).add_((1.-m) * w_client_proxy[key].detach().data)
            
            # Update client-side global model
            logger.info(">>> load Fed-Averaged weight to the global client model ...")
            for client in client_list:
                client.model.load_state_dict(w_glob_client)
            logger.info(">>> load Fed-Averaged weight to the proxy client model ...")
            proxy_client.model.load_state_dict(w_glob_client)
            
            global_model_client.load_state_dict(w_glob_client)
        
        logger.info(f"This epoch takes {epoch_e_time}\n")
        # ------------------------------------------------------------------------------------------------------------------------
        
        lr_p = proxy_client.lr_scheduler_client.get_lr()
        for i, g in enumerate(proxy_client.optimizer_client.param_groups):
            g["lr"] = lr_p[i]
        
        for client in client_list:
            lr_ = client.lr_scheduler_client.get_lr()
            for i, g in enumerate(client.optimizer_client.param_groups):
                g["lr"] = lr_[i]
        
        lr_ = server.lr_scheduler_server.get_lr()
        for i, g in enumerate(server.optimizer_server.param_groups):
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
            torch.save(global_model_client.state_dict(), client_model_state_file)
            torch.save(server.model.state_dict(), server_model_state_file)
            
            curr_avg_perf = 0
            for client_idx, client in enumerate(client_list):
                print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
                print(f"--------------- Client {sc.COLOR_BROWN}{client_idx} {sc.COLOR_LIGHT_PURPLE}Evaluating{sc.ENDC} ------------")
                print(f"{sc.COLOR_LIGHT_PURPLE}------------------------------------------------------------{sc.ENDC}")
                
                perf_indicator = client.evaluate(
                    server=server,
                    final_output_dir=final_output_dir,
                    wdb=wdb,
                )
                curr_avg_perf = curr_avg_perf + perf_indicator / len(client_list)
            
            # avg perf가 가장 높은 성능이 나왔을 때만 model save
            if curr_avg_perf > max(avg_perf_buf):
                logger.info(f"Epoch [{epoch}] best avg performance detected: {max(avg_perf_buf):.4f} => {curr_avg_perf:.4f}")
                avg_perf_buf.append(round(float(curr_avg_perf), 4))
                logger.info(f"Current Best Average Performances: {avg_perf_buf}")
                
                # logging
                logger.info(f"=> saving best client checkpoint to {final_output_dir}")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_client": get_model_name(config),
                        "client_state_dict": global_model_client.state_dict(),
                        "server_state_dict": server.model.state_dict(),
                        "perf": perf_indicator,
                        "optimizer": client_list[0].optimizer_client.state_dict(),
                        "HM_LOSS": config.LOSS.HM_LOSS,
                        "unc_loss": config.LOSS.UNC_LOSS,
                        "client_idx": client_idx
                    },
                    final_output_dir,
                )
            
    # saving model state file    
    final_model_client_state_file = os.path.join(final_output_dir, "final_state_client.pt")
    final_model_server_state_file = os.path.join(final_output_dir, "final_state_server.pt")
    logger.info(f"saving final client model state to {final_model_client_state_file}")
    logger.info(f"saving final server model state to {final_model_server_state_file}")
    torch.save(global_model_client.state_dict(), final_model_client_state_file)
    torch.save(server.model.state_dict(), final_model_client_state_file)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
