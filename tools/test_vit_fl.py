import sys, os
import os
import numpy as np

home_dir = os.path.dirname(os.path.abspath(__file__ + "/../"))
sys.path.append(os.path.join(home_dir, "lib"))
sys.path.append(home_dir)

import torch

from lib.models.backbones.vit import ViT
from lib.models.vit_pose import ViTPose
from lib.models.heads import TopdownHeatmapSimpleHead

from lib.core.config import config
from lib.utils.utils import ShellColors as sc
from lib.utils.utils import (
    save_checkpoint,
    create_logger_sfl,
    init_random_seed,
    set_random_seed,
    load_checkpoint,
    show_info,
    parse_args
)
from fl_client import FLClient

def main(args):
    wdb = None
    if args.wandb:
        import wandb
        from datetime import datetime
        
        now = datetime.now()
        today = now.strftime("%m%d_%H:%M")
        name = f"{args.pretrained}_val_{today}"
        wdb = wandb
        wdb.init(
            config=config,
            project="FL Validation",
            name = name,
        )
    
    # config.DATASET.NUMBER_OF_SPLITS = args.split_num
    
    #TODO: resolution 하나만 넣어도 List인지 확인하기.
    print(f"args.test_res: {args.test_res}")
    print(f"Type of args.test_res: {type(args.test_res)}")
    if len(args.test_res) == 2:
        print("hello")
        im_size_h = args.test_res[0]
        im_size_w = args.test_res[1]
        hm_size_h = int(im_size_h / 4)
        hm_size_w = int(im_size_w / 4)
        
        config.MODEL.IMAGE_SIZE = np.array([im_size_w, im_size_h])
        config.MODEL.HEATMAP_SIZE = np.array([hm_size_w, hm_size_h])
    
    show_info(0, args, config)
    
    print("------------- config image size ---------------------")
    print(config.MODEL.IMAGE_SIZE)
    print("------------- config heatmap size ---------------------")
    print(config.MODEL.HEATMAP_SIZE)

    if "small" in args.cfg:
        from lib.models.extra.vit_small_uncertainty_config import extra
    elif "large" in args.cfg:
        from lib.models.extra.vit_large_uncertainty_config import extra
    elif "huge" in args.cfg:
        from lib.models.extra.vit_huge_uncertainty_config import extra
    else:
        raise FileNotFoundError(f"Check config file name!!")


    logger, final_output_dir = create_logger_sfl(config, args.cfg, f"test_{args.gpu}")
    
    seed = init_random_seed(args.seed)
    logger.info(f"Set random seed to {seed}")
    set_random_seed(seed)
    
    pretrained_path = os.path.join(home_dir, args.pretrained)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    perf_indicator = 0.0
    
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
    if pretrained_path:
        load_checkpoint(global_fl_model, pretrained_path)
    
    # global fl model -> gpu
    global_fl_model.to(device)
    
    client = FLClient(
        idx=0, # dataset의 index
        config=config,
        device=device,
        init_model=global_fl_model,
        extra=extra,
        wdb=wdb,
        logger=logger,
        im_size=config.MODEL.IMAGE_SIZE,
        hm_size=config.MODEL.HEATMAP_SIZE,
        split_size=1,
        batch_size=32,
        is_proxy=False,
    )

    # ## Backbone - ViT
    # backbone = ViT(
    #     img_size=extra["backbone"]["img_size"],
    #     patch_size=extra["backbone"]["patch_size"],
    #     embed_dim=extra["backbone"]["embed_dim"],
    #     in_channels=3,
    #     num_heads=extra["backbone"]["num_heads"],
    #     depth=extra["backbone"]["depth"],
    #     qkv_bias=True,
    #     drop_path_rate=extra["backbone"]["drop_path_rate"],
    #     use_gpe=config.MODEL.USE_GPE,
    #     use_lpe=config.MODEL.USE_LPE,
    #     use_gap=config.MODEL.USE_GAP,
    # )

    # validate
    print(f"{sc.COLOR_LIGHT_PURPLE}--------------------------------------------------------------------------------------------------------{sc.ENDC}")
    print(f"{sc.COLOR_LIGHT_PURPLE}[Evaluate] {sc.COLOR_BROWN}{args.pretrained} {sc.COLOR_LIGHT_PURPLE}in {sc.COLOR_BROWN}{args.test_res} {sc.ENDC}")
    print(f"{sc.COLOR_LIGHT_PURPLE}--------------------------------------------------------------------------------------------------------{sc.ENDC}")
    
    # evaluate performance of each clients
    perf_indicator = client.evaluate(
        backbone=global_fl_model.backbone,
        keypoint_head=global_fl_model.keypoint_head,
        final_output_dir=final_output_dir,
        wdb=wdb,
    )



if __name__ == "__main__":
    args = parse_args()
    main(args=args)
