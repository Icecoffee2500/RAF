#!/bin/bash

# For Code Test! Only Use This!
python3 raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 0 --samples_per_client 4000 --client_num 3 --train_bs 32 --kd_alpha 0.0 --loss_scale 1 --fed fedprox --kd_use --client_res high mid low


# fl - mr augmentation only
python3 raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 0 --samples_per_client 4000 --client_num 3 --train_bs 32 --kd_alpha 1.0 --loss_scale 1 --kd_use --client_res high mid low

# cl - mrkd (upper bound)
python3 raf/experiments/train_hpe_cl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 0 --train_bs 32 --kd_alpha 0.5 --loss_scale 1 --kd_use --client_res high

# For COCO!
# python3 raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/coco/vit/vit_small_multi_res_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 0 --gnc_split_num 4000 --gnc_num 3 --gnc_train_bs 32 --prc_split_num 4 --prc_num 0 --prc_train_bs 32 --kd_alpha 0.0 --loss_scale 1 --fed fedprox --kd_use --gnc_res high mid low