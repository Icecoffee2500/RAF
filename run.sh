#!/bin/bash

# For Test! Only Use This!
python3 tools/train_vit_fl.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained checkpoints/mae_pretrain_vit_small.pth --wandb --gpu 0 --gnc_split_num 4 --gnc_num 3 --gnc_train_bs 32 --prc_split_num 4 --prc_num 0 --prc_train_bs 32 --kd_alpha 0.0 --loss_scale 1 --fed fedprox --kd_use --gnc_res high mid low