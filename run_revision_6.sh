#!/bin/bash

# FedAvg-NoKD-HHH
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 6 --samples_per_client 4000 --client_num 3 --train_bs 32 --loss_scale 1 --client_res high high high

# FedAvg-NoKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 6 --samples_per_client 4000 --client_num 3 --train_bs 32 --loss_scale 1 --client_res high mid low

# FedBN-NoKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 6 --samples_per_client 4000 --client_num 3 --train_bs 32 --loss_scale 1 --client_res high mid low --fed fedbn


# FedDyn-NoKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 6 --samples_per_client 4000 --client_num 3 --train_bs 32 --loss_scale 1 --client_res high mid low --fed feddyn --dyn_alpha 1e-7

# FedDyn-MRKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 7 --samples_per_client 4000 --client_num 3 --train_bs 32 --kd_alpha 0.5 --loss_scale 1 --kd_use --client_res high mid low --fed feddyn --dyn_alpha 1e-7