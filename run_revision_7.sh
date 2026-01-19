#!/bin/bash

# # FedAvg-MRKD-HHH
# uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 7 --samples_per_client 4000 --client_num 3 --train_bs 32 --kd_alpha 0.5 --loss_scale 1 --kd_use --client_res high high high

# # FedBN-MRKD-HML
# uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 7 --samples_per_client 4000 --client_num 3 --train_bs 32 --kd_alpha 0.5 --loss_scale 1 --kd_use --client_res high mid low --fed fedbn

# MOON-mu_1e-2-NoKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 7 --samples_per_client 4000 --client_num 3 --train_bs 32 --client_res high mid low --fed moon --mu_con 1e-2

# MOON-mu_1e-4-NoKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 7 --samples_per_client 4000 --client_num 3 --train_bs 32 --client_res high mid low --fed moon --mu_con 1e-4

# MOON-mu_1e-6-NoKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 7 --samples_per_client 4000 --client_num 3 --train_bs 32 --client_res high mid low --fed moon --mu_con 1e-6

# MOON-mu_1e-8-NoKD-HML
uv run raf/experiments/train_hpe_fl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 7 --samples_per_client 4000 --client_num 3 --train_bs 32 --client_res high mid low --fed moon --mu_con 1e-8

# CL
uv run raf/experiments/train_hpe_cl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 0 --samples_per_client 12000 --train_bs 32 --loss_scale 1 --client_res high

# CL - vit-h / 576x432 - no kd
uv run raf/experiments/train_hpe_cl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-huge.yaml --pretrained ../pretrained/mae_pretrain_vit_huge.pth --wandb --gpu 6 --samples_per_client 12000 --train_bs 8 --test_bs 16 --loss_scale 1 --client_res max_high
# CL - vit-h / 576x432 - mrkd
uv run raf/experiments/train_hpe_cl.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-huge.yaml --pretrained ../pretrained/mae_pretrain_vit_huge.pth --wandb --gpu 0 --samples_per_client 12000 --train_bs 4 --test_bs 4 --loss_scale 1 --client_res max_high --kd_use --kd_alpha 0.5