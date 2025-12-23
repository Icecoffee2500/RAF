#!/bin/bash

# HML / feddyn / no kd
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-nokd.pth --wandb --gpu 7 --test_res 64 48
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-nokd.pth --wandb --gpu 7 --test_res 128 96
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-nokd.pth --wandb --gpu 7 --test_res 192 144
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-nokd.pth --wandb --gpu 7 --test_res 256 192
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-nokd.pth --wandb --gpu 7 --test_res 320 240
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-nokd.pth --wandb --gpu 7 --test_res 384 288
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-nokd.pth --wandb --gpu 7 --test_res 512 384

# HML / feddyn / mrkd
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 64 48
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 128 96
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 192 144
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 256 192
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 320 240
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 384 288
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_feddyn-coef_1e-8-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 512 384

# HML / moon / no kd
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-1-4000-hml-nokd.pth --wandb --gpu 7 --test_res 64 48
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-1-4000-hml-nokd.pth --wandb --gpu 7 --test_res 128 96
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-1-4000-hml-nokd.pth --wandb --gpu 7 --test_res 192 144
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-1-4000-hml-nokd.pth --wandb --gpu 7 --test_res 256 192
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-1-4000-hml-nokd.pth --wandb --gpu 7 --test_res 320 240
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-1-4000-hml-nokd.pth --wandb --gpu 7 --test_res 384 288
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-1-4000-hml-nokd.pth --wandb --gpu 7 --test_res 512 384

uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-3-4000-hml-nokd.pth --wandb --gpu 7 --test_res 64 48
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-3-4000-hml-nokd.pth --wandb --gpu 7 --test_res 128 96
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-3-4000-hml-nokd.pth --wandb --gpu 7 --test_res 192 144
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-3-4000-hml-nokd.pth --wandb --gpu 7 --test_res 256 192
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-3-4000-hml-nokd.pth --wandb --gpu 7 --test_res 320 240
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-3-4000-hml-nokd.pth --wandb --gpu 7 --test_res 384 288
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-3-4000-hml-nokd.pth --wandb --gpu 7 --test_res 512 384

uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-nokd.pth --wandb --gpu 7 --test_res 64 48
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-nokd.pth --wandb --gpu 7 --test_res 128 96
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-nokd.pth --wandb --gpu 7 --test_res 192 144
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-nokd.pth --wandb --gpu 7 --test_res 256 192
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-nokd.pth --wandb --gpu 7 --test_res 320 240
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-nokd.pth --wandb --gpu 7 --test_res 384 288
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-nokd.pth --wandb --gpu 7 --test_res 512 384

uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-7-4000-hml-nokd.pth --wandb --gpu 7 --test_res 64 48
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-7-4000-hml-nokd.pth --wandb --gpu 7 --test_res 128 96
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-7-4000-hml-nokd.pth --wandb --gpu 7 --test_res 192 144
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-7-4000-hml-nokd.pth --wandb --gpu 7 --test_res 256 192
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-7-4000-hml-nokd.pth --wandb --gpu 7 --test_res 320 240
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-7-4000-hml-nokd.pth --wandb --gpu 7 --test_res 384 288
uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-7-4000-hml-nokd.pth --wandb --gpu 7 --test_res 512 384









uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained ../pretrained/mae_pretrain_vit_small.pth --wandb --gpu 0 --test_res 256 192





# fl_fedavg-4000-hhh-no_kd.pth
# fl_fedavg-4000-hhh-mrkd.pth

# fl_fedavg-4000-hml-no_kd.pth

# fl_fedbn-4000-hml-highBn-mrkd.pth
# fl_fedbn-4000-hml-midBn-mrkd.pth
# fl_fedbn-4000-hml-lowBn-mrkd.pth

# fl_fedbn-4000-hml-highBn-no_kd.pth
# fl_fedbn-4000-hml-midBn-no_kd.pth
# fl_fedbn-4000-hml-lowBn-no_kd.pth

# fl_feddyn-coef_1e-8-4000-hml-nokd.pth
# fl_feddyn-coef_1e-8-4000-hml-mrkd.pth
# fl_feddyn-coef_1e-9-4000-hml-nokd.pth
# fl_feddyn-coef_1e-9-4000-hml-mrkd.pth

# fl_moon-mu_con_1-4000-hml-nokd.pth
# fl_moon-mu_con_2-4000-hml-nokd.pth
# fl_moon-mu_con_5-4000-hml-nokd.pth
# fl_moon-mu_con_1e-1-4000-hml-nokd.pth
# fl_moon-mu_con_1e-2-4000-hml-nokd.pth
# fl_moon-mu_con_1e-3-4000-hml-nokd.pth
# fl_moon-mu_con_1e-4-4000-hml-nokd.pth
# fl_moon-mu_con_1e-5-4000-hml-nokd.pth
# fl_moon-mu_con_1e-6-4000-hml-nokd.pth
# fl_moon-mu_con_1e-7-4000-hml-nokd.pth
# fl_moon-mu_con_1e-8-4000-hml-nokd.pth

# fl_moon-mu_con_1e-8-4000-hml-mrkd.pth