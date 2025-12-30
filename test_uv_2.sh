#!/bin/bash

# uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 64 48
# uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 128 96
# uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 192 144
# uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 256 192
# uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 320 240
# uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 384 288
# uv run raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_moon-mu_con_1e-5-4000-hml-mrkd.pth --wandb --gpu 7 --test_res 512 384

python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 128 96 --test_interpolate --interpolate_im_shape 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 128 96 --test_interpolate --interpolate_im_shape 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 128 96 --test_interpolate --interpolate_im_shape 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 128 96 --test_interpolate --interpolate_im_shape 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 128 96 --test_interpolate --interpolate_im_shape 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 128 96 --test_interpolate --interpolate_im_shape 512 384

python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedavg-4000-hml-no_kd.pth --wandb --gpu 7 --test_res 512 384