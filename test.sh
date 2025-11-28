#!/bin/bash

# high / fedbn / no kd
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-no_kd.pth --wandb --gpu 0 --test_res 64 48
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-no_kd.pth --wandb --gpu 0 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-no_kd.pth --wandb --gpu 0 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-no_kd.pth --wandb --gpu 0 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-no_kd.pth --wandb --gpu 0 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-no_kd.pth --wandb --gpu 0 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-no_kd.pth --wandb --gpu 0 --test_res 512 384

# mid / fedbn / no kd
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-no_kd.pth --wandb --gpu 0 --test_res 64 48
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-no_kd.pth --wandb --gpu 0 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-no_kd.pth --wandb --gpu 0 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-no_kd.pth --wandb --gpu 0 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-no_kd.pth --wandb --gpu 0 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-no_kd.pth --wandb --gpu 0 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-no_kd.pth --wandb --gpu 0 --test_res 512 384

# low / fedbn / no kd
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-no_kd.pth --wandb --gpu 0 --test_res 64 48
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-no_kd.pth --wandb --gpu 0 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-no_kd.pth --wandb --gpu 0 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-no_kd.pth --wandb --gpu 0 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-no_kd.pth --wandb --gpu 0 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-no_kd.pth --wandb --gpu 0 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-no_kd.pth --wandb --gpu 0 --test_res 512 384

# high / fedbn / mrkd
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-mrkd.pth --wandb --gpu 0 --test_res 64 48
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-mrkd.pth --wandb --gpu 0 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-mrkd.pth --wandb --gpu 0 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-mrkd.pth --wandb --gpu 0 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-mrkd.pth --wandb --gpu 0 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-mrkd.pth --wandb --gpu 0 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-highBn-mrkd.pth --wandb --gpu 0 --test_res 512 384

# mid / fedbn / mrkd
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-mrkd.pth --wandb --gpu 0 --test_res 64 48
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-mrkd.pth --wandb --gpu 0 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-mrkd.pth --wandb --gpu 0 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-mrkd.pth --wandb --gpu 0 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-mrkd.pth --wandb --gpu 0 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-mrkd.pth --wandb --gpu 0 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-midBn-mrkd.pth --wandb --gpu 0 --test_res 512 384

# low / fedbn / mrkd
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-mrkd.pth --wandb --gpu 0 --test_res 64 48
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-mrkd.pth --wandb --gpu 0 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-mrkd.pth --wandb --gpu 0 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-mrkd.pth --wandb --gpu 0 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-mrkd.pth --wandb --gpu 0 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-mrkd.pth --wandb --gpu 0 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg raf/configs/hpe/mpii_mpii_mpii/vit-small_256x192_192x144_128x96_sfl.yaml --pretrained output/fl_fedbn-4000-hml-lowBn-mrkd.pth --wandb --gpu 0 --test_res 512 384