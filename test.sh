#!/bin/bash

python3 raf/experiments/test_hpe.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_fl_test.yaml --pretrained output/fl_base-fulldata-hml-only_mr_mr_aug.pth --wandb --gpu 0 --test_res 64 48
python3 raf/experiments/test_hpe.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_fl_test.yaml --pretrained output/fl_base-fulldata-hml-only_mr_mr_aug.pth --wandb --gpu 0 --test_res 128 96
python3 raf/experiments/test_hpe.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_fl_test.yaml --pretrained output/fl_base-fulldata-hml-only_mr_mr_aug.pth --wandb --gpu 0 --test_res 192 144
python3 raf/experiments/test_hpe.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_fl_test.yaml --pretrained output/fl_base-fulldata-hml-only_mr_mr_aug.pth --wandb --gpu 0 --test_res 256 192
python3 raf/experiments/test_hpe.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_fl_test.yaml --pretrained output/fl_base-fulldata-hml-only_mr_mr_aug.pth --wandb --gpu 0 --test_res 320 240
python3 raf/experiments/test_hpe.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_fl_test.yaml --pretrained output/fl_base-fulldata-hml-only_mr_mr_aug.pth --wandb --gpu 0 --test_res 384 288
python3 raf/experiments/test_hpe.py --cfg experiments/mpii_mpii_mpii/vit-small_256x192_fl_test.yaml --pretrained output/fl_base-fulldata-hml-only_mr_mr_aug.pth --wandb --gpu 0 --test_res 512 384