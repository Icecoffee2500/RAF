#!/bin/bash
# CHECKPOINT="checkpoints/cl-segformer-pretrained/best.pth"
CHECKPOINT="checkpoints/fl-segformer-256_256_256-gpu1_federated/best.pth"
# CHECKPOINT="checkpoints/fl-segformer-512_512_256-gpu2_federated/best.pth"
# CHECKPOINT="checkpoints/fl-segformer-512_512_512-gpu3_federated/best.pth"

DATA_ROOT="/home/user_cau/taeheon_ws/RAF/data/cityscapes"

resolutions=(
    "128 256"
    "256 512" 
    "512 1024"
    "768 1536"
    "1024 2048"
)

for res in "${resolutions[@]}"; do
    echo "Testing at resolution: $res"
    python -m raf.segmentation.fl_experiments.test \
           --checkpoint $CHECKPOINT \
           --resolution $res \
           --data_root $DATA_ROOT
    echo "---"
done