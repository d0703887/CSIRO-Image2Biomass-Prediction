#!/bin/bash


python pretrain.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4 \
    --weight_decay 1e-3 \
    --model_name "facebook/dinov3-vits16-pretrain-lvd1689m" \
    --hidden_dim 128 \
    --wandb_mode "online" \
    --irish_data_folder "data/IrishGrassClover" \
    --grass_data_folder "data/GrassClover" \
    --csiro_data_folder "data/CSIRO"