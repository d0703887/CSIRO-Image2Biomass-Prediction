#!/bin/bash


python pretrain.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --stage2_start_epoch 5 \
    --model_name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --hidden_dim 128 \
    --wandb_mode "online" \
    --irish_data_folder "/content/pretrain_data/IrishGrassClover" \
    --grass_data_folder "/content/pretrain_data/GrassClover" \
    --csiro_data_folder "/content/data"