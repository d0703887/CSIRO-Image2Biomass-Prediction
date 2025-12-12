#!/bin/bash


python train.py \
    --epochs 30 \
    --batch_size 64 \
    --input_H 768 \
    --input_W 768 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --wandb_mode "online" \
    --model_name "facebook/dinov3-vits16-pretrain-lvd1689m" \
    --loss_coefficient 0.15 0.15 0.15 0.05 0.3 0.1 0.1 \
    --predict_total \
    --predict_height \
    --predict_has_clover \
    --freeze_backbone \
    --hidden_dim 1024 \
    --data_folder "data" \
    #--predict_gdm \