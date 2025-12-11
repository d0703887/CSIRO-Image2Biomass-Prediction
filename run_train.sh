#!/bin/bash


python train.py \
    --epochs 30 \
    --batch_size 64 \
    --input_H 768 \
    --input_W 768 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --wandb_mode "online" \
        --loss_coefficient 0.08 0.08 0.08 0.16 0.4 0.1 0.1 \
    --predict_total \
    #--predict_gdm \
    --predict_height \
    --predict_has_clover \
    --freeze_backbone \
        --hidden_dim 1024 \
        --data_path "data\train.csv" \