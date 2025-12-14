#!/bin/bash


python train.py \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --loss_coefficient 0.2 0.2 0.2 0.0 0.4 0.0 0.0 \
    --wandb_mode "online" \
    --model_name "facebook/dinov3-vits16-pretrain-lvd1689m" \
    --predict_total \
    --freeze_backbone \
    --hidden_dim 1024 \
    --data_folder "data" \
    --mode "cross-validation" \
    #--predict_gdm \
    #--predict_height \
    #--predict_has_clover \