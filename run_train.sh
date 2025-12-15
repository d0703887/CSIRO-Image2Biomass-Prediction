#!/bin/bash


python train.py \
    --epochs 20 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 5e-2 \
    --loss_coefficient 0.33 0.33 0.34 0.0 0.0 0.0 0.0 \
    --wandb_mode "online" \
    --model_name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --freeze_backbone \
    --hidden_dim 64 \
    --data_folder "data" \
    --mode "single-fold" \
    #--predict_total \
    #--predict_gdm \
    #--predict_height \
    #--predict_has_clover \