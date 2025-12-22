#!/bin/bash


python finetune.py \
    --epochs 25 \
    --batch_size 16 \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --loss_coefficient 0.34 0.33 0.33 0.0 \
    --model_name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --hidden_dim 128 \
    --freeze_backbone \
    --resolution 768 \
    --wandb_mode "online" \
    --data_folder "data/CSIRO" \
    --mode "single-fold" \
    #--stage2_start_epoch 10
