#!/bin/bash


python finetune_convnext.py \
    --epochs 25 \
    --batch_size 2 \
    --lr 1e-4 \
    --weight_decay 5e-2 \
    --loss_coefficient 1 1 1 0.0 \
    --accumulation_steps 2 \
    --model_name "facebook/dinov3-convnext-base-pretrain-lvd1689m" \
    --hidden_dim 512 \
    --training_mode "freeze_backbone" \
    --resolution 1024 \
    --wandb_mode "online" \
    --data_folder "data/CSIRO" \
    --mode "single-fold" \
    #-stage2_start_epoch 10
