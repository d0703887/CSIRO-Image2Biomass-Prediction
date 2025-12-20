#!/bin/bash


python finetune.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --loss_coefficient 0.3 0.3 0.3 0.1 \
    --wandb_mode "online" \
    --model_name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --hidden_dim 128 \
    --freeze_backbone \
    --data_folder "data/CSIRO" \
    --mode "single-fold" \
    #--stage2_start_epoch 10
