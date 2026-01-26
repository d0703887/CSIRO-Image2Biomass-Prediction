#!/bin/bash


python finetune_vit.py \
    --epochs 20 \
    --batch_size 2 \
    --lr 5e-4 \
    --weight_decay 5e-2 \
    --loss_coefficient 1 1 1 0.2 \
    --accumulation_steps 2 \
    --model_name "facebook/dinov3-vitb16-pretrain-lvd1689m" \
    --hidden_dim 128 \
    --training_mode "freeze_backbone" \
    --input_h 768 \
    --input_w 1536 \
    --wandb_mode "online" \
    --data_folder "data/CSIRO" \
    --mode "single-fold" \
    --stage2_start_epoch 5 \