#!/bin/bash


python finetune_vit.py \
    --epochs 25 \
    --batch_size 2 \
    --lr 1e-4 \
    --weight_decay 5e-2 \
    --loss_coefficient 1 1 1 1 \
    --accumulation_steps 2 \
    --model_name "facebook/dinov3-vits16plus-pretrain-lvd1689m" \
    --hidden_dim 512 \
    --training_mode "full_finetune" \
    --input_h 768 \
    --input_w 1536 \
    --wandb_mode "online" \
    --data_folder "data/CSIRO" \
    --mode "single-fold" \
    --stage2_start_epoch 5 \
    #--predict_height
