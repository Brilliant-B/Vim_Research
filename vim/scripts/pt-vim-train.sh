#!/bin/bash

N_GPUS=8
MODEL_NAME=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token
PRETRAINED=workbench/pretrained_ckpt/vim_tiny_73p1.pth
IMAGENET_PATH=datasets/imagenet/
OUTPUT_DIR=workbench/${MODEL_NAME}/train/
python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
    --use_env vim/main.py --model ${MODEL_NAME} --batch-size 128 --num_workers 25 \
    --data-path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR} --finetune ${PRETRAINED} --no_amp
