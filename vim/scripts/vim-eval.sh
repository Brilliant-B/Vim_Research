#!/bin/bash

MODEL_NAME=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token
CKPT_PATH=workbench/pretrained_ckpt/vim_tiny_73p1.pth
IMAGENET_PATH=datasets/imagenet/
OUTPUT_DIR=workbench/${MODEL_NAME}/eval/
python vim/main.py --eval --resume ${CKPT_PATH} --model ${MODEL_NAME} --batch-size 256 \
    --data-path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR}
