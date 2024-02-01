#!/bin/bash

MODEL_NAME=vimhi_tiny_patch16_256_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token
CKPT_PATH=workbench/${MODEL_NAME}/train/checkpoint.pth
IMAGENET_PATH=datasets/imagenet/
OUTPUT_DIR=workbench/${MODEL_NAME}/eval/
python vim/main.py --eval --resume ${CKPT_PATH} --model ${MODEL_NAME} --batch-size 512 --input-size 256 \
    --data-path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR}
