#!/bin/bash

MODEL=vim # vim_pcls, vim_cls2
INPUT_SIZE=256
PATCH_SIZE=16
MODEL_NAME=${MODEL}_tiny_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token
CKPT_PATH=workbench/${MODEL_NAME}/patch${PATCH_SIZE}_${INPUT_SIZE}/train/checkpoint.pth
IMAGENET_PATH=datasets/imagenet/
OUTPUT_DIR=workbench/${MODEL_NAME}/patch${PATCH_SIZE}_${INPUT_SIZE}/eval/
python vim/main.py --eval --resume ${CKPT_PATH} --batch-size 64 --data-path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR} \
    --model ${MODEL_NAME} --input-size ${INPUT_SIZE} --patch-size ${PATCH_SIZE} --hilbert \
