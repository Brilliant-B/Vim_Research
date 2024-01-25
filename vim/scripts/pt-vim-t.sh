#!/bin/bash
conda activate vim
cd vim;

N_GPUS=3
MODEL_NAME=vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token
IMAGENET_PATH=None
OUTPUT_DIR=./output/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
 --use_env main.py --model ${MODEL_NAME} --batch-size 128 --num_workers 25 \
 --data-path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR} --no_amp
