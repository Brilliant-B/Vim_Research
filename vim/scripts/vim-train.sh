#!/bin/bash

N_GPUS=8
MODEL_NAME=vimhi_tiny_patch16_256_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token
RESUME=workbench/${MODEL_NAME}/train/checkpoint_tfs.pth
IMAGENET_PATH=datasets/imagenet/
OUTPUT_DIR=workbench/${MODEL_NAME}/train/
python -m torch.distributed.launch --nproc_per_node=${N_GPUS} \
    --use_env vim/main.py --model ${MODEL_NAME} --batch-size 128 --num_workers 25 --input-size 256 \
    --data-path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR} --no_amp \
    --sched cosine --warmup-epochs 0 --lr 5e-4 \

# --start_epoch 10 --resume ${RESUME}
