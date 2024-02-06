#!/bin/bash

# vim, vim_pcls, vim_cls2
N_GPUS=8
MODEL=vim_cls2
INPUT_SIZE=256
PATCH_SIZE=16
EMBED_DIM=192
MODEL_NAME=${MODEL}_tiny_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual_with_cls_token
PRETRAINED=workbench/pretrained_ckpt/vim_tiny_73p1.pth
RESUME=workbench/${MODEL_NAME}/patch${PATCH_SIZE}_${INPUT_SIZE}_${EMBED_DIM}/train/checkpoint.pth
IMAGENET_PATH=datasets/imagenet/
OUTPUT_DIR=workbench/${MODEL_NAME}/patch${PATCH_SIZE}_${INPUT_SIZE}_${EMBED_DIM}/train/
python -m torch.distributed.launch --nproc_per_node=${N_GPUS} --use_env vim/main.py --batch-size 128 --num_workers 25 \
    --data-path ${IMAGENET_PATH} --output_dir ${OUTPUT_DIR} --no_amp --model ${MODEL_NAME} \
    --embed-dim ${EMBED_DIM} --input-size ${INPUT_SIZE} --patch-size ${PATCH_SIZE} --finetune ${PRETRAINED} \
    --sched cosine --warmup-epochs 0 --lr 5e-5 --hilbert \

# --hilbert
# --finetune ${PRETRAINED}
# --resume ${RESUME}
