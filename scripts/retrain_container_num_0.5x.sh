#!/usr/bin/env bash
# Fine-tune container_num 0.5x model after DFL regression target fix.
#
# Loads the previous best weights (container_num_0.5x/model_best_fp16.pth)
# as the starting point, then fine-tunes for 100 epochs with the corrected
# DFL target clamp (max = reg_max - 0.01 = 6.99 instead of the buggy 5.99).
#
# Config (matching the original run):
#   Model:    FlashDet m-0.5x (backbone=0.5x, fpn=96)
#   Input:    320x320
#   Classes:  1 (container_number)
#   Dataset:  data/container_num/{train,valid}
#   Epochs:   100
#   Finetune: container_num_0.5x/model_best_fp16.pth
#
# Usage:
#   bash scripts/retrain_container_num_0.5x.sh
#   bash scripts/retrain_container_num_0.5x.sh --amp          # with mixed precision
#   bash scripts/retrain_container_num_0.5x.sh --epochs 150   # override epochs

set -euo pipefail
cd "$(dirname "$0")/.."

SAVE_DIR="container_num_0.5x_fixed"
FINETUNE_FROM="container_num_0.5x/model_best_fp16.pth"
EPOCHS=100
BATCH_SIZE=32
LR=0.001
INPUT_SIZE=320
MODEL_SIZE="m-0.5x"
CLASS_FILE="classes/container_num.txt"
TRAIN_IMAGES="data/container_num/train"
VAL_IMAGES="data/container_num/valid"

python train.py \
    --model-size "$MODEL_SIZE" \
    --input-size "$INPUT_SIZE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --save-dir "$SAVE_DIR" \
    --finetune "$FINETUNE_FROM" \
    --class-file "$CLASS_FILE" \
    --train-images "$TRAIN_IMAGES" \
    --val-images "$VAL_IMAGES" \
    "$@"
