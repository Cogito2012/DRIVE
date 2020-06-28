#!/bin/bash

source activate pyRL

LOG_DIR="./logs/saliency"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi
rm -rf ${LOG_DIR}/train_saliency_*.log
rm -rf output/saliency/*

LOG="${LOG_DIR}/train_saliency_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

GPU_IDS=0

CUDA_VISIBLE_DEVICES=$GPU_IDS python main_saliency.py \
    --phase train \
    --batch_size 1 \
    --frame_interval 5 \
    --max_frames 32 \
    --input_shape 480 640 \
    --epoch 40 \
    --gpus $GPU_IDS \
    --num_workers 4 \
    --output ./output/saliency

echo "Done!"