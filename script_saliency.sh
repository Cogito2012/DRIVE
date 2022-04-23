#!/bin/bash

source activate pyRL

PHASE=$1
GPU_IDS=0

LOG_DIR="./logs/saliency"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi
rm -rf ${LOG_DIR}/${PHASE}_saliency_*.log

LOG="${LOG_DIR}/${PHASE}_saliency_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

case ${PHASE} in
  train)
    rm -rf output/saliency/*
    # run training
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
    ;;
  test)
    # run testing
    CUDA_VISIBLE_DEVICES=$GPU_IDS python main_saliency.py \
        --phase test \
        --input_shape 480 640 \
        --gpus $GPU_IDS \
        --output ./output/saliency \
        --model_weights saliency_model_25.pth
    ;;
  evaluate)
    # evaluate the results
    python main_saliency.py \
        --phase eval \
        --output ./output/saliency
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
echo "Done!"