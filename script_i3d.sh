#!/bin/bash

source activate pyRL

PHASE=$1
GPU_IDS=$2
OUTPUT=$3

LOG_DIR="./logs/i3d"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi

LOG="${LOG_DIR}/${PHASE}_i3d_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

case ${PHASE} in
  train)
    # run training
    CUDA_VISIBLE_DEVICES=$GPU_IDS python main_i3d.py \
        --phase train \
        --batch_size 12 \
        --input_shape 224 224 \
        --num_workers 4 \
        --epoch 50 \
        --learning_rate 0.0005 \
        --gpus $GPU_IDS \
        --train_all \
        --output ./output/$OUTPUT
    ;;
  test)
    # run testing
    CUDA_VISIBLE_DEVICES=$GPU_IDS python main_i3d.py \
        --phase test \
        --batch_size 1 \
        --input_shape 224 224 \
        --gpus $GPU_IDS \
        --model_weights output/$OUTPUT/checkpoints/i3d_accident_50.pth \
        --output ./output/$OUTPUT
    ;;
  *)
    echo "Invalid argument!"
    exit
    ;;
esac
echo "Done!"