#!/bin/bash

source activate pyRL

PHASE=$1
GPU_IDS=0

LOG_DIR="./logs/reinforce"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi
rm -rf ${LOG_DIR}/${PHASE}_reinforce_*.log

LOG="${LOG_DIR}/${PHASE}_reinforce_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# run training
# CUDA_VISIBLE_DEVICES=$GPU_IDS python main_reinforce.py --binary_cls
CUDA_VISIBLE_DEVICES=$GPU_IDS python main_reinforce.py --binary_cls --alpha 0 --output ./output/REINFORCE_alpha0

echo "Done!"