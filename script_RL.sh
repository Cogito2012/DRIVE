#!/bin/bash

source activate pyRL

PHASE=$1
EXP_TAG=$2
GPU_IDS=$3

LOG_DIR="./logs/${EXP_TAG}"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi
# rm -rf ${LOG_DIR}/${PHASE}_${EXP_TAG}_*.log

LOG="${LOG_DIR}/${PHASE}_${EXP_TAG}_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# run training
# CUDA_VISIBLE_DEVICES=$GPU_IDS python main_reinforce.py --binary_cls
# CUDA_VISIBLE_DEVICES=$GPU_IDS python main_reinforce.py --binary_cls --alpha 0 --output ./output/REINFORCE_alpha0

CUDA_VISIBLE_DEVICES=$GPU_IDS python main_sac.py --output ./output/${EXP_TAG} --phase ${PHASE} --config cfgs/sac_default.yml --test_epoch 40
# CUDA_VISIBLE_DEVICES=$GPU_IDS python main_sac.py --output ./output/${EXP_TAG}_det --config cfgs/sac_deterministic.yml

echo "Done!"