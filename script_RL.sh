#!/bin/bash

source activate pyRL

PHASE=$1
RL=$2
GPU_IDS=0

LOG_DIR="./logs/${RL}"
if [ ! -d $LOG_DIR ]; then
  mkdir -p -m 777 $LOG_DIR
  echo "mkdir -p -m 777 ${LOG_DIR} done"
fi
# rm -rf ${LOG_DIR}/${PHASE}_${RL}_*.log

LOG="${LOG_DIR}/${PHASE}_${RL}_`date +'%Y-%m-%d_%H-%M'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# run training
# CUDA_VISIBLE_DEVICES=$GPU_IDS python main_reinforce.py --binary_cls
# CUDA_VISIBLE_DEVICES=$GPU_IDS python main_reinforce.py --binary_cls --alpha 0 --output ./output/REINFORCE_alpha0

CUDA_VISIBLE_DEVICES=$GPU_IDS python main_sac.py --output ./output/SAC_LSTM --phase ${PHASE}
# CUDA_VISIBLE_DEVICES=$GPU_IDS python main_sac.py --output ./output/SAC_deterministic --config cfgs/sac_deterministic.yml

echo "Done!"