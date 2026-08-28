#!/usr/bin/env bash
set -euo pipefail

cd /home/ghw/code/BFM-Zero-MJLab

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0

exec ./run_train.sh \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir runs/bfm_zero_seed \
  --agent fb \
  --pace-base-checkpoint runs/bfm_zero_seed/20260804_040010 \
  --tracking-q \
  --tracking-actor-grad-ratio 0.30 \
  --data-path humanoidverse/data/seed_train_full10s_5k.pkl \
  --num-envs 512 \
  --buffer-size 2560000 \
  --cartwheel-aux-safe \
  --use-wandb
