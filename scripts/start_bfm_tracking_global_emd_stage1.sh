#!/usr/bin/env bash
set -euo pipefail

cd /home/ghw/code/BFM-Zero-MJLab

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0

# Start a fresh non-PACE Stage-II run from the final Stage-I model weights.
# The active single-dataset sampler uses the traditional global EMD weighting:
#   p_i proportional to 2 ** (2 * clamp(EMD_i, 0.5, 2.0)).
exec ./run_train.sh \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --work-dir runs/bfm_zero_seed \
  --agent fb \
  --tracking-q \
  --tracking-actor-grad-ratio 0.30 \
  --init-checkpoint runs/bfm_zero_seed/20260804_040010 \
  --data-path humanoidverse/data/seed_train_full10s_5k.pkl \
  --num-envs 512 \
  --buffer-size 2560000 \
  --cartwheel-aux-safe \
  --use-wandb
