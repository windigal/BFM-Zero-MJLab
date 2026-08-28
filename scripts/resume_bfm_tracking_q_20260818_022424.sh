#!/usr/bin/env bash
set -euo pipefail

cd /home/ghw/code/BFM-Zero-MJLab

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export WANDB_RUN_ID=huvg6skg
export WANDB_RESUME=must

exec .venv/bin/python -c '
import logging
import os
from pathlib import Path

import torchrunx
from humanoidverse.train import parse_args, run_train

args = parse_args()
run_dir = Path("runs/bfm_zero_seed/20260818_022424").resolve()

torchrunx_log_dir = run_dir / "torchrunx" / "resume_r050"
torchrunx_log_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCHRUNX_LOG_DIR"] = str(torchrunx_log_dir)

logging.basicConfig(level=logging.INFO)

torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=8,
    backend="nccl",
    worker_timeout=2 * 60 * 60,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
    + (
        "MUJOCO*",
        "UFO_CACHE_DIR",
        "BFMZERO_MJLAB_CACHE_DIR",
        "UV_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "TEMP",
        "TMP",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "WARP_CACHE_PATH",
    ),
).run(run_train, args, run_dir, True)
' \
  --agent fb \
  --gpu-ids all \
  --tracking-q \
  --disable-discriminator \
  --tracking-actor-grad-ratio 0.5 \
  --init-checkpoint runs/bfm_zero_seed/20260817_011205 \
  --data-path humanoidverse/data/seed_train_full10s_5k_clean.pkl \
  --num-envs 512 \
  --num-env-steps 192000000 \
  --checkpoint-every-steps 3200000 \
  --buffer-size 2560000 \
  --cartwheel-aux-safe \
  --use-wandb
