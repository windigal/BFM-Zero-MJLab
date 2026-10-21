#!/usr/bin/env bash
set -euo pipefail
export UFO_TRAIN_MODULE=humanoidverse.ppo_fb.train
exec "$(dirname "${BASH_SOURCE[0]}")/run_train.sh" "$@"
