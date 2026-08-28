#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${UFO_CACHE_DIR:-}" ]]; then
  work_dir=""
  args=("$@")
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --work-dir)
        if ((i + 1 < ${#args[@]})); then
          work_dir="${args[$((i + 1))]}"
        fi
        ;;
      --work-dir=*)
        work_dir="${args[$i]#--work-dir=}"
        ;;
    esac
  done

  if [[ -n "$work_dir" ]]; then
    work_parent="$(dirname "$work_dir")"
    if [[ "$(basename "$work_parent")" == "runs" ]]; then
      export UFO_CACHE_DIR="$(dirname "$work_parent")/cache"
    else
      export UFO_CACHE_DIR="$work_parent/cache"
    fi
  else
    export UFO_CACHE_DIR="$script_dir/cache"
  fi
fi

canonical_dir() {
  mkdir -p "$1"
  (cd "$1" && pwd -P)
}

# TorchInductor invokes the compiler from a temporary working directory. Its
# generated C++ paths must therefore be absolute; relative cache paths such as
# ./cache/torchinductor make g++ look below that temporary directory.
UFO_CACHE_DIR="$(canonical_dir "$UFO_CACHE_DIR")"
export UFO_CACHE_DIR

mkdir -p \
  "$UFO_CACHE_DIR/uv" \
  "$UFO_CACHE_DIR/pycache" \
  "$UFO_CACHE_DIR/tmp" \
  "$UFO_CACHE_DIR/torchinductor" \
  "$UFO_CACHE_DIR/triton" \
  "$UFO_CACHE_DIR/cuda" \
  "$UFO_CACHE_DIR/warp"

export BFMZERO_MJLAB_CACHE_DIR="$(canonical_dir "${BFMZERO_MJLAB_CACHE_DIR:-$UFO_CACHE_DIR}")"
export UV_CACHE_DIR="$(canonical_dir "${UV_CACHE_DIR:-$UFO_CACHE_DIR/uv}")"
export PYTHONPYCACHEPREFIX="$(canonical_dir "${PYTHONPYCACHEPREFIX:-$UFO_CACHE_DIR/pycache}")"
export TMPDIR="$(canonical_dir "${TMPDIR:-$UFO_CACHE_DIR/tmp}")"
export TEMP="$(canonical_dir "${TEMP:-$TMPDIR}")"
export TMP="$(canonical_dir "${TMP:-$TMPDIR}")"
export TORCHINDUCTOR_CACHE_DIR="$(canonical_dir "${TORCHINDUCTOR_CACHE_DIR:-$UFO_CACHE_DIR/torchinductor}")"
export TRITON_CACHE_DIR="$(canonical_dir "${TRITON_CACHE_DIR:-$UFO_CACHE_DIR/triton}")"
export CUDA_CACHE_PATH="$(canonical_dir "${CUDA_CACHE_PATH:-$UFO_CACHE_DIR/cuda}")"
export WARP_CACHE_PATH="$(canonical_dir "${WARP_CACHE_PATH:-$UFO_CACHE_DIR/warp}")"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

exec uv run python -m humanoidverse.train "$@"
