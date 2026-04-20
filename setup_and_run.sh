#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${ENV_NAME:-rtai-a2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
EXPECTED_TORCH_CUDA="${EXPECTED_TORCH_CUDA:-11.8}"
DEVICE="${DEVICE:-auto}"
DEEPXPLORE_DIR="${DEEPXPLORE_DIR:-../deepxplore}"
DATA_DIR="${DATA_DIR:-../../assignment1/data}"
EPOCHS="${EPOCHS:-5}"
TRAIN_LIMIT="${TRAIN_LIMIT:-10000}"
TEST_LIMIT="${TEST_LIMIT:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_SEEDS="${TRAIN_SEEDS:-1 2}"
RUN_SEEDS="${RUN_SEEDS:-30}"
ITERATIONS="${ITERATIONS:-20}"
MAX_VISUALIZATIONS="${MAX_VISUALIZATIONS:-5}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-0.75}"
DOWNLOAD="${DOWNLOAD:-auto}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_TEST="${SKIP_TEST:-0}"

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ROOT_DIR/$1" ;;
  esac
}

DATA_PATH="$(absolute_path "$DATA_DIR")"
DEEPXPLORE_PATH="$(absolute_path "$DEEPXPLORE_DIR")"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command was not found" >&2
  exit 1
fi

env_exists=0
if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$ENV_NAME"; then
  env_exists=1
fi

if [[ "$env_exists" == "0" ]]; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

CONDA_RUN=(conda run --no-capture-output -n "$ENV_NAME")

if [[ "$SKIP_INSTALL" != "1" ]]; then
  "${CONDA_RUN[@]}" python -m pip install --upgrade pip setuptools wheel
  "${CONDA_RUN[@]}" python -m pip install -r requirements.txt
fi

"${CONDA_RUN[@]}" python - "$EXPECTED_TORCH_CUDA" <<'PY'
import sys
import torch

expected_cuda = sys.argv[1]
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.version.cuda != expected_cuda:
    raise SystemExit(f"expected torch CUDA {expected_cuda}, got {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

download_args=()
if [[ "$DOWNLOAD" == "1" || "$DOWNLOAD" == "true" ]]; then
  download_args+=(--download)
elif [[ "$DOWNLOAD" == "auto" && ! -d "$DATA_PATH/cifar-10-batches-py" ]]; then
  mkdir -p "$DATA_PATH"
  download_args+=(--download)
fi

read -r -a TRAIN_SEED_ARGS <<< "$TRAIN_SEEDS"
need_train=0
for seed in "${TRAIN_SEED_ARGS[@]}"; do
  if [[ ! -f "$ROOT_DIR/models/resnet50_cifar10_seed${seed}.pt" ]]; then
    need_train=1
  fi
done

if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ "$FORCE_TRAIN" == "1" || "$FORCE_TRAIN" == "true" || "$need_train" == "1" ]]; then
    "${CONDA_RUN[@]}" python train_models.py \
      --data-dir "$DATA_DIR" \
      --epochs "$EPOCHS" \
      --train-limit "$TRAIN_LIMIT" \
      --test-limit "$TEST_LIMIT" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --seeds "${TRAIN_SEED_ARGS[@]}" \
      --device "$DEVICE" \
      "${download_args[@]}"
  else
    echo "existing model checkpoints found; skipping training"
  fi
fi

if [[ "$SKIP_TEST" != "1" ]]; then
  if [[ ! -d "$DEEPXPLORE_PATH" ]]; then
    echo "DeepXplore directory not found: $DEEPXPLORE_PATH" >&2
    exit 1
  fi
  "${CONDA_RUN[@]}" python test.py \
    --deepxplore-dir "$DEEPXPLORE_DIR" \
    --data-dir "$DATA_DIR" \
    --seeds "$RUN_SEEDS" \
    --iterations "$ITERATIONS" \
    --coverage-threshold "$COVERAGE_THRESHOLD" \
    --max-visualizations "$MAX_VISUALIZATIONS" \
    --device "$DEVICE" \
    "${download_args[@]}"
fi

echo "done"
