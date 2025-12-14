#!/usr/bin/env bash
set -e

# 用法:
#   bash run_twobranch_large_extra.sh        # 默认 GPU 2
#   bash run_twobranch_large_extra.sh 4      # 指定 GPU 4

GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="/data1/nwang60/datasets/CV-Dataset"

# 让 .env 生效
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "警告: 未找到 .env, 将使用当前环境变量"
fi

export WANDB_PROJECT="FDU-2025CV-PJ"
export WANDB_MODE="${WANDB_MODE:-online}"

run() {
  local NAME="$1"
  shift
  export WANDB_NAME="$NAME"

  echo "=============================="
  echo "GPU:         $GPU_ID"
  echo "Run name:    $NAME"
  echo "Extra args:  $*"
  echo "=============================="

  python train.py \
    --data_dir "$DATA_DIR" \
    --save_dir "./weights/$NAME" \
    "$@"
}

########################################
# large 两塔: lr 扫描
# (gpu0 已有 arch_twobranch_large_rgb_grad, lr=1e-3,
#  所以这里只跑新的 lr)
########################################

# large 双塔, mixup only, lr=5e-4
run extra_twobranch_large_mixup_only_lr5e4 \
  --channel_mode rgb_grad \
  --model_size large \
  --two_branch \
  --use_mixup \
  --mixup_alpha 0.4 \
  --lr 5e-4 \
  --epochs 50 \
  --use_wandb

# large 双塔, mixup only, lr=3e-4
run extra_twobranch_large_mixup_only_lr3e4 \
  --channel_mode rgb_grad \
  --model_size large \
  --two_branch \
  --use_mixup \
  --mixup_alpha 0.4 \
  --lr 3e-4 \
  --epochs 50 \
  --use_wandb


