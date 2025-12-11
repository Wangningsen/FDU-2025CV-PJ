#!/usr/bin/env bash
set -e

# 用法:
#   bash run_twobranch_small_extra.sh        # 默认 GPU 1
#   bash run_twobranch_small_extra.sh 3      # 指定 GPU 3

GPU_ID="${1:-1}"
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
# small 两塔: 只改 lr (补充 B1)
########################################

# E1: small 两塔, lr=5e-4, 无强增强/正则
run extra_twobranch_small_lr5e4 \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --lr 5e-4 \
  --epochs 50 \
  --use_wandb

# E2: small 两塔, lr=3e-4, 无强增强/正则
run extra_twobranch_small_lr3e4 \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --lr 3e-4 \
  --epochs 50 \
  --use_wandb

########################################
# small 两塔: 单独开正则 (不和 C 组重复)
########################################

# E3: small 两塔 + label smoothing (无 strong_aug/mixup)
run extra_twobranch_small_ls_only \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_label_smoothing \
  --label_smoothing 0.1 \
  --epochs 50 \
  --use_wandb

# E4: small 两塔 + mixup (无 strong_aug/ls)
run extra_twobranch_small_mixup_only \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_mixup \
  --mixup_alpha 0.4 \
  --epochs 50 \
  --use_wandb

########################################
# small 两塔: 强增强组合, 但用更保守 lr
# (补 C4 的 "换 lr" 版本)
########################################

# E5: small 两塔 + strong_aug + ls + mixup, lr=5e-4
run extra_twobranch_small_strong_ls_mixup_lr5e4 \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_strong_aug \
  --use_label_smoothing \
  --label_smoothing 0.1 \
  --use_mixup \
  --mixup_alpha 0.4 \
  --lr 5e-4 \
  --epochs 50 \
  --use_wandb
