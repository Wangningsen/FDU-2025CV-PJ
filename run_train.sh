#!/usr/bin/env bash
set -e

# 用法:
#   bash run_train.sh <run_name> [channel_mode] [fft_mode] [low_cut_ratio]
# 示例:
#   bash run_train.sh exp_rgb_fft_full   rgb_fft_grad full
#   bash run_train.sh exp_rgb_fft_hp10   rgb_fft_grad high 0.1
#   bash run_train.sh exp_rgb_only       rgb

RUN_NAME="$1"
CHANNEL_MODE="$2"
FFT_MODE="$3"
LOW_CUT_RATIO="$4"

if [ -z "$RUN_NAME" ]; then
  echo "用法: bash run_train.sh <run_name> [channel_mode] [fft_mode] [low_cut_ratio]"
  echo "示例: bash run_train.sh exp_rgb_fft_grad_run1 rgb_fft_grad high 0.1"
  exit 1
fi

if [ -z "$CHANNEL_MODE" ]; then
  CHANNEL_MODE="rgb_fft_grad"
fi

# fft_mode: full 或 high/highpass
if [ -z "$FFT_MODE" ]; then
  FFT_MODE="full"
fi

# 默认低频半径比例
if [ -z "$LOW_CUT_RATIO" ]; then
  LOW_CUT_RATIO="0.1"
fi

# 找到脚本所在目录并切过去
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 让 .env 生效
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "警告: 未找到 .env, 将使用当前环境变量"
fi

# 指定 wandb 的 project 和 run 名称
export WANDB_PROJECT="FDU-2025CV-PJ"
export WANDB_NAME="$RUN_NAME"
export WANDB_MODE="${WANDB_MODE:-online}"

# 根据 fft_mode 生成额外参数
FFT_FLAGS=""
if [ "$FFT_MODE" = "high" ] || [ "$FFT_MODE" = "highpass" ]; then
  FFT_FLAGS="--fft_highpass_only --fft_low_cut_ratio ${LOW_CUT_RATIO}"
fi

echo "=============================="
echo "启动训练"
echo "  WANDB_PROJECT   = $WANDB_PROJECT"
echo "  WANDB_NAME      = $WANDB_NAME"
echo "  WANDB_MODE      = $WANDB_MODE"
echo "  CHANNEL_MODE    = $CHANNEL_MODE"
echo "  FFT_MODE        = $FFT_MODE"
echo "  LOW_CUT_RATIO   = $LOW_CUT_RATIO"
echo "  EXTRA FFT FLAGS = $FFT_FLAGS"
echo "=============================="

python train.py \
  --data_dir /data1/nwang60/datasets/CV-Dataset \
  --save_dir ./weights/"$RUN_NAME" \
  --channel_mode "$CHANNEL_MODE" \
  $FFT_FLAGS \
  --use_wandb

