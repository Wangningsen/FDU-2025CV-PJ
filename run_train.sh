#!/usr/bin/env bash
set -e

# 用法:
#   bash run_train.sh <run_name> [channel_mode]
# 示例:
#   bash run_train.sh rgb_fft_grad_run1 rgb_fft_grad
#   bash run_train.sh rgb_only_run1 rgb

RUN_NAME="$1"
CHANNEL_MODE="$2"

if [ -z "$RUN_NAME" ]; then
  echo "用法: bash run_train.sh <run_name> [channel_mode]"
  echo "示例: bash run_train.sh rgb_fft_grad_run1 rgb_fft_grad"
  exit 1
fi

if [ -z "$CHANNEL_MODE" ]; then
  CHANNEL_MODE="rgb_fft_grad"
fi

# 找到脚本所在目录並切过去
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 让 .env 生效
# .env 内容建议至少有:
#   WANDB_API_KEY=...
#   WANDB_ENTITY=...
#   WANDB_MODE=online
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
# 如果 .env 里没写 WANDB_MODE 就默认 online
export WANDB_MODE="${WANDB_MODE:-online}"

echo "=============================="
echo "启动训练"
echo "  WANDB_PROJECT = $WANDB_PROJECT"
echo "  WANDB_NAME    = $WANDB_NAME"
echo "  WANDB_MODE    = $WANDB_MODE"
echo "  CHANNEL_MODE  = $CHANNEL_MODE"
echo "=============================="

# 这里假设:
#   数据在 ./dataset
#   权重保存到 ./weights/<run_name> 里
python train.py \
  --data_dir /data1/nwang60/datasets/CV-Dataset \
  --save_dir ./weights/"$RUN_NAME" \
  --channel_mode "$CHANNEL_MODE" \
  --use_wandb

