#!/usr/bin/env bash
set -e

# 找到脚本所在目录并切过去
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 可按需改成自己的路径
DATA_DIR="/data1/nwang60/datasets/CV-Dataset"

# 让 .env 生效 (里可以放 WANDB_API_KEY 等)
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "警告: 未找到 .env, 将使用当前环境变量"
fi

# wandb 配置
export WANDB_PROJECT="FDU-2025CV-PJ"
# 如果有 entity 则在 .env 里设置 WANDB_ENTITY, train.py 会自动用
export WANDB_MODE="${WANDB_MODE:-online}"

run() {
  local NAME="$1"
  shift
  export WANDB_NAME="$NAME"

  echo "=============================="
  echo "Run name:      $NAME"
  echo "Extra args:    $*"
  echo "=============================="

  python train.py \
    --data_dir "$DATA_DIR" \
    --save_dir "./weights/$NAME" \
    "$@"
}

########################################
# Group A: 架构 + 通道消融
########################################

# A1 - small 单塔, RGB only baseline
run arch_small_rgb \
  --channel_mode rgb \
  --model_size small \
  --epochs 50 \
  --use_wandb

# A2 - small 单塔, RGB + Sobel (你目前最强 baseline)
run arch_small_rgb_grad \
  --channel_mode rgb_grad \
  --model_size small \
  --epochs 50 \
  --use_wandb

# A3 - small 单塔, RGB + FFT
run arch_small_rgb_fft \
  --channel_mode rgb_fft \
  --model_size small \
  --epochs 50 \
  --use_wandb

# A4 - small 单塔, RGB + FFT + Sobel (full 频谱)
run arch_small_rgb_fft_grad_full \
  --channel_mode rgb_fft_grad \
  --model_size small \
  --epochs 50 \
  --use_wandb

# A5 - small 单塔, RGB + FFT(highpass 0.1) + Sobel
run arch_small_rgb_fft_grad_hp01 \
  --channel_mode rgb_fft_grad \
  --model_size small \
  --fft_highpass_only \
  --fft_low_cut_ratio 0.1 \
  --epochs 50 \
  --use_wandb

# A6 - large 单塔, RGB + Sobel
run arch_large_rgb_grad \
  --channel_mode rgb_grad \
  --model_size large \
  --epochs 50 \
  --use_wandb

# A7 - large 单塔, RGB + FFT + Sobel (full 频谱)
run arch_large_rgb_fft_grad_full \
  --channel_mode rgb_fft_grad \
  --model_size large \
  --epochs 50 \
  --use_wandb

# A8 - large 单塔, RGB + FFT(highpass 0.1) + Sobel
run arch_large_rgb_fft_grad_hp01 \
  --channel_mode rgb_fft_grad \
  --model_size large \
  --fft_highpass_only \
  --fft_low_cut_ratio 0.1 \
  --epochs 50 \
  --use_wandb

########################################
# Group B: 双塔结构消融 (只看 rgb_grad)
########################################

# B1 - small 双塔, RGB + Sobel
run arch_twobranch_small_rgb_grad \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --epochs 50 \
  --use_wandb

# B2 - large 双塔, RGB + Sobel
run arch_twobranch_large_rgb_grad \
  --channel_mode rgb_grad \
  --model_size large \
  --two_branch \
  --epochs 50 \
  --use_wandb

########################################
# Group C: 正则化消融 - 在 best 架构上 (small 双塔 rgb_grad)
########################################

# C1 - best 架构 + 强增强
run reg_twobranch_small_strong \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_strong_aug \
  --epochs 50 \
  --use_wandb

# C2 - best 架构 + 强增强 + label smoothing
run reg_twobranch_small_strong_ls \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_strong_aug \
  --use_label_smoothing \
  --label_smoothing 0.1 \
  --epochs 50 \
  --use_wandb

# C3 - best 架构 + 强增强 + mixup
run reg_twobranch_small_strong_mixup \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_strong_aug \
  --use_mixup \
  --mixup_alpha 0.4 \
  --epochs 50 \
  --use_wandb

# C4 - best 架构 + 强增强 + label smoothing + mixup
run reg_twobranch_small_strong_ls_mixup \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_strong_aug \
  --use_label_smoothing \
  --label_smoothing 0.1 \
  --use_mixup \
  --mixup_alpha 0.4 \
  --epochs 50 \
  --use_wandb

########################################
# Group D: 正则化对单塔 baseline 的影响
########################################

# D1 - small 单塔 rgb_grad + label smoothing
run reg_single_small_rgb_grad_ls \
  --channel_mode rgb_grad \
  --model_size small \
  --use_label_smoothing \
  --label_smoothing 0.1 \
  --epochs 50 \
  --use_wandb

# D2 - small 单塔 rgb_grad + label smoothing + mixup
run reg_single_small_rgb_grad_ls_mixup \
  --channel_mode rgb_grad \
  --model_size small \
  --use_label_smoothing \
  --label_smoothing 0.1 \
  --use_mixup \
  --mixup_alpha 0.4 \
  --epochs 50 \
  --use_wandb

