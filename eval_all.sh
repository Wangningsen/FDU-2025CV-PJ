#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="/data1/nwang60/datasets/CV-Dataset"

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
else
  echo "Warning: .env not found, using current environment"
fi

eval_run() {
  local NAME="$1"
  local CHANNEL_MODE="$2"
  local MODEL_SIZE="$3"
  shift 3
  # "$@" contains extra flags, such as:
  #   --two_branch
  #   --fft_highpass_only
  #   --fft_low_cut_ratio <value>
  #   --use_tta
  local CKPT="./weights/${NAME}/best_model.pth"

  echo "=============================="
  echo "Evaluating:       $NAME"
  echo "Checkpoint:       $CKPT"
  echo "Channel mode:     $CHANNEL_MODE"
  echo "Model size:       $MODEL_SIZE"
  echo "Extra eval args:  $*"
  echo "=============================="

  python eval_val.py \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CKPT" \
    --channel_mode "$CHANNEL_MODE" \
    --model_size "$MODEL_SIZE" \
    --img_size 256 \
    --batch_size 64 \
    --val_ratio 0.1 \
    --seed 42 \
    "$@"
}

########################################
# Group A: architecture + channel ablations
########################################

eval_run arch_small_rgb \
  rgb \
  small

eval_run arch_small_rgb_grad \
  rgb_grad \
  small

eval_run arch_small_rgb_fft \
  rgb_fft \
  small

eval_run arch_small_rgb_fft_grad_full \
  rgb_fft_grad \
  small

eval_run arch_small_rgb_fft_grad_hp01 \
  rgb_fft_grad \
  small \
  --fft_highpass_only \
  --fft_low_cut_ratio 0.1

eval_run arch_large_rgb_grad \
  rgb_grad \
  large

eval_run arch_large_rgb_fft_grad_full \
  rgb_fft_grad \
  large

eval_run arch_large_rgb_fft_grad_hp01 \
  rgb_fft_grad \
  large \
  --fft_highpass_only \
  --fft_low_cut_ratio 0.1

########################################
# Group B: two-branch ablations
########################################

eval_run arch_twobranch_small_rgb_grad \
  rgb_grad \
  small \
  --two_branch

eval_run arch_twobranch_large_rgb_grad \
  rgb_grad \
  large \
  --two_branch

########################################
# Group C: regularization on best two-branch (small rgb_grad)
########################################

eval_run reg_twobranch_small_strong \
  rgb_grad \
  small \
  --two_branch

eval_run reg_twobranch_small_strong_ls \
  rgb_grad \
  small \
  --two_branch

eval_run reg_twobranch_small_strong_mixup \
  rgb_grad \
  small \
  --two_branch

eval_run reg_twobranch_small_strong_ls_mixup \
  rgb_grad \
  small \
  --two_branch


########################################
# Group D: regularization on single-branch baseline
########################################

eval_run reg_single_small_rgb_grad_ls \
  rgb_grad \
  small

eval_run reg_single_small_rgb_grad_ls_mixup \
  rgb_grad \
  small

########################################
# Group E: large two-branch extras (lr + regularization sweeps)
########################################

eval_run extra_twobranch_large_lr5e4 \
  rgb_grad \
  large \
  --two_branch

eval_run extra_twobranch_large_lr3e4 \
  rgb_grad \
  large \
  --two_branch

eval_run extra_twobranch_large_ls_only \
  rgb_grad \
  large \
  --two_branch

eval_run extra_twobranch_large_strong_ls_lr5e4 \
  rgb_grad \
  large \
  --two_branch

eval_run extra_twobranch_large_strong_ls_mixup_lr5e4 \
  rgb_grad \
  large \
  --two_branch

########################################
# Group F: small two-branch extras (lr + regularization sweeps)
########################################

eval_run extra_twobranch_small_lr5e4 \
  rgb_grad \
  small \
  --two_branch

eval_run extra_twobranch_small_lr3e4 \
  rgb_grad \
  small \
  --two_branch

eval_run extra_twobranch_small_ls_only \
  rgb_grad \
  small \
  --two_branch

eval_run extra_twobranch_small_mixup_only \
  rgb_grad \
  small \
  --two_branch

eval_run extra_twobranch_small_strong_ls_mixup_lr5e4 \
  rgb_grad \
  small \
  --two_branch
