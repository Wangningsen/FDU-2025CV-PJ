# General code style rules

- Don't use typing.
- Write as concise code as possible.
- Don't be overly defensive, avoid using try/except too much.
- Declare variables close to where they are used, and avoid creating variables that are used only once.
- Avoid defining new functions for very small snippets, or functions used only once.
# Spec: Generate `eval_all.sh` for CV-PJ experiments

## Context

The repository has:

- `train.py`: training entry script
- `eval_val.py`: evaluation script that:
  - Recreates the validation split from the training set using `--val_ratio` and `--seed`
  - Loads a checkpoint from `--checkpoint`
  - Computes binary classification metrics (accuracy, precision, recall, F1, etc.)
  - Writes metrics to a CSV file whose name is derived from the checkpoint directory name

There are multiple training launcher scripts:

- `run_experiments.sh`
- `run_twobranch_large_extra.sh`
- `run_twobranch_small_extra.sh`

Each of these scripts defines a helper function:

```bash
run() {
  local NAME="$1"
  shift
  export WANDB_NAME="$NAME"

  # ...
  python train.py \
    --data_dir "$DATA_DIR" \
    --save_dir "./weights/$NAME" \
    "$@"
}
````

Then they call this `run` function multiple times, for example:

```bash
# Example from run_experiments.sh

# A2 - small single-branch, RGB + Sobel
run arch_small_rgb_grad \
  --channel_mode rgb_grad \
  --model_size small \
  --epochs 50 \
  --use_wandb
```

This means:

* The experiment name is `arch_small_rgb_grad`
* The checkpoint is stored at `./weights/arch_small_rgb_grad/best_model.pth`
* The key training configuration flags for this experiment are:

  * `--channel_mode rgb_grad`
  * `--model_size small`
  * Possibly `--two_branch` (if present)
  * Possibly `--fft_highpass_only` and `--fft_low_cut_ratio <value>` (if present)
  * `--use_strong_aug`, `--use_label_smoothing`, `--use_mixup` are training only and do not need to be passed into `eval_val.py`

The validation split logic in `train.py` and `eval_val.py` is:

* Both use `AIGCDataset` over `train/`
* Both use:

  * `--val_ratio` (default 0.1)
  * `--seed` (default 42)
* `random_split` with `torch.Generator().manual_seed(seed)` is used to derive train and val indices

Therefore, using **the same seed and val_ratio** in `eval_val.py` will reconstruct the same validation split as used in training.

## Goal

Write a Bash script named `eval_all.sh` that:

1. Lives at the repository root (same directory as `train.py`, `eval_val.py`, `run_experiments.sh`, etc.).

2. Sequentially runs `eval_val.py` on the validation split for **all experiments** defined in:

   * `run_experiments.sh`
   * `run_twobranch_large_extra.sh`
   * `run_twobranch_small_extra.sh`

3. For each experiment, it:

   * Uses the same logical configuration as in training
     (same `--channel_mode`, `--model_size`, `--two_branch`, FFT flags, etc.).
   * Points `--checkpoint` to `./weights/<EXPERIMENT_NAME>/best_model.pth`.
   * Uses consistent global settings:

     * `--data_dir` identical to the one in the training scripts (`$DATA_DIR`).
     * `--img_size 256` (same as training default).
     * `--batch_size 64` for evaluation (you can choose this constant).
     * `--val_ratio 0.1` (same as training default).
     * `--seed 42` (same as training default).
     * Optionally `--use_tta` to enable test-time augmentation (horizontal flip).
   * Lets `eval_val.py` write metrics into a CSV file:

     * If `--output_csv` is **not** specified, `eval_val.py` will automatically create:

       * `./weights/<EXPERIMENT_NAME>/<EXPERIMENT_NAME>.csv`

4. Prints a short log before each evaluation command so the user can see which experiment is being evaluated.

## Evaluation script requirements

Create a file called `eval_all.sh` with the following properties:

* Shebang:

  ```bash
  #!/usr/bin/env bash
  ```

* Early exit on error:

  ```bash
  set -e
  ```

* Determine its own directory and switch into it:

  ```bash
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR"
  ```

* Reuse the same `DATA_DIR` convention as `run_experiments.sh`, e.g.:

  ```bash
  DATA_DIR="/data1/nwang60/datasets/CV-Dataset"
  ```

* Optionally load `.env` if present (same pattern as `run_experiments.sh`):

  ```bash
  if [ -f ".env" ]; then
    set -a
    source .env
    set +a
  else
    echo "Warning: .env not found, using current environment"
  fi
  ```

* Define a helper function `eval_run` that takes:

  * The **experiment name** (`NAME`)
  * The **channel_mode**
  * The **model_size**
  * Optional flags (`two_branch`, `fft_highpass_only`, `fft_low_cut_ratio`, `use_tta`)

Example interface:

```bash
eval_run() {
  local NAME="$1"
  local CHANNEL_MODE="$2"
  local MODEL_SIZE="$3"
  shift 3
  # "$@" contains extra flags: --two_branch, --fft_highpass_only, --fft_low_cut_ratio ...
  local CKPT="./weights/${NAME}/best_model.pth"

  echo "=============================="
  echo "Evaluating:   $NAME"
  echo "Checkpoint:   $CKPT"
  echo "Channel mode: $CHANNEL_MODE"
  echo "Model size:   $MODEL_SIZE"
  echo "Extra eval args: $*"
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
```

Note:

* `eval_val.py` accepts:

  * `--channel_mode`
  * `--model_size`
  * `--two_branch` (flag)
  * `--fft_highpass_only` (flag)
  * `--fft_low_cut_ratio <float>`
  * `--use_tta` (flag)
  * plus `--data_dir`, `--checkpoint`, `--img_size`, `--batch_size`, `--val_ratio`, `--seed`

* You do **not** need to pass training-only options such as `--epochs`, `--use_strong_aug`, `--use_label_smoothing`, `--use_mixup`, `--mixup_alpha`, or `--use_wandb` into `eval_val.py`.

## Mapping from training scripts to evaluation calls

For **each** `run ...` call in the three training scripts:

1. Extract:

   * `NAME` (the first argument to `run`)
   * `--channel_mode` value (if present; otherwise default is `"rgb_fft_grad"`)
   * `--model_size` value (`small` or `large`)
   * `--two_branch` flag (if present)
   * `--fft_highpass_only` flag (if present)
   * `--fft_low_cut_ratio <value>` (if present)

2. In `eval_all.sh`, add a matching `eval_run` call with:

   * The same `NAME`
   * The same `channel_mode` and `model_size`
   * The same optional flags that affect the feature construction:

     * Add `--two_branch` if the training command had `--two_branch`
     * Add `--fft_highpass_only` and `--fft_low_cut_ratio <same value>` if the training command had them
   * Optionally add `--use_tta` if you want test-time augmentation for all experiments

### Example mapping

Given this training command in `run_experiments.sh`:

```bash
run arch_large_rgb_fft_grad_hp01 \
  --channel_mode rgb_fft_grad \
  --model_size large \
  --fft_highpass_only \
  --fft_low_cut_ratio 0.1 \
  --epochs 50 \
  --use_wandb
```

You should add this evaluation line inside `eval_all.sh`:

```bash
eval_run arch_large_rgb_fft_grad_hp01 \
  rgb_fft_grad \
  large \
  --fft_highpass_only \
  --fft_low_cut_ratio 0.1 \
  --use_tta
```

Another example, for a two-branch model in `run_experiments.sh`:

```bash
run arch_twobranch_small_rgb_grad \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --epochs 50 \
  --use_wandb
```

The corresponding evaluation call in `eval_all.sh`:

```bash
eval_run arch_twobranch_small_rgb_grad \
  rgb_grad \
  small \
  --two_branch \
  --use_tta
```

For experiments without `fft_highpass_only` or `two_branch`, simply omit those flags in the `eval_run` call.

## Final structure of `eval_all.sh`

The final script should look like this (in outline):

```bash
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
  # ... definition as above ...
}

########################################
# Group A: architecture + channel ablations
########################################

eval_run arch_small_rgb rgb small
eval_run arch_small_rgb_grad rgb_grad small
eval_run arch_small_rgb_fft rgb_fft small
eval_run arch_small_rgb_fft_grad_full rgb_fft_grad small
eval_run arch_small_rgb_fft_grad_hp01 rgb_fft_grad small --fft_highpass_only --fft_low_cut_ratio 0.1
eval_run arch_large_rgb_grad rgb_grad large
eval_run arch_large_rgb_fft_grad_full rgb_fft_grad large
eval_run arch_large_rgb_fft_grad_hp01 rgb_fft_grad large --fft_highpass_only --fft_low_cut_ratio 0.1

########################################
# Group B: two-branch ablations
########################################

eval_run arch_twobranch_small_rgb_grad rgb_grad small --two_branch
eval_run arch_twobranch_large_rgb_grad rgb_grad large --two_branch

########################################
# Group C, D: regularization and extra experiments
# (Add entries here based on run_twobranch_large_extra.sh
#  and run_twobranch_small_extra.sh in the same pattern)
########################################
```

You must:

* Fill in **all** experiment names and corresponding flags from:

  * `run_experiments.sh`
  * `run_twobranch_large_extra.sh`
  * `run_twobranch_small_extra.sh`
* Ensure each `eval_run` call uses the correct `channel_mode`, `model_size`, `two_branch`, and FFT flags from the matching training command.

Once `eval_all.sh` is generated, it should be runnable as:

```bash
bash eval_all.sh
```

This will sequentially evaluate all experiments on the validation split and write per-experiment CSV metric files under each `./weights/<EXPERIMENT_NAME>/` directory.

