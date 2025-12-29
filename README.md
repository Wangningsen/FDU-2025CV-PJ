# AIGCNet: Lightweight Gradient Aware CNNs for AI Generated Image Detection under Generator Shift

## Overview
We tackle binary classification of real vs AI generated images under the course rules of training from scratch: no pretrained weights and no external data beyond the provided dataset. The method builds ultra lightweight residual CNNs (AIGCNetSmall and AIGCNetLarge) in both single-branch and two-branch forms. Inputs can include RGB, Sobel gradients, and optional FFT magnitude channels. Models are trained on a train/val split carved from the course training set, and evaluated both in-domain and under distribution shift on the AIGCDetectBenchmark.

## Key Features
- Compact residual CNNs: single-branch AIGCNetSmall ~2.76M params; two-branch small ~5.51M params; larger backbone option for ablations.
- Flexible inputs: `rgb` (3ch), `rgb_fft` (4ch), `rgb_grad` (5ch), `rgb_fft_grad` (6ch), with optional FFT high-pass filtering.
- Single-branch and two-branch RGB+Sobel architectures; two-branch concatenates features from separate RGB and gradient towers.
- Regularization switches for strong augmentation, label smoothing, mixup, TTA, and simple logit ensembling.
- OOD evaluation on multiple generator families via automated per-subset flattening of AIGCDetectBenchmark.

## Repository Structure
```text
.
|-- train.py
|-- inference.py
|-- dataset.py
|-- models/
|   |-- model.py
|   `-- __init__.py
|-- eval/
|   |-- eval_val.py
|   |-- eval_test.py
|   `-- eval_aigcdetect.py
|-- scripts/
|   |-- run_experiments.sh
|   |-- run_twobranch_small_extra.sh
|   |-- run_twobranch_large_extra.sh
|   |-- run_twobranch_large_extra_extra.sh
|   |-- run_train.sh
|   |-- eval_all.sh
|   `-- eval_all_test.sh
|-- results/              # validation and benchmark CSVs
|-- weights/              # checkpoints per run name (created after training)
|-- misc/                 # dataset helpers and analysis utilities
`-- docs/                 # task notes
```

## Environment and Requirements
- Recommended: Python 3.10, PyTorch 2.x with CUDA if available.
- Core dependencies from the code: `torch`, `torchvision`, `numpy`, `Pillow`, `scikit-learn` (AIGCDetect evaluation), `wandb` (optional logging).
- Example setup:
```bash
conda create -n cvpj python=3.10
conda activate cvpj
pip install torch torchvision torchaudio  # choose the right CUDA build from pytorch.org
pip install numpy pillow scikit-learn wandb
```

## Dataset Preparation
- Expected layout for training and validation:
```text
data_root/
  train/
    0_real/
    1_fake/
```
- Test evaluation (`eval_test.py`) expects labeled folders (default `test_new`):
```text
data_root/
  test_new/
    0_real/
    1_fake/
```
- Inference on unlabeled test images (`inference.py`) reads `data_root/test/` with images directly inside.
- `misc/organize_test.py` can reorganize a labeled CSV (`misc/label.csv`) into `test_new/`.
- `misc/export_val_for_npr.py` shows how the train/val split is reconstructed with `val_ratio` and `seed`.
- Training scripts read `DATA_DIR` from `.env` if present (see `scripts/run_experiments.sh`).

## Model Architecture
### Single-branch (AIGCNetSmall / AIGCNetLarge)
- Stem: 3x3 conv -> BN -> ReLU.
- Three residual stages with downsampling: 32 -> 64 -> 128 -> 256 (small) or 64 -> 128 -> 256 -> 512 (large); 2 blocks per stage for small, 3 per stage for large.
- Global average pooling and a linear 2-way classifier.
- Channel modes:
  - `rgb`: 3 channels.
  - `rgb_fft`: add 1-channel FFT magnitude.
  - `rgb_grad`: add Sobel Gx, Gy (5 channels total).
  - `rgb_fft_grad`: RGB + FFT + Sobel (6 channels), optional `--fft_highpass_only --fft_low_cut_ratio 0.1` to remove low-frequency components.

### Two-branch (TwoBranchAIGCNet)
- Separate RGB (3ch) and gradient (2ch) backbones with shared structure (small or large).
- Features are concatenated and fed to a final linear head; parameter count for the small two-branch model is ~5.51M.
- `train.py` forces `--channel_mode rgb_grad` when `--two_branch` is enabled.

## Training
Core entrypoint: `train.py` (AdamW + cosine LR schedule).
```bash
python train.py \
  --data_dir /path/to/data_root \
  --save_dir ./weights/extra_twobranch_small_mixup_only \
  --model_size small \
  --channel_mode rgb_grad \
  --two_branch \
  --epochs 50 \
  --batch_size 64 \
  --lr 3e-4 \
  --use_mixup --mixup_alpha 0.4 \
  --use_label_smoothing --label_smoothing 0.1 \
  --use_strong_aug \
  --use_tta
```
Key options:
- `--channel_mode {rgb,rgb_fft,rgb_grad,rgb_fft_grad}` selects input channels; `--two_branch` uses RGB+Sobel towers.
- Regularization: `--use_strong_aug`, `--use_label_smoothing`, `--use_mixup`, `--use_tta`.
- FFT controls: `--fft_highpass_only`, `--fft_low_cut_ratio`.
- Logging: `--use_wandb` plus environment variables `WANDB_PROJECT`, `WANDB_ENTITY`.
- Outputs: a run folder under `--save_dir` with `best_model.pth` and `train.log`.

Helper launchers under `scripts/` mirror these flags (e.g., `run_experiments.sh`, `run_twobranch_*`); each run name maps to `weights/<run_name>/`.

## Validation Evaluation
Recreate the train/val split with the same `--val_ratio` and `--seed`.
```bash
python eval/eval_val.py \
  --data_dir /path/to/data_root \
  --checkpoint weights/extra_twobranch_small_mixup_only/best_model.pth \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --val_ratio 0.1 \
  --seed 42 \
  --use_tta
```
- Reports accuracy, precision, recall, F1, TP/TN/FP/FN.
- Writes a CSV next to the checkpoint (`<run_name>.csv` by default).
- Batch evaluation script: `scripts/eval_all.sh`.

## Test and OOD Evaluation
- Held-out test with labels:
```bash
python eval/eval_test.py \
  --data_dir /path/to/data_root \
  --test_subdir test_new \
  --checkpoint weights/extra_twobranch_small_mixup_only/best_model.pth \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_tta
```
Outputs `<run_name>_test.csv`. Batch variant: `scripts/eval_all_test.sh`.

- External benchmark (AIGCDetectBenchmark):
```bash
python eval/eval_aigcdetect.py \
  --benchmark_root /path/to/AIGCDetectBenchmark \
  --checkpoint weights/extra_twobranch_small_mixup_only/best_model.pth \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --work_dir ./aigcdetect_tmp \
  --use_tta
```
Each subset is flattened into `work_dir/<subset>/{0_real,1_fake}` via symlinks/copies, evaluated, and aggregated into `<run_name>_aigcdetect.csv` (per-subset plus global metrics).

- Unlabeled inference for submission:
```bash
python inference.py \
  --data_dir /path/to/data_root \
  --checkpoint weights/extra_twobranch_small_mixup_only/best_model.pth \
  --channel_mode rgb_grad \
  --model_size small \
  --two_branch \
  --use_tta \
  --output result.csv
```

## Results
Validation split (0.1 hold-out from train, seed 42):

| Experiment | Channels / Branch | Reg. | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|---|
| arch_small_rgb | RGB / single | - | 0.9388 | 0.9451 | 0.9335 | 0.9393 |
| arch_small_rgb_grad | RGB+Sobel / single | - | 0.9450 | 0.9738 | 0.9163 | 0.9442 |
| arch_twobranch_small_rgb_grad | RGB+Sobel / two | - | 0.9350 | 0.9634 | 0.9064 | 0.9340 |
| extra_twobranch_small_mixup_only | RGB+Sobel / two | mixup | **0.9600** | **0.9795** | **0.9409** | **0.9598** |
| reg_twobranch_small_strong | RGB+Sobel / two | strong aug | 0.8163 | 0.8106 | 0.8325 | 0.8214 |

OOD benchmark (AIGCDetectBenchmark) for `extra_twobranch_small_mixup_only`:

| Subset | Acc | Prec | Rec | F1 |
|---|---|---|---|---|
| ADM | **0.9728** | 0.9692 | 0.9767 | 0.9729 |
| Stable Diffusion v1.5 | 0.9711 | **0.9721** | 0.9701 | **0.9711** |
| Wukong | 0.9689 | 0.9718 | 0.9658 | 0.9688 |
| Midjourney | 0.7107 | 0.9282 | 0.4567 | 0.6122 |
| GauGAN | 0.5036 | 0.5588 | 0.0342 | 0.0645 |
| WhichFaceIsReal | 0.4885 | 0.1892 | **0.0070** | 0.0135 |
| GLOBAL | 0.7668 | 0.9251 | 0.5806 | 0.7135 |
| MEAN_SUBSETS | 0.7254 | - | - | - |

Observations: mixup-only two-branch delivers the best in-domain metrics; strong augmentation alone reduces val accuracy. OOD accuracy is high on diffusion-style subsets but drops sharply on GAN-based or low-diversity sources (e.g., GauGAN, WhichFaceIsReal), highlighting distribution shift challenges.

## Reproducing Our Main Experiments
1. Install the environment and dependencies.
2. Prepare the dataset folders (`train/0_real`, `train/1_fake`, and `test_new/` if labeled test is available).
3. Launch training for the main model (e.g., `extra_twobranch_small_mixup_only`) via `train.py` or the matching script under `scripts/`.
4. Run validation with `eval/eval_val.py` (or `scripts/eval_all.sh`) using the same `val_ratio` and `seed`.
5. Evaluate on the held-out test split with `eval/eval_test.py` and on AIGCDetectBenchmark with `eval/eval_aigcdetect.py`.
6. Review the generated CSVs under each `weights/<run_name>/` and the aggregated files in `results/`.

## Limitations and Future Work
- Models are intentionally small and trained without any external data or pretrained weights.
- Transformer-style architectures and self-supervised pretraining are not explored.
- FFT channel design is simple (log-magnitude plus optional high-pass); richer frequency cues or learned filters could help.
- Robustness to unseen GAN families remains limited; stronger domain generalization and augmentation strategies are future directions.

## Acknowledgments
This project is developed for the Fudan University computer vision course (FDU-2025CV-PJ). Thanks to the course staff and the AIGCDetectBenchmark authors for providing evaluation data. WandB is used for experiment tracking when enabled.
