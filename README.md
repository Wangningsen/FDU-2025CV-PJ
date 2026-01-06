# AIGCNet: Lightweight Gradient Aware CNNs for AI Generated Image Detection under Generator Shift

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-blue)]([https://huggingface.co/MedVLSynther](https://huggingface.co/2025FDU-CG))

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

| experiment                       | block size | two_branch | channel_mode | label smoothing | Data Augmentation | ACC   | PREC  | REC   | F1    |
|----------------------------------|------------|------------|--------------|-----------------|-------------------|-------|-------|-------|-------|
| arch_small_rgb                   | small      |    FALSE   | RGB          |      FALSE      |        N/A        | 93.88 | 94.51 | 93.35 | 93.93 |
| arch_small_rgb_grad              | small      |    FALSE   | RGB, grad    |      FALSE      |        N/A        | 94.50 | 97.38 | 91.63 | 94.42 |
| reg_single_small_rgb_grad_ls     | small      |    FALSE   | RGB, grad    |       TRUE      |        N/A        | 94.50 | 97.14 | 91.87 | 94.43 |
| arch_large_rgb_grad              | large      |    FALSE   | RGB, grad    |      FALSE      |        N/A        | 94.38 | 95.47 | 93.35 | 94.40 |
| extra_twobranch_large_lr3e4      | large      |    TRUE    | RGB, grad    |      FALSE      |        N/A        | 94.88 | 96.68 | 93.10 | 94.86 |
| extra_twobranch_small_mixup_only | small      |    TRUE    | RGB, grad    |      FALSE      |       mix up      | 96.00 | 97.95 | 94.09 | 95.98 |

OOD benchmark (AIGCDetectBenchmark) for `extra_twobranch_small_mixup_only`:

| Generator  | Ours  | CNNSpot | FreDect | Fusing | GramNet | LNP   | LGrad | DIRE-G | DIRE-D | UnivFD | RPTCon | NPR   |
|------------|-------|---------|---------|--------|---------|-------|-------|--------|--------|--------|--------|-------|
| ProGAN     | 56.54 |  100.00 |   99.36 | 100.00 |   99.99 | 99.67 | 99.83 |  95.19 |  52.75 |  99.81 | 100.00 | 99.90 |
| StyleGan   | 47.47 |   90.17 |   78.02 |  85.20 |   87.05 | 91.75 | 91.08 |  83.03 |  51.31 |  84.93 |  92.77 | 96.10 |
| BigGAN     | 55.93 |   71.17 |   81.97 |  77.40 |   67.33 | 77.75 | 85.62 |  70.12 |  49.70 |  95.08 |  95.80 | 87.30 |
| CycleGAN   | 57.04 |   87.62 |   78.77 |  87.00 |   86.07 | 84.10 | 86.94 |  74.19 |  49.58 |  98.33 |  70.17 | 90.30 |
| StarGAN    | 73.91 |   94.60 |   94.62 |  97.00 |   95.05 | 99.92 | 99.27 |  95.47 |  46.72 |  95.75 |  99.97 | 99.60 |
| GauGAN     | 50.36 |   81.42 |   80.57 |  77.00 |   69.35 | 75.39 | 78.46 |  67.79 |  51.23 |  99.47 |  71.58 | 85.40 |
| Stylegan2  | 49.03 |   86.91 |   66.19 |  83.30 |   87.28 | 94.64 | 85.32 |  75.31 |  51.72 |  74.96 |  89.55 | 98.10 |
| WFIR       | 48.85 |   91.65 |   50.75 |  66.80 |   86.80 | 70.85 | 55.70 |  58.05 |  53.30 |  86.90 |  85.80 | 60.70 |
| ADM        | 97.28 |   60.39 |   63.42 |  49.00 |   58.61 | 84.73 | 67.15 |  75.78 |  98.25 |  66.87 |  82.17 | 84.90 |
| Glide      | 96.07 |   58.07 |   54.13 |  57.20 |   54.50 | 80.52 | 66.11 |  71.75 |  92.42 |  62.46 |  83.79 | 96.70 |
| Midjourney | 71.07 |   51.39 |   45.87 |  52.20 |   50.02 | 65.55 | 65.35 |  58.01 |  89.45 |  56.13 |  90.12 | 92.60 |
| SDv1.4     | 96.73 |   50.57 |   38.79 |  51.00 |   51.70 | 85.55 | 63.02 |  49.74 |  91.24 |  63.66 |  95.38 | 97.40 |
| SDv1.5     | 97.11 |   50.53 |   39.21 |  51.40 |   52.16 | 85.67 | 63.67 |  49.83 |  91.63 |  63.49 |  95.30 | 97.50 |
| VQDM       | 96.12 |   56.46 |   77.80 |  55.10 |   52.86 | 74.46 | 72.99 |  53.68 |  91.90 |  85.31 |  88.91 | 90.10 |
| Wukong     | 96.89 |   51.03 |   40.30 |  51.70 |   50.76 | 82.06 | 59.55 |  54.46 |  90.90 |  70.93 |  91.07 | 91.70 |
| DALLE2     | 74.70 |   50.45 |   34.70 |  52.80 |   49.25 | 88.75 | 65.45 |  66.48 |  92.45 |  50.75 |  96.60 | 99.60 |
| Average    | 72.82 |   70.78 |   64.03 |  68.38 |   68.67 | 83.84 | 75.34 |  68.68 |  71.53 |  78.43 |  89.31 | 91.70 |

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
