# General code style rules

- Don't use typing.
- Write as concise code as possible.
- Don't be overly defensive, avoid using try/except too much.
- Declare variables close to where they are used, and avoid creating variables that are used only once.
- Avoid defining new functions for very small snippets, or functions used only once.

# FDU-2025CV-PJ: Extended Experiment Plan (for Code Assistant)

This document describes **concrete coding tasks** to upgrade the baseline deepfake detector in this repo.  
The goals are:

1. Add a larger backbone variant.
2. Add a two-branch RGB + Sobel model.
3. Add training tricks (augmentations, label smoothing, etc.) with clean flags for ablation.
4. Add simple analysis utilities.

All tasks must **not** use any pretrained models (no torchvision pretrained weights, no external checkpoints).  
Everything stays as a small CNN trained from scratch on the given dataset.

The main entrypoint is `train.py`. Model definitions live under `models/`.  
All new functionality should be controllable via command line flags so different configs can be compared.

---

## 0. Argument parsing and configuration

**File:** `train.py`

1. Locate the place where `argparse.ArgumentParser` is created (e.g. `def parse_args()`).
2. Add the following arguments (names and choices exactly as below):

```python
parser.add_argument(
    "--model_size",
    type=str,
    default="small",
    choices=["small", "large"],
    help="Backbone size: small (default) or large."
)

parser.add_argument(
    "--two_branch",
    action="store_true",
    help="If set, use a two-branch RGB + Sobel model instead of a single-branch model."
)

parser.add_argument(
    "--use_label_smoothing",
    action="store_true",
    help="If set, enable label smoothing for classification loss."
)

parser.add_argument(
    "--label_smoothing",
    type=float,
    default=0.1,
    help="Epsilon value for label smoothing (only used if --use_label_smoothing)."
)

parser.add_argument(
    "--use_mixup",
    action="store_true",
    help="If set, enable Mixup during training."
)

parser.add_argument(
    "--mixup_alpha",
    type=float,
    default=0.4,
    help="Alpha parameter for Mixup Beta distribution."
)

parser.add_argument(
    "--use_strong_aug",
    action="store_true",
    help="If set, enable stronger data augmentations (crop, blur, JPEG, grayscale)."
)

parser.add_argument(
    "--use_tta",
    action="store_true",
    help="If set, enable simple test-time augmentation (horizontal flip)."
)

parser.add_argument(
    "--ensemble_from",
    type=str,
    default="",
    help="Optional: comma-separated list of checkpoint paths to ensemble at inference. "
         "If non-empty, load these models and average their logits."
)
````

3. Ensure these new flags are included in any config or logging (e.g. if using wandb, log them in the config dict).

---

## 1. Model upgrade: Large backbone variant

We assume there is already a small backbone (e.g. `AIGCNet` or a similar CNN) in `models/`.
We want to add a **large** variant with:

* Channels: 64–128–256–512 (double the current 32–64–128–256).
* One extra residual block per stage compared to the small version.

### 1.1 Add `AIGCNetSmall` and `AIGCNetLarge`

**File:** `models/aigcnet.py` (or the file where the current backbone is defined)

Tasks:

1. Refactor the existing backbone into a class named `AIGCNetSmall` if it is not already split:

   * Keep the current channel sizes (likely 32–64–128–256).
   * Keep the current number of blocks per stage (for example, 2 residual blocks per stage).

2. Add a new class `AIGCNetLarge` with:

   * Initial conv stem with 64 channels instead of 32.
   * Stages with output channels: 64, 128, 256, 512.
   * Each stage should have **one more residual block** than `AIGCNetSmall`.

     * Example: if small has [2, 2, 2, 2] blocks per stage, large should have [3, 3, 3, 3].

3. Both classes should expose the same interface:

   ```python
   class AIGCNetSmall(nn.Module):
       def __init__(self, in_channels: int = 3, num_classes: int = 2):
           ...

       def forward(self, x):
           ...
   ```

   ```python
   class AIGCNetLarge(nn.Module):
       def __init__(self, in_channels: int = 3, num_classes: int = 2):
           ...

       def forward(self, x):
           ...
   ```

4. If there is a `__all__` or a factory in `models/__init__.py`, export both:

   ```python
   from .aigcnet import AIGCNetSmall, AIGCNetLarge
   ```

### 1.2 Wire `model_size` into model creation

**File:** `train.py`

1. Find where the model is instantiated (something like `model = AIGCNet(...)`).

2. Modify this so it uses `args.model_size` and (later) `args.two_branch`:

   ```python
   from models import AIGCNetSmall, AIGCNetLarge, TwoBranchAIGCNet

   def build_model(args):
       if args.two_branch:
           # Two-branch always uses RGB + Sobel, input_channels can be computed accordingly
           model = TwoBranchAIGCNet(
               backbone_size=args.model_size,  # "small" or "large"
               num_classes=num_classes
           )
       else:
           backbone_cls = AIGCNetSmall if args.model_size == "small" else AIGCNetLarge
           model = backbone_cls(
               in_channels=input_channels,
               num_classes=num_classes
           )
       return model
   ```

3. Ensure training script prints the model configuration and number of parameters:

   ```python
   n_params = sum(p.numel() for p in model.parameters())
   print(f"Model: {model.__class__.__name__}, size={args.model_size}, "
         f"two_branch={args.two_branch}, params={n_params / 1e6:.2f}M")
   ```

---

## 2. Two-branch RGB + Sobel model

The idea:

* Branch 1: takes RGB input (3 channels).
* Branch 2: takes Sobel gradients (2 channels: Gx, Gy).
* Each branch uses its own backbone (small or large), no weight sharing.
* Features are fused at the end of the backbones by concatenation and a small head.

We assume the dataset code can already produce a stacked tensor of `[RGB, SobelGx, SobelGy]` as channels.
For the two-branch model, we will split the channels: `x_rgb = x[:, :3]`, `x_grad = x[:, 3:5]`.

### 2.1 Add `TwoBranchAIGCNet`

**File:** `models/aigcnet.py` (same file as AIGCNet classes)

1. Add a new class:

   ```python
   class TwoBranchAIGCNet(nn.Module):
       def __init__(self, backbone_size: str = "small", num_classes: int = 2):
           super().__init__()
           assert backbone_size in ["small", "large"]

           backbone_cls = AIGCNetSmall if backbone_size == "small" else AIGCNetLarge

           # RGB branch: 3-channel input
           self.rgb_backbone = backbone_cls(in_channels=3, num_classes=None)

           # Sobel branch: 2-channel input
           self.grad_backbone = backbone_cls(in_channels=2, num_classes=None)

           # Assume both backbones output a feature vector of dimension F
           # After concat, we get 2F -> final classifier head
           feature_dim = self.rgb_backbone.output_dim  # add this attribute to backbones if needed
           self.classifier = nn.Linear(2 * feature_dim, num_classes)

       def forward(self, x):
           # x: (B, C, H, W) where C >= 5 (RGB + SobelGx + SobelGy + maybe others)
           x_rgb = x[:, :3]      # (B, 3, H, W)
           x_grad = x[:, 3:5]    # (B, 2, H, W)

           feat_rgb = self.rgb_backbone.forward_features(x_rgb)
           feat_grad = self.grad_backbone.forward_features(x_grad)

           feat = torch.cat([feat_rgb, feat_grad], dim=1)
           logits = self.classifier(feat)
           return logits
   ```

2. For this to work, the single-branch backbones should expose a method like:

   ```python
   def forward_features(self, x) -> torch.Tensor:
       """
       Returns a global-pooled feature vector of shape (B, F)
       without applying the final classifier.
       """
   ```

   And the normal `forward` can call `forward_features` then its own classifier.

3. Make sure the backbones define an attribute `output_dim` indicating the feature dimension `F`.

4. Export `TwoBranchAIGCNet` from `models/__init__.py`.

### 2.2 Wire `--two_branch` and input mode

We assume there is already some way to select input mode (`rgb_only`, `rgb_sobel`, etc.) in `dataset.py` and/or `train.py`.
For the two-branch model, we want the dataset to produce at least RGB + Sobel (5 channels total).

**File:** `dataset.py` and `train.py`

Tasks:

1. Ensure there is an input mode (e.g. `"rgb_grad"`) where:

   * A sample returns a tensor with channels `[R, G, B, SobelGx, SobelGy]`.

2. In `train.py`, when `args.two_branch` is `True`:

   * Force the input mode to be `"rgb_grad"` (override any default if needed).
   * Set `input_channels` accordingly (5 channels or more, but the model will only use the first 5).

3. In the `build_model(args)` function (see section 1.2), instantiate `TwoBranchAIGCNet` when `args.two_branch` is `True`.

---

## 3. Stronger data augmentation

We want to add an optional stronger augmentation pipeline to make the model more robust.
Use **only** torchvision and PIL, no external libraries.

Augmentations to include:

* `RandomResizedCrop(256, scale=(0.8, 1.0))`
* `RandomHorizontalFlip`
* `ColorJitter` (already there if used)
* Random Gaussian blur
* Random JPEG-like compression
* Random grayscale

### 3.1 Add strong transform builder

**File:** `dataset.py`

1. Locate the place where training transforms are built (e.g. `build_train_transform` or similar).

2. Add a helper function:

   ```python
   from torchvision import transforms
   from PIL import Image
   import io
   import random

   def random_jpeg_compress(img: Image.Image, quality_range=(30, 80)):
       q = random.randint(quality_range[0], quality_range[1])
       buffer = io.BytesIO()
       img.save(buffer, format="JPEG", quality=q)
       buffer.seek(0)
       return Image.open(buffer).convert("RGB")

   class RandomJPEGCompression(object):
       def __init__(self, quality_range=(30, 80), p=0.5):
           self.quality_range = quality_range
           self.p = p

       def __call__(self, img):
           if random.random() < self.p:
               return random_jpeg_compress(img, self.quality_range)
           return img
   ```

3. Add a function to build the strong augmentation pipeline:

   ```python
   def build_strong_train_transform(img_size=256):
       return transforms.Compose([
           transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
           transforms.RandomHorizontalFlip(),
           transforms.ColorJitter(
               brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
           ),
           transforms.RandomApply(
               [transforms.GaussianBlur(kernel_size=3)],
               p=0.3,
           ),
           transforms.RandomApply(
               [RandomJPEGCompression(quality_range=(30, 80))],
               p=0.5,
           ),
           transforms.RandomApply(
               [transforms.RandomGrayscale(p=1.0)],
               p=0.2,
           ),
           transforms.ToTensor(),
       ])
   ```

4. In the dataset class initialization, add a parameter `use_strong_aug: bool = False`.
   If `use_strong_aug` is `True`, use `build_strong_train_transform`, otherwise keep the original transform.

5. Update `train.py` to pass `use_strong_aug=args.use_strong_aug` when creating the training dataset.

---

## 4. Regularization: label smoothing and Mixup

### 4.1 Label smoothing implementation

**File:** `train.py`

1. Add a helper to apply label smoothing to integer labels:

   ```python
   import torch.nn.functional as F

   def smooth_labels(targets, num_classes, smoothing):
       # targets: (B,) integer labels
       with torch.no_grad():
           confidence = 1.0 - smoothing
           off_value = smoothing / (num_classes - 1)
           one_hot = torch.full(
               (targets.size(0), num_classes),
               off_value,
               device=targets.device
           )
           one_hot.scatter_(1, targets.unsqueeze(1), confidence)
       return one_hot
   ```

2. In the training loop, instead of:

   ```python
   loss = criterion(logits, labels)
   ```

   do:

   ```python
   if args.use_label_smoothing:
       num_classes = logits.size(1)
       soft_labels = smooth_labels(labels, num_classes, args.label_smoothing)
       log_probs = F.log_softmax(logits, dim=1)
       loss = -(soft_labels * log_probs).sum(dim=1).mean()
   else:
       loss = criterion(logits, labels)
   ```

   Here `criterion` can still be `nn.CrossEntropyLoss()` for the non-smoothed case.

### 4.2 Mixup implementation

**File:** `train.py`

1. Add a helper:

   ```python
   import numpy as np

   def mixup_data(x, y, alpha):
       if alpha <= 0:
           return x, y, y, 1.0
       lam = np.random.beta(alpha, alpha)
       batch_size = x.size(0)
       index = torch.randperm(batch_size).to(x.device)
       mixed_x = lam * x + (1 - lam) * x[index, :]
       y_a, y_b = y, y[index]
       return mixed_x, y_a, y_b, lam
   ```

2. In the training loop, if `args.use_mixup` is `True`:

   ```python
   if args.use_mixup:
       inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
       outputs = model(inputs)

       if args.use_label_smoothing:
           num_classes = outputs.size(1)
           soft_a = smooth_labels(targets_a, num_classes, args.label_smoothing)
           soft_b = smooth_labels(targets_b, num_classes, args.label_smoothing)
           soft_labels = lam * soft_a + (1 - lam) * soft_b
           log_probs = F.log_softmax(outputs, dim=1)
           loss = -(soft_labels * log_probs).sum(dim=1).mean()
       else:
           loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
   else:
       outputs = model(inputs)
       # use label smoothing branch from 4.1
   ```

3. Ensure metrics are still computed on the **hard** predictions vs the original labels (no need to change that).

---

## 5. Simple TTA and ensemble at inference

**File:** `inference.py` and/or `train.py` (depending on how evaluation is implemented)

### 5.1 Test-time augmentation

For TTA we only use a simple horizontal flip:

1. In the evaluation/inference function, if `args.use_tta` is `False`, keep current behavior.
2. If `args.use_tta` is `True`:

   ```python
   def tta_predict(model, x):
       # x: (B, C, H, W)
       logits1 = model(x)
       x_flipped = torch.flip(x, dims=[3])  # flip width
       logits2 = model(x_flipped)
       return (logits1 + logits2) / 2.0
   ```

   Use `tta_predict` instead of a single forward call.

### 5.2 Simple ensemble from checkpoints

We want the ability to average logits from multiple trained models (e.g. one RGB-only, one RGB+Sobel).

1. If `args.ensemble_from` is a non-empty string:

   * Split by comma to get a list of checkpoint paths.
   * For each path:

     * Create a model with the same architecture as at training time.
     * Load the checkpoint.
     * Put the model in eval mode and add to a list `models`.

2. During inference, instead of using a single `model`, use:

   ```python
   def ensemble_predict(models, x):
       logits_sum = None
       for m in models:
           logits = m(x)
           if logits_sum is None:
               logits_sum = logits
           else:
               logits_sum = logits_sum + logits
       return logits_sum / len(models)
   ```

3. Combine TTA and ensemble if both are enabled:

   * For each model, apply TTA or not depending on `args.use_tta`.
   * Average over all models.

---

## 6. Analysis utilities (optional but recommended)

These parts are "analysis" oriented. They can live in a separate script, e.g. `analysis_tools.py`.

### 6.1 Patch-level score map

**File:** `analysis_tools.py` (new file)

Implement a function that, given:

* a trained model,
* an image tensor `(1, C, H, W)`,
* a patch size (e.g. 128),

slides a window over the image and records the **fake-class logit** for each patch.

Rough outline:

```python
def compute_patch_score_map(model, img, patch_size=128, stride=64, fake_class_index=1):
    """
    img: (1, C, H, W) tensor.
    Returns: a 2D tensor of shape (num_patches_y, num_patches_x)
    with the logits for the fake class.
    """
    model.eval()
    _, C, H, W = img.shape
    scores = []
    ys = []
    xs = []

    for y in range(0, H - patch_size + 1, stride):
        row_scores = []
        for x in range(0, W - patch_size + 1, stride):
            patch = img[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                logits = model(patch)
                fake_logit = logits[:, fake_class_index].item()
            row_scores.append(fake_logit)
        scores.append(row_scores)

    return torch.tensor(scores)
```

This can later be visualized as a heatmap in a notebook.

### 6.2 Robustness test under perturbations

Add a small script or function that:

* Loads a trained RGB-only model and an RGB+Sobel (or two-branch) model.
* Applies:

  * Gaussian blur,
  * JPEG compression,
  * random noise,

  to validation images and compares accuracy drop for each model.

This analysis is not strictly required for code correctness, but the helper functions from section 3 (blur, JPEG) can be reused here.

---

## 7. Suggested ablation matrix (for the human experimenter)

This section is for the human, not for the code assistant.
Once all flags are implemented, you can easily run:

1. **Backbone size ablation** (single-branch, RGB+Sobel):

   * `--model_size small --two_branch False --use_strong_aug False`
   * `--model_size large --two_branch False --use_strong_aug False`

2. **Two-branch ablation**:

   * `--model_size small --two_branch False`
   * `--model_size small --two_branch True`
   * `--model_size large --two_branch False`
   * `--model_size large --two_branch True`

3. **Regularization ablation**:

   * Base: no smoothing, no mixup.
   * `--use_label_smoothing --label_smoothing 0.1`
   * `--use_mixup --mixup_alpha 0.4`
   * Both together.

4. **Robustness / augmentation**:

   * `--use_strong_aug False` vs `--use_strong_aug True`.

5. **Ensemble / TTA**:

   * Single model vs ensemble of RGB-only and RGB+Sobel checkpoints.
   * With and without `--use_tta`.

These can be summarized in the report as several small tables showing gains from each block.

---

