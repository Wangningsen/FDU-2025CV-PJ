#!/usr/bin/env python
"""
Analyze frequency domain statistics for real vs fake images.

Usage (from repo root):
    python analyze_freq_stats.py \
        --data_dir /data1/nwang60/datasets/CV-Dataset \
        --split train \
        --img_size 256 \
        --max_per_class 1000 \
        --out_dir ./freq_stats

Expected directory structure:
    <data_dir>/<split>/
        0_real/
        1_fake/
"""

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


def list_images_with_label(root: str, subdir: str, label: int) -> List[Tuple[str, int]]:
    folder = os.path.join(root, subdir)
    if not os.path.isdir(folder):
        raise RuntimeError(f"Directory not found: {folder}")
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if is_image_file(f)
    ]
    return [(fp, label) for fp in files]


def load_image_as_tensor(path: str, img_size: int) -> torch.Tensor:
    """
    Load image as float tensor in [0, 1], shape (3, H, W).
    No random augmentation, only resize.
    """
    img = Image.open(path).convert("RGB")
    if img_size is not None:
        img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # H, W, 3
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # 3, H, W
    return tensor


def compute_fft_mag(gray: torch.Tensor) -> torch.Tensor:
    """
    gray: (H, W), float32 in [0, 1]
    return: log-amplitude spectrum, (H, W)
    """
    F = torch.fft.fft2(gray)
    F = torch.fft.fftshift(F)
    mag = torch.abs(F)
    mag = torch.log1p(mag)
    return mag


def prepare_radius_map(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Precompute integer radius for each frequency coordinate.
    Returns rr: (H, W) with integer radii 0...max_r.
    """
    cy, cx = h // 2, w // 2
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr.long()
    return rr


def accumulate_stats(
    paths_with_labels: List[Tuple[str, int]],
    img_size: int,
    max_per_class: int,
    device: torch.device,
):
    """
    Loop over images, accumulate:
      - sum of magnitude spectra per class
      - radial sums per class
    """
    # Shuffle to avoid ordering bias
    random.shuffle(paths_with_labels)

    # Take at most max_per_class per label
    selected = {0: [], 1: []}
    for path, label in paths_with_labels:
        if len(selected[label]) < max_per_class:
            selected[label].append((path, label))
        if len(selected[0]) >= max_per_class and len(selected[1]) >= max_per_class:
            break

    print(
        f"Using {len(selected[0])} real images and "
        f"{len(selected[1])} fake images."
    )
    if len(selected[0]) == 0 or len(selected[1]) == 0:
        raise RuntimeError("Not enough images for one of the classes.")

    sum_real = None
    sum_fake = None
    n_real = 0
    n_fake = 0

    rr = None
    radial_sum_real = None
    radial_cnt_real = None
    radial_sum_fake = None
    radial_cnt_fake = None

    # Combine for single loop
    combined = selected[0] + selected[1]

    for idx, (path, label) in enumerate(combined):
        img = load_image_as_tensor(path, img_size).to(device)  # 3, H, W
        r, g, b = img[0], img[1], img[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # H, W
        mag = compute_fft_mag(gray)  # H, W

        if sum_real is None:
            # Initialize accumulators
            h, w = mag.shape
            sum_real = torch.zeros_like(mag)
            sum_fake = torch.zeros_like(mag)
            rr = prepare_radius_map(h, w, device=device)
            max_r = int(rr.max().item())
            radial_sum_real = torch.zeros(max_r + 1, device=device)
            radial_cnt_real = torch.zeros(max_r + 1, device=device)
            radial_sum_fake = torch.zeros(max_r + 1, device=device)
            radial_cnt_fake = torch.zeros(max_r + 1, device=device)

        if label == 0:
            sum_real += mag
            n_real += 1
            radial_sum_real.scatter_add_(0, rr.view(-1), mag.view(-1))
            radial_cnt_real.scatter_add_(0, rr.view(-1), torch.ones_like(mag).view(-1))
        else:
            sum_fake += mag
            n_fake += 1
            radial_sum_fake.scatter_add_(0, rr.view(-1), mag.view(-1))
            radial_cnt_fake.scatter_add_(0, rr.view(-1), torch.ones_like(mag).view(-1))

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} / {len(combined)} images")

    # Compute means
    mean_real = sum_real / max(n_real, 1)
    mean_fake = sum_fake / max(n_fake, 1)

    # Radial profiles
    prof_real = radial_sum_real / torch.clamp(radial_cnt_real, min=1)
    prof_fake = radial_sum_fake / torch.clamp(radial_cnt_fake, min=1)

    return mean_real, mean_fake, prof_real, prof_fake


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """
    Jensen-Shannon divergence between two discrete distributions.
    """
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        a = a[mask]
        b = b[mask]
        return np.sum(a * (np.log(a + eps) - np.log(b + eps)))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(js)


def plot_mean_spectra(mean_real, mean_fake, out_dir: str):
    mean_real_np = mean_real.cpu().numpy()
    mean_fake_np = mean_fake.cpu().numpy()
    diff_np = mean_fake_np - mean_real_np

    vmin = min(mean_real_np.min(), mean_fake_np.min())
    vmax = max(mean_real_np.max(), mean_fake_np.max())

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Mean spectrum (real)")
    plt.imshow(mean_real_np, cmap="magma", vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Mean spectrum (fake)")
    plt.imshow(mean_fake_np, cmap="magma", vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Fake minus real")
    plt.imshow(diff_np, cmap="bwr")
    plt.colorbar()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mean_spectra.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved mean spectra figure to {out_path}")


def plot_radial_profiles(prof_real, prof_fake, out_dir: str):
    pr = prof_real.cpu().numpy()
    pf = prof_fake.cpu().numpy()

    # Optionally truncate very high radii tail
    max_r = len(pr)
    radii = np.arange(max_r) / float(max_r - 1)

    plt.figure(figsize=(6, 4))
    plt.plot(radii, pr, label="real")
    plt.plot(radii, pf, label="fake")
    plt.xlabel("Normalized radius (0 low freq, 1 high freq)")
    plt.ylabel("Average log magnitude")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "radial_profiles.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved radial profiles figure to {out_path}")

    # Also plot difference
    diff = pf - pr
    plt.figure(figsize=(6, 4))
    plt.plot(radii, diff)
    plt.xlabel("Normalized radius")
    plt.ylabel("Fake minus real")
    plt.grid(alpha=0.3)
    out_path = os.path.join(out_dir, "radial_diff.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved radial difference figure to {out_path}")

    # Print simple distances
    # Normalize to probability for JS
    pr_clip = np.clip(pr, a_min=0.0, a_max=None)
    pf_clip = np.clip(pf, a_min=0.0, a_max=None)
    js = js_divergence(pr_clip, pf_clip)
    # L2 distance after L2 normalization
    pr_norm = pr_clip / (np.linalg.norm(pr_clip) + 1e-8)
    pf_norm = pf_clip / (np.linalg.norm(pf_clip) + 1e-8)
    l2 = float(np.linalg.norm(pr_norm - pf_norm))

    print(f"Jensen-Shannon divergence between radial profiles: {js:.6f}")
    print(f"L2 distance between normalized profiles: {l2:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze real vs fake frequency statistics."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Root dataset directory that contains <split>/0_real and 1_fake.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to analyze (e.g. train).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Resize images to this size before FFT.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=1000,
        help="Max number of images per class to use.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./freq_stats",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    split_root = os.path.join(args.data_dir, args.split)
    if not os.path.isdir(split_root):
        raise RuntimeError(f"Split directory not found: {split_root}")

    paths_real = list_images_with_label(split_root, "0_real", label=0)
    paths_fake = list_images_with_label(split_root, "1_fake", label=1)
    all_paths = paths_real + paths_fake

    (
        mean_real,
        mean_fake,
        prof_real,
        prof_fake,
    ) = accumulate_stats(
        all_paths,
        img_size=args.img_size,
        max_per_class=args.max_per_class,
        device=device,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # Save numpy arrays for later use if needed
    np.save(os.path.join(args.out_dir, "mean_real.npy"), mean_real.cpu().numpy())
    np.save(os.path.join(args.out_dir, "mean_fake.npy"), mean_fake.cpu().numpy())
    np.save(os.path.join(args.out_dir, "prof_real.npy"), prof_real.cpu().numpy())
    np.save(os.path.join(args.out_dir, "prof_fake.npy"), prof_fake.cpu().numpy())
    print(f"Saved raw arrays to {args.out_dir}")

    plot_mean_spectra(mean_real, mean_fake, args.out_dir)
    plot_radial_profiles(prof_real, prof_fake, args.out_dir)


if __name__ == "__main__":
    main()
