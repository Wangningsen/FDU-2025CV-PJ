#!/usr/bin/env python
import argparse
import os
import shutil
import csv
import numpy as np
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许轻微损坏的图片尽量被读出来

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import AIGCDataset, build_transforms
from models import AIGCNetSmall, AIGCNetLarge, TwoBranchAIGCNet
from utils import set_seed


# ----------------- 通用配置与工具 ----------------- #

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def is_image_file(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTS)


def is_valid_image(path: str) -> bool:
    """
    用 PIL 简单验证图片是否可读，如果坏图就跳过。
    """
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"[WARN] Bad image skipped: {path} ({e})")
        return False


def resolve_channel_mode(mode: str):
    """
    与你 eval_val.py 里保持一致：
    返回 (use_extra_channels, use_fft, use_grad, in_channels)
    """
    if mode == "rgb":
        return False, False, False, 3
    if mode == "rgb_fft":
        return True, True, False, 4
    if mode == "rgb_grad":
        return True, False, True, 5
    # 默认 rgb_fft_grad
    return True, True, True, 6


def build_model(args, num_classes: int, input_channels: int):
    if args.two_branch:
        model = TwoBranchAIGCNet(
            backbone_size=args.model_size,
            num_classes=num_classes,
        )
    else:
        backbone_cls = AIGCNetSmall if args.model_size == "small" else AIGCNetLarge
        model = backbone_cls(
            in_channels=input_channels,
            num_classes=num_classes,
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {model.__class__.__name__}, "
        f"size={args.model_size}, two_branch={args.two_branch}, "
        f"in_channels={input_channels}, params={n_params/1e6:.2f}M"
    )
    return model


def tta_predict(model, x, use_tta: bool):
    if not use_tta:
        return model(x)
    # 简单 TTA：水平翻转
    logits1 = model(x)
    logits2 = model(torch.flip(x, dims=[3]))
    return (logits1 + logits2) / 2.0


def compute_binary_metrics(labels, preds, probs, threshold: float):
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    probs = np.asarray(probs)

    assert labels.shape == preds.shape == probs.shape

    n = labels.shape[0]
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc = (tp + tn) / n if n > 0 else 0.0

    prec_den = tp + fp
    precision = tp / prec_den if prec_den > 0 else 0.0

    rec_den = tp + fn
    recall = tp / rec_den if rec_den > 0 else 0.0

    f1_den = precision + recall
    f1 = 2 * precision * recall / f1_den if f1_den > 0 else 0.0

    pos_rate = preds.mean() if n > 0 else 0.0

    return {
        "num_samples": int(n),
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pos_rate": float(pos_rate),
    }


def default_csv_path(checkpoint_path: str) -> str:
    ckpt_path = checkpoint_path.rstrip("/\\")
    ckpt_dir = os.path.dirname(ckpt_path)
    run_name = os.path.basename(ckpt_dir)
    return os.path.join(ckpt_dir, f"{run_name}_aigcdetect.csv")


# ----------------- 关键：把任意层级拍平成 0_real / 1_fake 视图 ----------------- #

def build_flat_two_class_view(
    benchmark_root: str,
    subset_name: str,
    work_root: str,
) -> Tuple[str, int, int]:
    """
    对于某个子集（例如 cyclegan、progan 等），递归搜索其内部所有目录名为
    0_real / 1_fake 的文件夹，把里面的图片全部聚合到一个新的临时目录：

        <work_root>/<subset_name>/
            0_real/
            1_fake/

    返回: (view_root, real_count, fake_count)
    """
    subset_root = os.path.join(benchmark_root, subset_name)
    view_root = os.path.join(work_root, subset_name)

    real_dest = os.path.join(view_root, "0_real")
    fake_dest = os.path.join(view_root, "1_fake")

    # 先清掉旧的，避免重复链接
    if os.path.exists(view_root):
        shutil.rmtree(view_root)
    os.makedirs(real_dest, exist_ok=True)
    os.makedirs(fake_dest, exist_ok=True)

    real_count = 0
    fake_count = 0

    for root, dirs, files in os.walk(subset_root):
        for fname in files:
            if not is_image_file(fname):
                continue

            src = os.path.join(root, fname)
            # 筛掉坏图
            if not is_valid_image(src):
                continue

            # 根据路径判断类别：看到 /0_real/ 就当真图，/1_fake/ 就当假图
            norm_root = root.replace("\\", "/")
            if "/0_real" in norm_root:
                label = 0
            elif "/1_fake" in norm_root:
                label = 1
            else:
                continue

            if label == 0:
                idx = real_count
                real_count += 1
                dst_dir = real_dest
            else:
                idx = fake_count
                fake_count += 1
                dst_dir = fake_dest

            ext = os.path.splitext(fname)[1]
            dst_name = f"{subset_name}_{idx:06d}{ext}"
            dst = os.path.join(dst_dir, dst_name)

            # 优先用软链接，软链接不行就复制
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    return view_root, real_count, fake_count


# ----------------- 主流程 ----------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AIGCNet on AIGCDetectBenchmark by subset."
    )

    parser.add_argument(
        "--benchmark_root",
        type=str,
        required=True,
        help="Root directory of AIGCDetectBenchmark.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. weights/extra_twobranch_small_mixup_only/best_model.pth",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "large"],
    )
    parser.add_argument(
        "--two_branch",
        action="store_true",
        help="Use TwoBranchAIGCNet (must match training).",
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="rgb_grad",
        choices=["rgb", "rgb_fft", "rgb_grad", "rgb_fft_grad"],
        help="Channel mode used at training time.",
    )
    parser.add_argument(
        "--fft_highpass_only",
        action="store_true",
        help="If set, keep only high frequency FFT components (must match training).",
    )
    parser.add_argument(
        "--fft_low_cut_ratio",
        type=float,
        default=0.1,
        help="Low frequency radius ratio for highpass FFT (must match training).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Image size, must match training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on P(fake).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="./aigcdetect_tmp",
        help="Temporary directory to create flattened 0_real / 1_fake views.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Output CSV path for per-subset and overall metrics. "
             "If empty, use <ckpt_dir>/<run_name>_aigcdetect.csv",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Enable simple test-time augmentation (horizontal flip).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # eval transform（和你自己训练时 val transform 保持一致）
    _, eval_transform = build_transforms(
        img_size=args.img_size,
        use_strong_aug=False,
    )

    # 通道配置
    use_extra, use_fft, use_grad, in_channels = resolve_channel_mode(args.channel_mode)

    # 模型
    model = build_model(args, num_classes=2, input_channels=in_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 准备临时工作目录
    work_root = args.work_dir
    if os.path.exists(work_root):
        shutil.rmtree(work_root)
    os.makedirs(work_root, exist_ok=True)

    # 遍历 benchmark_root 下的每个子集
    subset_names: List[str] = []
    for name in sorted(os.listdir(args.benchmark_root)):
        subset_path = os.path.join(args.benchmark_root, name)
        # 只要真正的目录；跳过 .cache、*.tar.gz 等
        if not os.path.isdir(subset_path):
            continue
        if name.startswith("."):
            continue
        if name.endswith(".tar.gz"):
            continue
        subset_names.append(name)

    if not subset_names:
        print("No valid subsets found under", args.benchmark_root)
        return

    print("Found subsets:", subset_names)

    per_subset_results = []
    all_labels = []
    all_probs = []

    for subset in subset_names:
        print("=" * 60)
        print(f"Evaluating subset: {subset}")
        print("=" * 60)

        view_root, real_count, fake_count = build_flat_two_class_view(
            args.benchmark_root, subset, work_root
        )

        if real_count == 0 or fake_count == 0:
            print(
                f"[WARN] Skip subset {subset}: "
                f"real_count={real_count}, fake_count={fake_count}"
            )
            continue

        # 用和训练时一致的 AIGCDataset 读取我们刚刚拍平的 0_real / 1_fake 目录
        dataset = AIGCDataset(
            root=view_root,
            transform=eval_transform,
            use_strong_aug=False,
            use_extra_channels=use_extra,
            use_fft=use_fft,
            use_grad=use_grad,
            highpass_only=args.fft_highpass_only,
            low_cut_ratio=args.fft_low_cut_ratio,
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        subset_labels = []
        subset_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = tta_predict(model, images, args.use_tta)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)

                subset_labels.append(labels.cpu().numpy())
                subset_probs.append(probs.cpu().numpy())

        if len(subset_labels) == 0:
            print(f"[WARN] Subset {subset} produced no samples, skip.")
            continue

        subset_labels = np.concatenate(subset_labels, axis=0)
        subset_probs = np.concatenate(subset_probs, axis=0)
        subset_preds = (subset_probs >= args.threshold).astype(int)

        metrics = compute_binary_metrics(
            subset_labels, subset_preds, subset_probs, args.threshold
        )
        metrics["subset"] = subset
        per_subset_results.append(metrics)

        all_labels.append(subset_labels)
        all_probs.append(subset_probs)

        print(
            f"{subset}: "
            f"n={metrics['num_samples']}, "
            f"acc={metrics['accuracy']:.4f}, "
            f"prec={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"tp={metrics['tp']}, tn={metrics['tn']}, "
            f"fp={metrics['fp']}, fn={metrics['fn']}"
        )

    if not per_subset_results:
        print("No valid subsets evaluated.")
        return

    # 全局（所有子集拼在一起）的指标
    all_labels_cat = np.concatenate(all_labels, axis=0)
    all_probs_cat = np.concatenate(all_probs, axis=0)
    all_preds_cat = (all_probs_cat >= args.threshold).astype(int)
    global_metrics = compute_binary_metrics(
        all_labels_cat, all_preds_cat, all_probs_cat, args.threshold
    )
    global_metrics["subset"] = "GLOBAL"

    # 简单“子集平均 acc”（每个子集 acc 等权平均）
    mean_acc = float(
        np.mean([m["accuracy"] for m in per_subset_results])
    )

    print("=" * 60)
    print(
        f"GLOBAL: n={global_metrics['num_samples']}, "
        f"acc={global_metrics['accuracy']:.4f}, "
        f"prec={global_metrics['precision']:.4f}, "
        f"recall={global_metrics['recall']:.4f}, "
        f"f1={global_metrics['f1']:.4f}, "
        f"tp={global_metrics['tp']}, tn={global_metrics['tn']}, "
        f"fp={global_metrics['fp']}, fn={global_metrics['fn']}"
    )
    print(f"Mean subset accuracy (unweighted): {mean_acc:.4f}")
    print("=" * 60)

    # 写 CSV
    csv_path = args.output_csv or default_csv_path(args.checkpoint)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "subset",
            "num_samples",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tp",
            "tn",
            "fp",
            "fn",
            "pos_rate",
        ]
        writer.writerow(header)

        for m in per_subset_results:
            writer.writerow([
                m["subset"],
                m["num_samples"],
                m["threshold"],
                m["accuracy"],
                m["precision"],
                m["recall"],
                m["f1"],
                m["tp"],
                m["tn"],
                m["fp"],
                m["fn"],
                m["pos_rate"],
            ])

        # 加一行 GLOBAL
        writer.writerow([
            "GLOBAL",
            global_metrics["num_samples"],
            global_metrics["threshold"],
            global_metrics["accuracy"],
            global_metrics["precision"],
            global_metrics["recall"],
            global_metrics["f1"],
            global_metrics["tp"],
            global_metrics["tn"],
            global_metrics["fp"],
            global_metrics["fn"],
            global_metrics["pos_rate"],
        ])

        # 再加一行 MEAN_SUBSETS，只填我们关心的平均 acc
        writer.writerow([
            "MEAN_SUBSETS",
            "",
            "",
            mean_acc,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ])

    print(f"Per-subset and overall metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()

