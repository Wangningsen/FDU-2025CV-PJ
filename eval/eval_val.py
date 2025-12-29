import argparse
import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import AIGCDataset, build_transforms
from models import AIGCNetSmall, AIGCNetLarge, TwoBranchAIGCNet
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the validation split (re-created from train set)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Root dir of dataset, contains train/ and test/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. weights/arch_xxx/best_model.pth",
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
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio used during training. Must match to reproduce the split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the original train/val split.",
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="rgb_grad",
        choices=["rgb", "rgb_fft", "rgb_grad", "rgb_fft_grad"],
        help="Channel mode used at training time.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Backbone size used at training time.",
    )
    parser.add_argument(
        "--two_branch",
        action="store_true",
        help="Use two-branch RGB+Sobel model (must match training).",
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
        "--use_tta",
        action="store_true",
        help="Enable simple test-time augmentation (horizontal flip).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on P(fake) for binary metrics.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help=(
            "Optional output csv path. "
            "If empty, will use <ckpt_dir>/<run_name>.csv, "
            "where run_name is the folder name containing the checkpoint."
        ),
    )
    return parser.parse_args()


def resolve_channel_mode(mode: str):
    """Return (use_extra_channels, use_fft, use_grad, in_channels)."""
    if mode == "rgb":
        return False, False, False, 3
    if mode == "rgb_fft":
        return True, True, False, 4
    if mode == "rgb_grad":
        return True, False, True, 5
    # "rgb_fft_grad"
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
        f"in_channels={input_channels}, params={n_params / 1e6:.2f}M"
    )
    return model


def tta_predict(model, x):
    """Simple TTA: original + horizontal flip, average logits."""
    logits1 = model(x)
    logits2 = model(torch.flip(x, dims=[3]))
    return (logits1 + logits2) / 2.0


def predict_logits(model, x, use_tta: bool):
    if use_tta:
        return tta_predict(model, x)
    return model(x)


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

    accuracy = (tp + tn) / n if n > 0 else 0.0

    prec_den = tp + fp
    precision = tp / prec_den if prec_den > 0 else 0.0

    rec_den = tp + fn
    recall = tp / rec_den if rec_den > 0 else 0.0

    f1_den = precision + recall
    f1 = 2 * precision * recall / f1_den if f1_den > 0 else 0.0

    pos_rate = preds.mean() if n > 0 else 0.0

    metrics = {
        "num_samples": n,
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pos_rate": float(pos_rate),
    }
    return metrics


def default_csv_path(checkpoint_path: str) -> str:
    ckpt_path = checkpoint_path.rstrip("/\\")
    ckpt_dir = os.path.dirname(ckpt_path)
    run_name = os.path.basename(ckpt_dir)
    return os.path.join(ckpt_dir, f"{run_name}.csv")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root = os.path.join(args.data_dir, "train")

    # eval transform
    _, eval_transform = build_transforms(
        img_size=args.img_size,
        use_strong_aug=False,
    )

    # channel config
    use_extra, use_fft, use_grad, in_channels = resolve_channel_mode(
        args.channel_mode
    )

    # 1. 先构造 full_dataset, 按训练时逻辑重建索引
    full_dataset = AIGCDataset(
        root=train_root,
        transform=eval_transform,      # transform 不影响 random_split 的索引
        use_strong_aug=False,
        use_extra_channels=use_extra,
        use_fft=use_fft,
        use_grad=use_grad,
        highpass_only=args.fft_highpass_only,
        low_cut_ratio=args.fft_low_cut_ratio,
    )

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size

    _, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    from torch.utils.data import Subset  # 只是为了明确类型
    val_dataset = Subset(full_dataset, val_subset.indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2. 构造模型并加载权重
    model = build_model(
        args,
        num_classes=2,
        input_channels=in_channels,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. 在验证集上推理
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = predict_logits(model, images, args.use_tta)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if len(all_labels) == 0:
        print("No samples in validation set, exiting.")
        return

    labels = np.concatenate(all_labels, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= args.threshold).astype(int)

    # 4. 计算二分类指标
    metrics = compute_binary_metrics(labels, preds, probs, args.threshold)

    print(
        f"Validation metrics: "
        f"acc={metrics['accuracy']:.4f}, "
        f"prec={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"tp={metrics['tp']}, tn={metrics['tn']}, "
        f"fp={metrics['fp']}, fn={metrics['fn']}"
    )

    # 5. 写 CSV
    csv_path = args.output_csv or default_csv_path(args.checkpoint)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        # 把关键配置也写进去, 方便之后对照
        writer.writerow(["model_size", args.model_size])
        writer.writerow(["two_branch", args.two_branch])
        writer.writerow(["channel_mode", args.channel_mode])
        writer.writerow(["fft_highpass_only", args.fft_highpass_only])
        writer.writerow(["fft_low_cut_ratio", args.fft_low_cut_ratio])
        writer.writerow(["use_tta", args.use_tta])
        writer.writerow(["seed", args.seed])
        writer.writerow(["val_ratio", args.val_ratio])
        writer.writerow(["checkpoint", args.checkpoint])

    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()

