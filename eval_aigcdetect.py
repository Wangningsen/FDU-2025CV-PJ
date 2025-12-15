import argparse
import os
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AIGCDataset, build_transforms
from models import AIGCNetSmall, AIGCNetLarge, TwoBranchAIGCNet
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on an AIGCDetect-style benchmark. "
                    "Each subfolder under benchmark_dir has 0_real/ and 1_fake/."
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        required=True,
        help="Root dir of benchmark, e.g. /path/to/AIGCDetectBenchmark",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. weights/extra_twobranch_small_mixup_only/best_model.pth",
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
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed (for any internal shuffling).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help=(
            "Optional output csv path. If empty, will use "
            "<ckpt_dir>/<run_name>_<benchmark_name>.csv"
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


def default_csv_path(checkpoint_path: str, benchmark_dir: str) -> str:
    ckpt_path = checkpoint_path.rstrip("/\\")
    ckpt_dir = os.path.dirname(ckpt_path)
    run_name = os.path.basename(ckpt_dir)
    bench_name = os.path.basename(os.path.normpath(benchmark_dir))
    return os.path.join(ckpt_dir, f"{run_name}_{bench_name}.csv")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # eval transform
    _, eval_transform = build_transforms(
        img_size=args.img_size,
        use_strong_aug=False,
    )

    # channel config
    use_extra, use_fft, use_grad, in_channels = resolve_channel_mode(
        args.channel_mode
    )

    # build model and load checkpoint
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

    benchmark_root = args.benchmark_dir
    if not os.path.isdir(benchmark_root):
        raise ValueError(f"benchmark_dir not found: {benchmark_root}")

    subset_names = [
        d for d in sorted(os.listdir(benchmark_root))
        if os.path.isdir(os.path.join(benchmark_root, d))
    ]
    if not subset_names:
        raise ValueError(f"No subfolders found under benchmark_dir: {benchmark_root}")

    print("Subsets to evaluate:")
    for name in subset_names:
        print(f"  - {name}")

    subset_results = []
    global_labels = []
    global_probs = []

    for subset in subset_names:
        subset_root = os.path.join(benchmark_root, subset)
        print("=" * 60)
        print(f"Evaluating subset: {subset}")
        print(f"Root: {subset_root}")

        # 每个子文件夹内部是 0_real 和 1_fake
        try:
            dataset = AIGCDataset(
                root=subset_root,
                transform=eval_transform,
                use_strong_aug=False,
                use_extra_channels=use_extra,
                use_fft=use_fft,
                use_grad=use_grad,
                highpass_only=args.fft_highpass_only,
                low_cut_ratio=args.fft_low_cut_ratio,
            )
        except RuntimeError as e:
            print(f"[WARN] Skipping subset {subset} due to error: {e}")
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = predict_logits(model, images, args.use_tta)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        if len(all_labels) == 0:
            print(f"[WARN] Subset {subset} has no samples after loading, skip")
            continue

        labels = np.concatenate(all_labels, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        preds = (probs >= args.threshold).astype(int)

        metrics = compute_binary_metrics(labels, preds, probs, args.threshold)
        subset_results.append((subset, metrics))

        # 累计到全局
        global_labels.append(labels)
        global_probs.append(probs)

        print(
            f"[{subset}] acc={metrics['accuracy']:.4f}, "
            f"prec={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"tp={metrics['tp']}, tn={metrics['tn']}, "
            f"fp={metrics['fp']}, fn={metrics['fn']}, "
            f"num_samples={metrics['num_samples']}"
        )

    if not subset_results:
        print("No valid subsets evaluated, exiting.")
        return

    # 1) 子集层面的宏平均 acc
    mean_acc = float(np.mean([m["accuracy"] for _, m in subset_results]))
    print("=" * 60)
    print(
        f"Macro mean accuracy across {len(subset_results)} subsets: "
        f"{mean_acc:.4f}"
    )

    # 2) 也给一个全局 micro 指标（所有样本拼起来）
    global_labels = np.concatenate(global_labels, axis=0)
    global_probs = np.concatenate(global_probs, axis=0)
    global_preds = (global_probs >= args.threshold).astype(int)
    global_metrics = compute_binary_metrics(
        global_labels, global_preds, global_probs, args.threshold
    )
    print(
        f"Global (micro) metrics over all subsets: "
        f"acc={global_metrics['accuracy']:.4f}, "
        f"prec={global_metrics['precision']:.4f}, "
        f"recall={global_metrics['recall']:.4f}, "
        f"f1={global_metrics['f1']:.4f}, "
        f"tp={global_metrics['tp']}, tn={global_metrics['tn']}, "
        f"fp={global_metrics['fp']}, fn={global_metrics['fn']}, "
        f"num_samples={global_metrics['num_samples']}"
    )

    # 写 CSV
    csv_path = args.output_csv or default_csv_path(args.checkpoint, args.benchmark_dir)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )
        for subset, m in subset_results:
            writer.writerow(
                [
                    subset,
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
                ]
            )

        # 最后一行写宏平均和全局信息
        writer.writerow(
            [
                "MACRO_MEAN",
                "",
                args.threshold,
                mean_acc,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "GLOBAL_MICRO",
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
            ]
        )

    print(f"Per subset metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
