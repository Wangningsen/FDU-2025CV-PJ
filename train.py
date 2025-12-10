import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from dataset import AIGCDataset, build_transforms
from models import AIGCNetSmall, AIGCNetLarge, TwoBranchAIGCNet
from utils import set_seed, ensure_dir, accuracy, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AIGC image classifier from scratch",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Root dir of dataset, contains train/ and test/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./weights",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of training data used as validation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (HxW)",
    )
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="rgb_fft_grad",
        choices=["rgb", "rgb_fft", "rgb_grad", "rgb_fft_grad"],
        help="Channels for ablation: rgb / rgb_fft / rgb_grad / rgb_fft_grad",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Backbone size: small (default) or large.",
    )
    parser.add_argument(
        "--two_branch",
        action="store_true",
        help="If set, use a two-branch RGB + Sobel model instead of a single-branch model.",
    )
    parser.add_argument(
        "--use_label_smoothing",
        action="store_true",
        help="If set, enable label smoothing for classification loss.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Epsilon value for label smoothing (only used if --use_label_smoothing).",
    )
    parser.add_argument(
        "--use_mixup",
        action="store_true",
        help="If set, enable Mixup during training.",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
        help="Alpha parameter for Mixup Beta distribution.",
    )
    parser.add_argument(
        "--use_strong_aug",
        action="store_true",
        help="If set, enable stronger data augmentations (crop, blur, JPEG, grayscale).",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="If set, enable simple test-time augmentation (horizontal flip).",
    )
    parser.add_argument(
        "--ensemble_from",
        type=str,
        default="",
        help=(
            "Optional: comma-separated list of checkpoint paths to ensemble at inference. "
            "If non-empty, load these models and average their logits."
        ),
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log training to Weights and Biases",
    )
    parser.add_argument(
        "--fft_highpass_only",
        action="store_true",
        help="Use only high frequency components in FFT channel",
    )
    parser.add_argument(
        "--fft_low_cut_ratio",
        type=float,
        default=0.1,
        help="Low frequency radius ratio to cut when using highpass FFT",
    )
    return parser.parse_args()


def resolve_channel_mode(mode: str):
    if mode == "rgb":
        return False, False, False, 3
    if mode == "rgb_fft":
        return True, True, False, 4
    if mode == "rgb_grad":
        return True, False, True, 5
    return True, True, True, 6


def smooth_labels(targets, num_classes, smoothing):
    confidence = 1.0 - smoothing
    off_value = smoothing / (num_classes - 1)
    one_hot = torch.full(
        (targets.size(0), num_classes),
        off_value,
        device=targets.device,
    )
    one_hot.scatter_(1, targets.unsqueeze(1), confidence)
    return one_hot


def mixup_data(x, y, alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def tta_predict(model, x):
    logits1 = model(x)
    logits2 = model(torch.flip(x, dims=[3]))
    return (logits1 + logits2) / 2.0


def predict_logits(model, x, use_tta):
    if isinstance(model, (list, tuple)):
        logits_sum = None
        for m in model:
            preds = tta_predict(m, x) if use_tta else m(x)
            logits_sum = preds if logits_sum is None else logits_sum + preds
        return logits_sum / float(len(model))
    return tta_predict(model, x) if use_tta else model(x)


def build_model(args, num_classes, input_channels):
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
        f"Model: {model.__class__.__name__}, size={args.model_size}, "
        f"two_branch={args.two_branch}, params={n_params / 1e6:.2f}M"
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if args.use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                images, labels, args.mixup_alpha
            )
            outputs = model(inputs)
            if args.use_label_smoothing:
                num_classes = outputs.size(1)
                soft_a = smooth_labels(
                    targets_a, num_classes, args.label_smoothing
                )
                soft_b = smooth_labels(
                    targets_b, num_classes, args.label_smoothing
                )
                soft_labels = lam * soft_a + (1 - lam) * soft_b
                log_probs = F.log_softmax(outputs, dim=1)
                loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:
                loss = (
                    lam * criterion(outputs, targets_a)
                    + (1 - lam) * criterion(outputs, targets_b)
                )
        else:
            outputs = model(images)
            if args.use_label_smoothing:
                num_classes = outputs.size(1)
                soft_labels = smooth_labels(
                    labels, num_classes, args.label_smoothing
                )
                log_probs = F.log_softmax(outputs, dim=1)
                loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:
                loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, labels) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    if isinstance(model, (list, tuple)):
        for m in model:
            m.eval()
    else:
        model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = predict_logits(model, images, args.use_tta)
        if args.use_label_smoothing:
            num_classes = outputs.size(1)
            soft_labels = smooth_labels(
                labels, num_classes, args.label_smoothing
            )
            log_probs = F.log_softmax(outputs, dim=1)
            loss = -(soft_labels * log_probs).sum(dim=1).mean()
        else:
            loss = criterion(outputs, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, labels) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)

    if args.two_branch and args.channel_mode != "rgb_grad":
        print("Two-branch enabled: forcing channel_mode to rgb_grad for RGB+Sobel.")
        args.channel_mode = "rgb_grad"

    log_path = os.path.join(args.save_dir, "train.log")
    train_root = os.path.join(args.data_dir, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, eval_transform = build_transforms(
        img_size=args.img_size, use_strong_aug=args.use_strong_aug
    )

    use_extra, use_fft, use_grad, in_channels = resolve_channel_mode(
        args.channel_mode
    )

    full_dataset = AIGCDataset(
        root=train_root,
        transform=train_transform,
        use_strong_aug=args.use_strong_aug,
        use_extra_channels=use_extra,
        use_fft=use_fft,
        use_grad=use_grad,
        highpass_only=args.fft_highpass_only,
        low_cut_ratio=args.fft_low_cut_ratio,
    )

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    val_base_dataset = AIGCDataset(
        root=train_root,
        transform=eval_transform,
        use_extra_channels=use_extra,
        use_fft=use_fft,
        use_grad=use_grad,
        highpass_only=args.fft_highpass_only,
        low_cut_ratio=args.fft_low_cut_ratio,
    )
    val_dataset = torch.utils.data.Subset(val_base_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, num_classes=2, input_channels=in_channels)
    model.to(device)

    print(
        f"Config: model_size={args.model_size}, two_branch={args.two_branch}, "
        f"use_label_smoothing={args.use_label_smoothing}, "
        f"label_smoothing={args.label_smoothing}, use_mixup={args.use_mixup}, "
        f"mixup_alpha={args.mixup_alpha}, use_strong_aug={args.use_strong_aug}, "
        f"use_tta={args.use_tta}, ensemble_from='{args.ensemble_from}', "
        f"channel_mode={args.channel_mode}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    run = None
    if args.use_wandb:
        project = os.getenv("WANDB_PROJECT", "aigc-det")
        entity = os.getenv("WANDB_ENTITY", None)
        run = wandb.init(
            project=project,
            entity=entity,
            config=vars(args),
        )
        wandb.watch(model, log="gradients", log_freq=100)

    best_val_acc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")

    with open(log_path, "w", encoding="utf-8") as f_log:
        header = (
            "epoch,train_loss,train_acc,val_loss,val_acc,lr,elapsed_sec\n"
        )
        f_log.write(header)
        f_log.flush()

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args,
            )

            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
                args,
            )

            scheduler.step()
            elapsed = time.time() - start_time
            lr_now = scheduler.get_last_lr()[0]

            log_line = (
                f"{epoch},{train_loss:.6f},{train_acc:.4f},"
                f"{val_loss:.6f},{val_acc:.4f},{lr_now:.6e},{elapsed:.2f}\n"
            )
            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} "
                f"lr={lr_now:.2e} time={elapsed:.1f}s"
            )
            f_log.write(log_line)
            f_log.flush()

            if args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "lr": lr_now,
                        "time/epoch_sec": elapsed,
                    }
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc": val_acc,
                    },
                    best_model_path,
                )

        print(
            f"Training finished. Best val acc {best_val_acc:.4f} at epoch {best_epoch}"
        )
        print(f"Best model saved to {best_model_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
