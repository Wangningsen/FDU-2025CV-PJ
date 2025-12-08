import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import AIGCDataset, build_transforms
from models.model import create_model
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
        help="Which channels to use for ablation",
    )
    return parser.parse_args()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
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
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
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

    log_path = os.path.join(args.save_dir, "train.log")
    train_root = os.path.join(args.data_dir, "train")

    if args.channel_mode == "rgb":
        use_extra_channels = False
        use_fft = False
        use_grad = False
        in_channels = 3
    elif args.channel_mode == "rgb_fft":
        use_extra_channels = True
        use_fft = True
        use_grad = False
        in_channels = 4
    elif args.channel_mode == "rgb_grad":
        use_extra_channels = True
        use_fft = False
        use_grad = True
        in_channels = 5
    else:  # "rgb_fft_grad"
        use_extra_channels = True
        use_fft = True
        use_grad = True
        in_channels = 6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform, eval_transform = build_transforms(img_size=args.img_size)

    full_dataset = AIGCDataset(
        root=train_root,
        transform=train_transform,
        use_extra_channels=use_extra_channels,
        use_fft=use_fft,
        use_grad=use_grad,
    )

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # 验证集要用 eval_transform 和相同的 handcrafted 通道
    val_dataset.dataset.transform = eval_transform

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

    model = create_model(in_channels=in_channels, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

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
            )

            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device,
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


if __name__ == "__main__":
    main()
