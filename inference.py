import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from dataset import AIGCTestDataset, build_transforms
from models import AIGCNetSmall, AIGCNetLarge, TwoBranchAIGCNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for AIGC classifier",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Root dir of dataset, contains test/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to best_model.pth",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./result.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
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
        help="Which channels to use, must match training",
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
    return parser.parse_args()


def resolve_channel_mode(mode: str):
    if mode == "rgb":
        return False, False, False, 3
    if mode == "rgb_fft":
        return True, True, False, 4
    if mode == "rgb_grad":
        return True, False, True, 5
    return True, True, True, 6


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


def build_model(args, in_channels, num_classes=2):
    if args.two_branch:
        return TwoBranchAIGCNet(
            backbone_size=args.model_size,
            num_classes=num_classes,
        )
    backbone_cls = AIGCNetSmall if args.model_size == "small" else AIGCNetLarge
    return backbone_cls(in_channels=in_channels, num_classes=num_classes)


def load_models(args, device, in_channels):
    model_paths = []
    if args.ensemble_from:
        model_paths = [p.strip() for p in args.ensemble_from.split(",") if p.strip()]
    if not model_paths:
        if not args.checkpoint:
            raise ValueError("Please provide --checkpoint or --ensemble_from paths.")
        model_paths = [args.checkpoint]

    models = []
    for path in model_paths:
        model = build_model(args, in_channels=in_channels)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def run_inference(models, loader, device, use_tta):
    for m in models:
        m.eval()
    ref = models if len(models) > 1 else models[0]
    all_results = []

    for images, image_ids in loader:
        images = images.to(device, non_blocking=True)
        outputs = predict_logits(ref, images, use_tta)
        preds = outputs.argmax(dim=1)

        preds = preds.cpu().numpy().tolist()
        for img_id, label in zip(image_ids, preds):
            all_results.append((img_id, int(label)))

    return all_results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.two_branch and args.channel_mode != "rgb_grad":
        print("Two-branch enabled: forcing channel_mode to rgb_grad for RGB+Sobel.")
        args.channel_mode = "rgb_grad"

    _, eval_transform = build_transforms(img_size=args.img_size)

    use_extra_channels, use_fft, use_grad, in_channels = resolve_channel_mode(
        args.channel_mode
    )

    test_root = os.path.join(args.data_dir, "test")
    test_dataset = AIGCTestDataset(
        root=test_root,
        transform=eval_transform,
        use_extra_channels=use_extra_channels,
        use_fft=use_fft,
        use_grad=use_grad,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    models = load_models(args, device, in_channels)
    results = run_inference(models, test_loader, device, args.use_tta)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for image_id, label in results:
            writer.writerow([image_id, label])

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
