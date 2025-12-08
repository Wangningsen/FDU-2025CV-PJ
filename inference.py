import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from dataset import AIGCTestDataset, build_transforms
from models.model import create_model


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
        required=True,
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
    return parser.parse_args()


@torch.no_grad()
def run_inference(
    model,
    loader,
    device,
):
    model.eval()
    all_results = []

    for images, image_ids in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)  # 0 or 1

        preds = preds.cpu().numpy().tolist()
        for img_id, label in zip(image_ids, preds):
            all_results.append((img_id, int(label)))

    return all_results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, eval_transform = build_transforms(img_size=args.img_size)

    test_root = os.path.join(args.data_dir, "test")
    test_dataset = AIGCTestDataset(
        root=test_root,
        transform=eval_transform,
        use_extra_channels=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
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
    else:
        use_extra_channels = True
        use_fft = True
        use_grad = True
        in_channels = 6

    test_dataset = AIGCTestDataset(
        root=test_root,
        transform=eval_transform,
        use_extra_channels=use_extra_channels,
        use_fft=use_fft,
        use_grad=use_grad,
    )

    model = create_model(in_channels=in_channels, num_classes=2)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    results = run_inference(model, test_loader, device)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        for image_id, label in results:
            writer.writerow([image_id, label])

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
