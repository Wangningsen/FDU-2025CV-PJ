# export_val_for_npr.py
import os
import shutil
import torch

from torch.utils.data import random_split
from dataset import AIGCDataset, build_transforms
from utils import set_seed

# 你的原始数据路径
DATA_DIR = "/data1/nwang60/datasets/CV-Dataset"
TRAIN_ROOT = os.path.join(DATA_DIR, "train")

# 给 NPR 用的新目录结构：
#   <OUT_PARENT>/
#       ours/
#           0_real/
#           1_fake/
OUT_PARENT = "/data1/nwang60/datasets/NPR_ValForCVPJ"
OUT_ROOT = os.path.join(OUT_PARENT, "ours")

IMG_SIZE = 256
VAL_RATIO = 0.1
SEED = 42


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "0_real"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "1_fake"), exist_ok=True)

    # transform 对 samples 的列表没有影响, 随便给个就好
    _, eval_transform = build_transforms(img_size=IMG_SIZE, use_strong_aug=False)

    full_dataset = AIGCDataset(
        root=TRAIN_ROOT,
        transform=eval_transform,
        use_strong_aug=False,
        use_extra_channels=False,  # 只要路径和 label
        use_fft=False,
        use_grad=False,
        highpass_only=False,
        low_cut_ratio=0.1,
    )

    set_seed(SEED)
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size
    train_set, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"Total: {len(full_dataset)}, train: {len(train_set)}, val: {len(val_subset)}")
    print("Exporting val images to", OUT_ROOT)

    for idx in val_subset.indices:
        path, label = full_dataset.samples[idx]
        cls = "0_real" if label == 0 else "1_fake"
        dst_dir = os.path.join(OUT_ROOT, cls)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(path))
        shutil.copy2(path, dst_path)

    print("Done. Val set for NPR is at:", OUT_ROOT)
    print("Structure:")
    print(OUT_ROOT)
    print(" ├─ 0_real/")
    print(" └─ 1_fake/")


if __name__ == "__main__":
    main()
