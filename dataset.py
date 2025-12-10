import io
import os
import random
from typing import List, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith(IMG_EXTENSIONS)


# dataset.py

def compute_handcrafted_channels(
    x: torch.Tensor,
    use_fft: bool = True,
    use_grad: bool = True,
    highpass_only: bool = False,
    low_cut_ratio: float = 0.1,
) -> torch.Tensor:
    """
    x: (3, H, W), [0,1]
    返回: RGB + [可选 FFT 高频] + [可选 Sobel Gx, Gy]
    highpass_only=True 时, 只保留高频, 把频谱中心的一块低频置零
    """
    assert x.ndim == 3 and x.shape[0] == 3

    r, g, b = x[0], x[1], x[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b  # (H, W)
    gray = gray.unsqueeze(0)  # (1, H, W)

    channels = [x]

    if use_fft:
        # 全局 FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)

        if highpass_only:
            # 在频谱中心挖一个低频“洞”
            _, H, W = fft_shift.shape
            cy, cx = H // 2, W // 2
            radius = int(low_cut_ratio * min(H, W))

            y1 = max(cy - radius, 0)
            y2 = min(cy + radius, H)
            x1 = max(cx - radius, 0)
            x2 = min(cx + radius, W)

            fft_shift[:, y1:y2, x1:x2] = 0

        mag = torch.abs(fft_shift)
        mag = torch.log1p(mag)  # 压动态范围

        mag = mag - mag.min()
        if mag.max() > 0:
            mag = mag / mag.max()

        channels.append(mag)  # (1, H, W)

    if use_grad:
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32,
            device=gray.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32,
            device=gray.device,
        ).view(1, 1, 3, 3)

        gray_nchw = gray.unsqueeze(0)  # (1, 1, H, W)
        gx = F.conv2d(gray_nchw, sobel_x, padding=1).squeeze(0)  # (1, H, W)
        gy = F.conv2d(gray_nchw, sobel_y, padding=1).squeeze(0)  # (1, H, W)

        channels.append(gx)
        channels.append(gy)

    out = torch.cat(channels, dim=0)
    return out


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



class AIGCDataset(Dataset):
    """
    用于 train/val 的有标签数据集

    期望目录结构:
    root/
      0_real/
      1_fake/
    """

    def __init__(
        self,
        root: str,
        transform=None,
        use_strong_aug: bool = False,
        use_extra_channels: bool = True,
        use_fft: bool = True,
        use_grad: bool = True,
        highpass_only: bool = True,
        low_cut_ratio: float = 0.1,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.use_strong_aug = use_strong_aug
        self.use_extra_channels = use_extra_channels
        self.use_fft = use_fft
        self.use_grad = use_grad
        self.highpass_only = highpass_only
        self.low_cut_ratio = low_cut_ratio

        self.samples: List[Tuple[str, int]] = []
        for sub in ["0_real", "1_fake"]:
            label = 0 if sub.startswith("0") else 1
            subdir = os.path.join(root, sub)
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                if is_image_file(fname):
                    path = os.path.join(subdir, fname)
                    self.samples.append((path, label))

        self.samples.sort(key=lambda t: t[0])

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")

        if self.transform is None:
            train_transform, _ = build_transforms(use_strong_aug=use_strong_aug)
            self.transform = train_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)  # (3, H, W)

        if self.use_extra_channels:
            img = compute_handcrafted_channels(
                img,
                use_fft=self.use_fft,
                use_grad=self.use_grad,
                highpass_only=self.highpass_only,
                low_cut_ratio=self.low_cut_ratio,
            )
        return img, label


class AIGCTestDataset(Dataset):
    """
    用于 test 的无标签数据，返回 (tensor, image_id)
    期望目录: root/ 下全是图像文件
    """

    def __init__(
        self,
        root: str,
        transform=None,
        use_strong_aug: bool = False,
        use_extra_channels: bool = True,
        use_fft: bool = True,
        use_grad: bool = True,
        highpass_only: bool = True,
        low_cut_ratio: float = 0.1,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.use_strong_aug = use_strong_aug
        self.use_extra_channels = use_extra_channels
        self.use_fft = use_fft
        self.use_grad = use_grad
        self.highpass_only = highpass_only
        self.low_cut_ratio = low_cut_ratio

        files = [
            f
            for f in os.listdir(root)
            if is_image_file(f) and os.path.isfile(os.path.join(root, f))
        ]
        files.sort()
        self.files: List[str] = files

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in test dir {root}")

        if self.transform is None:
            _, eval_transform = build_transforms()
            self.transform = eval_transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        path = os.path.join(self.root, fname)
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.use_extra_channels:
            img = compute_handcrafted_channels(
                img,
                use_fft=self.use_fft,
                use_grad=self.use_grad,
                highpass_only=self.highpass_only,
                low_cut_ratio=self.low_cut_ratio,
            )

        return img, fname


def build_strong_train_transform(img_size=256):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
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
        ]
    )


def build_transforms(img_size=256, use_strong_aug=False):
    if use_strong_aug:
        train_transform = build_strong_train_transform(img_size=img_size)
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05,
                ),
                transforms.ToTensor(),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    return train_transform, eval_transform
