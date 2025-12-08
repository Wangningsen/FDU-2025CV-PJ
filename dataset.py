import os
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
) -> torch.Tensor:
    """
    输入: x, (3, H, W), [0,1]
    输出: 按配置拼接额外通道:
      - RGB 必有
      - use_fft: 加 1 个 FFT 幅度通道
      - use_grad: 加 2 个梯度通道 Gx, Gy
    """
    assert x.ndim == 3 and x.shape[0] == 3

    r, g, b = x[0], x[1], x[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b  # (H, W)
    gray = gray.unsqueeze(0)  # (1, H, W)

    channels = [x]  # 先放 RGB

    if use_fft:
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        mag = torch.abs(fft_shift)
        mag = torch.log1p(mag)
        mag = mag - mag.min()
        if mag.max() > 0:
            mag = mag / mag.max()
        channels.append(mag)  # (1, H, W)

    if use_grad:
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        gray_nchw = gray.unsqueeze(0)  # (1, 1, H, W)
        gx = F.conv2d(gray_nchw, sobel_x, padding=1).squeeze(0)  # (1, H, W)
        gy = F.conv2d(gray_nchw, sobel_y, padding=1).squeeze(0)  # (1, H, W)

        channels.append(gx)
        channels.append(gy)

    out = torch.cat(channels, dim=0)
    return out



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
        use_extra_channels: bool = True,
        use_fft: bool = True,
        use_grad: bool = True,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.use_extra_channels = use_extra_channels
        self.use_fft = use_fft
        self.use_grad = use_grad

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
        use_extra_channels: bool = True,
        use_fft: bool = True,
        use_grad: bool = True,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.use_extra_channels = use_extra_channels
        self.use_fft = use_fft
        self.use_grad = use_grad

        files = [
            f
            for f in os.listdir(root)
            if is_image_file(f) and os.path.isfile(os.path.join(root, f))
        ]
        files.sort()
        self.files: List[str] = files

        if len(self.files) == 0:
            raise RuntimeError(f"No images found in test dir {root}")

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
            )

        return img, fname


def build_transforms(img_size: int = 256):
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
