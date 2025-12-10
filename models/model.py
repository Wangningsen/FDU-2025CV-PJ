import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels, stride=1)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.skip = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.skip is not None:
            identity = self.skip(identity)
        out = out + identity
        out = nn.functional.relu(out, inplace=True)
        return out


def make_layer(in_channels, out_channels, num_blocks, stride):
    layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.constant_(m.bias, 0.0)


class AIGCNetSmall(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.output_dim = 256
        self.stem = ConvBlock(in_channels, 32, stride=1)
        self.layer1 = make_layer(32, 64, num_blocks=2, stride=2)
        self.layer2 = make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = make_layer(128, 256, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = (
            nn.Linear(self.output_dim, num_classes)
            if num_classes is not None
            else None
        )
        self.apply(init_weights)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        if self.classifier is None:
            return feat
        return self.classifier(feat)


class AIGCNetLarge(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.output_dim = 512
        self.stem = ConvBlock(in_channels, 64, stride=1)
        self.layer1 = make_layer(64, 128, num_blocks=3, stride=2)
        self.layer2 = make_layer(128, 256, num_blocks=3, stride=2)
        self.layer3 = make_layer(256, 512, num_blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = (
            nn.Linear(self.output_dim, num_classes)
            if num_classes is not None
            else None
        )
        self.apply(init_weights)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        if self.classifier is None:
            return feat
        return self.classifier(feat)


class TwoBranchAIGCNet(nn.Module):
    def __init__(self, backbone_size="small", num_classes=2):
        super().__init__()
        assert backbone_size in ["small", "large"]
        backbone_cls = AIGCNetSmall if backbone_size == "small" else AIGCNetLarge
        self.rgb_backbone = backbone_cls(in_channels=3, num_classes=None)
        self.grad_backbone = backbone_cls(in_channels=2, num_classes=None)
        feature_dim = self.rgb_backbone.output_dim
        self.classifier = nn.Linear(feature_dim * 2, num_classes)

    def forward(self, x):
        x_rgb = x[:, :3]
        x_grad = x[:, 3:5]
        feat_rgb = self.rgb_backbone.forward_features(x_rgb)
        feat_grad = self.grad_backbone.forward_features(x_grad)
        feat = torch.cat([feat_rgb, feat_grad], dim=1)
        return self.classifier(feat)


def create_model(
    in_channels=6,
    num_classes=2,
    model_size="small",
    two_branch=False,
):
    if two_branch:
        return TwoBranchAIGCNet(backbone_size=model_size, num_classes=num_classes)
    backbone = AIGCNetSmall if model_size == "small" else AIGCNetLarge
    return backbone(in_channels=in_channels, num_classes=num_classes)
