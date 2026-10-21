"""
models.py
---------
Model architectures for TrashNet waste classification:
1. BaselineCNN: Custom lightweight 4-stage convolutional network built from scratch with Kaiming initialization.
2. TransferMobileNet: Pretrained MobileNetV3-Small optimized for edge deployment / low latency.
3. TransferResNet: Pretrained ResNet-18 standard robust vision backbone.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU + MaxPool block."""
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class BaselineCNN(nn.Module):
    """
    Custom 4-stage Convolutional Neural Network built from scratch.
    Designed for fast inference, modest parameter footprint, and serving as a true scratch baseline.
    """
    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 224x224 -> 112x112
            ConvBlock(32, 64),   # 112x112 -> 56x56
            ConvBlock(64, 128),  # 56x56 -> 28x28
            ConvBlock(128, 256), # 28x28 -> 14x14
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.pool(feat)
        out = self.classifier(pooled)
        return out


class TransferMobileNet(nn.Module):
    """
    Transfer learning with MobileNetV3-Small.
    High efficiency, ideal for real-world automated smart bins and battery-powered cameras.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.base_model = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

        in_features = self.base_model.classifier[3].in_features
        self.base_model.classifier[3] = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def unfreeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


class TransferResNet(nn.Module):
    """
    Transfer learning with ResNet-18.
    Standard robust residual network with strong feature representation.
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.base_model = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def unfreeze_backbone(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


def get_model(model_name: str, num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """Factory to instantiate models by name."""
    name = model_name.lower()
    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    elif name in ["mobilenet", "mobilenet_v3", "transfer_mobilenet"]:
        return TransferMobileNet(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif name in ["resnet", "resnet18", "transfer_resnet"]:
        return TransferResNet(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: 'baseline', 'mobilenet', 'resnet'.")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
