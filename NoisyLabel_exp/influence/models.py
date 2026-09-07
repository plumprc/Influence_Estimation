"""CIFAR models used by the noisy-label influence experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, resnet50, vgg19_bn


MODEL_NAMES = ("resnet18", "resnet50", "vgg19_bn", "mobilenetv2")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return F.relu(out)


class ResNet18CIFAR(nn.Module):
    """CIFAR-style ResNet-18 with a 3x3 stem and no initial max-pool."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.fc(features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def make_resnet18_cifar10() -> ResNet18CIFAR:
    return ResNet18CIFAR(num_classes=10)


def make_resnet50_cifar10() -> torch.nn.Module:
    """CIFAR-style ResNet-50 with a 3x3 stem and no initial max-pool."""
    model = resnet50(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_vgg19_bn_cifar10() -> torch.nn.Module:
    """CIFAR-style VGG-19-BN with a compact classifier.

    The torchvision classifier retains two 4096-unit layers and has over 120M
    parameters. CIFAR VGG implementations commonly use a single linear head,
    which gives a model whose parameter count is comparable to ResNet-50.
    """
    model = vgg19_bn(num_classes=10)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, 10)
    # VGG uses a flat Conv/BN/ReLU sequence rather than named blocks.
    model.influence_final_block_prefixes = (
        "features.49.",
        "features.50.",
        "classifier.",
    )
    return model


def make_mobilenetv2_cifar10() -> torch.nn.Module:
    """CIFAR-style MobileNetV2 with stride-1 stem convolution."""
    model = mobilenet_v2(num_classes=10)
    model.features[0][0].stride = (1, 1)
    return model


def make_model(name: str) -> torch.nn.Module:
    factories = {
        "resnet18": make_resnet18_cifar10,
        "resnet50": make_resnet50_cifar10,
        "vgg19_bn": make_vgg19_bn_cifar10,
        "mobilenetv2": make_mobilenetv2_cifar10,
    }
    if name not in factories:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
    return factories[name]()
