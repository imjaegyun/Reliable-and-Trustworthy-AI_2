from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


class NormalizeLayer(nn.Module):
    def __init__(self, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class CIFAR10ResNet50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Use a CIFAR-10 normalization layer and a 32x32-friendly ResNet stem.
        self.normalize = NormalizeLayer(CIFAR10_MEAN, CIFAR10_STD)
        self.backbone = resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(CIFAR10_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))


def build_model(device: torch.device | str = "cpu") -> CIFAR10ResNet50:
    model = CIFAR10ResNet50()
    return model.to(device)


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> CIFAR10ResNet50:
    model = build_model(device)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def save_checkpoint(path: str, model: nn.Module, seed: int, epoch: int, accuracy: float) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "epoch": epoch,
            "accuracy": accuracy,
            "architecture": "CIFAR10ResNet50",
        },
        path,
    )
