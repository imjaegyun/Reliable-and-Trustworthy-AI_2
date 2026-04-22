from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models import build_model, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../../assignment1/data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--test-limit", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_name: str) -> torch.device:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = torch.cuda.is_available()
    if device_name == "auto":
        return torch.device("cuda" if cuda_available else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return device


def subset_dataset(dataset: torch.utils.data.Dataset, limit: int | None, seed: int) -> torch.utils.data.Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def make_loaders(
    args: argparse.Namespace,
    repo_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    data_dir = resolve_path(repo_dir, args.data_dir)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=args.download,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=args.download,
        transform=test_transform,
    )
    train_set = subset_dataset(train_set, args.train_limit, seed)
    test_set = subset_dataset(test_set, args.test_limit, seed + 1000)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def train_one(seed: int, args: argparse.Namespace, repo_dir: Path) -> Path:
    # Different seeds produce independently trained models for differential testing.
    set_seed(seed)
    device = resolve_device(args.device)
    print(f"device={device}")
    train_loader, test_loader = make_loaders(args, repo_dir, seed, device)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_accuracy = 0.0
    models_dir = resolve_path(repo_dir, args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / f"resnet50_cifar10_seed{seed}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"seed {seed} epoch {epoch}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        accuracy = evaluate(model, test_loader, device)
        print(f"seed={seed} epoch={epoch} accuracy={accuracy:.4f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(str(output_path), model, seed, epoch, best_accuracy)

    return output_path


def main() -> int:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    for seed in args.seeds:
        output_path = train_one(seed, args, repo_dir)
        print(f"saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
