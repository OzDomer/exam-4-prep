#!/usr/bin/env python3
"""
Train an image classifier on YOUR OWN dataset using transfer learning
(ResNet-18 pretrained on ImageNet). It saves a checkpoint (model.pt) that both
predict.py and the web app (main.py) load.

Dataset layout -- either works:

1) Unsplit (the script makes a stratified train/val split for you):
    data/
      classA/  img1.jpg  img2.jpg ...
      classB/  ...

2) Pre-split:
    data/
      train/  classA/...  classB/...
      val/    classA/...  classB/...      ("valid" / "validation" also accepted)

Examples:
    python train.py --data_dir ./data
    python train.py --data_dir ./data --epochs 15 --batch_size 32
    python train.py --data_dir ./data --no-pretrained
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

from model import DEFAULT_MEAN, DEFAULT_STD, choose_device


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Folder containing one sub-folder per class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.pt"),
        help="Where to save the best model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.20,
        help="Validation fraction when the dataset has no val/valid folder.",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialize ResNet-18 with ImageNet weights.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def has_image(path: Path) -> bool:
    try:
        return any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            for p in path.rglob("*")
        )
    except OSError:
        return False


def class_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    ignored = {"train", "training", "val", "valid", "validation", "test", "testing"}
    result = []
    for child in path.iterdir():
        if child.is_dir() and child.name.lower() not in ignored and has_image(child):
            result.append(child)
    return sorted(result)


def descendants(root: Path, max_depth: int = 3) -> Iterable[Path]:
    root = root.resolve()
    yield root
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue
        try:
            children = [p for p in current.iterdir() if p.is_dir()]
        except OSError:
            continue
        for child in children:
            yield child
            stack.append((child, depth + 1))


def first_named_dir(parent: Path, names: tuple[str, ...]) -> Path | None:
    try:
        by_name = {p.name.lower(): p for p in parent.iterdir() if p.is_dir()}
    except OSError:
        return None
    for name in names:
        candidate = by_name.get(name)
        if candidate is not None:
            return candidate
    return None


def locate_dataset(data_dir: Path) -> tuple[Path, Path | None]:
    """Returns (train_or_full_dir, val_dir_or_none)."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    candidates = list(descendants(data_dir, max_depth=3))

    # Prefer an explicit train/val layout.
    for candidate in candidates:
        train_dir = first_named_dir(candidate, ("train", "training"))
        if train_dir is None or len(class_dirs(train_dir)) < 2:
            continue
        val_dir = first_named_dir(candidate, ("val", "valid", "validation"))
        if val_dir is not None and len(class_dirs(val_dir)) < 2:
            val_dir = None
        return train_dir, val_dir

    # Otherwise pick the directory whose children look most like class folders
    # (most images wins). This also skips any extra wrapper folder a dataset
    # zip may have unpacked into.
    scored: list[tuple[int, Path]] = []
    for candidate in candidates:
        classes = class_dirs(candidate)
        if len(classes) < 2:
            continue
        image_count = sum(
            1
            for c in classes
            for p in c.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        scored.append((image_count, candidate))

    if not scored:
        raise RuntimeError(
            "Could not find class folders. Expected either:\n"
            "  data/ClassName/images...\n"
            "or\n"
            "  data/train/ClassName/images...\n"
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], None


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03
            ),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )
    return train_transform, eval_transform


def stratified_indices(targets: list[int], val_fraction: float, seed: int):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val_split must be between 0 and 1.")

    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        by_class.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in by_class.values():
        rng.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        val_count = max(1, round(len(indices) * val_fraction))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def get_targets(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = dataset.dataset.targets
        return [int(base_targets[i]) for i in dataset.indices]
    return [int(x) for x in dataset.targets]


def make_datasets(data_dir: Path, image_size: int, val_split: float, seed: int):
    train_transform, eval_transform = make_transforms(image_size)
    train_dir, val_dir = locate_dataset(data_dir)

    print(f"Using training images from: {train_dir}")
    if val_dir is not None:
        print(f"Using validation images from: {val_dir}")

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)

        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise RuntimeError(
                "Training and validation folders have different class names/order."
            )
        return train_dataset, val_dataset, train_dataset.classes

    # Same files, two transform objects, split by shared indices.
    train_base = datasets.ImageFolder(train_dir, transform=train_transform)
    val_base = datasets.ImageFolder(train_dir, transform=eval_transform)

    train_indices, val_indices = stratified_indices(
        train_base.targets, val_split, seed
    )
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)

    return train_dataset, val_dataset, train_base.classes


def make_class_weights(dataset, num_classes: int) -> torch.Tensor:
    labels = torch.tensor(get_targets(dataset), dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()

    if torch.any(counts == 0):
        raise RuntimeError("At least one class has no training images.")

    # Inverse-frequency weights so rare classes are not ignored.
    weights = counts.sum() / (num_classes * counts)
    return weights


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device()
    print(f"Device: {device}")

    train_dataset, val_dataset, class_names = make_datasets(
        args.data_dir, args.image_size, args.val_split, args.seed
    )

    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    ).to(device)

    class_weights = make_class_weights(train_dataset, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_acc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "architecture": "resnet18",
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "class_names": class_names,
                "image_size": args.image_size,
                "mean": DEFAULT_MEAN,
                "std": DEFAULT_STD,
                "best_val_accuracy": best_val_acc,
            }
            torch.save(checkpoint, args.output)
            print(f"  Saved new best model -> {args.output}")

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {args.output}")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
