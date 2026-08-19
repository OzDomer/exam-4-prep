#!/usr/bin/env python3
"""
Train a 784/16/16/10 MLP to classify handwritten digits (MNIST).

This is the "plain neural network from scratch" flavor (no ResNet, no transfer
learning) -- use it when the task specifies a small fully-connected network or a
simple grayscale dataset like MNIST.

Dataset: the Kaggle "mnist-handwritten-digits" set, stored as NumPy arrays:
    train_images.npy   (N, 28, 28) uint8
    train_labels.npy   (N,)
    test_images.npy    (M, 28, 28) uint8   [used for validation]
    test_labels.npy    (M,)

Examples:
    python train.py --data_dir ./data
    python train.py --data_dir ./data --epochs 30 --batch_size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_mlp, choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MNIST 784/16/16/10 MLP.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Folder containing the .npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.pt"),
        help="Where to save the best model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_file(data_dir: Path, name: str) -> Path:
    # Direct hit, else search sub-folders (a zip may unpack into a wrapper dir).
    direct = data_dir / name
    if direct.exists():
        return direct
    matches = list(data_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {data_dir}")
    return matches[0]


def load_split(data_dir: Path, images_name: str, labels_name: str):
    images = np.load(find_file(data_dir, images_name))
    labels = np.load(find_file(data_dir, labels_name))
    # (N, 28, 28) uint8 -> (N, 784) float in [0, 1]
    x = torch.from_numpy(images.reshape(len(images), -1)).float() / 255.0
    y = torch.from_numpy(labels).long()
    return x, y


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = choose_device()
    print(f"Device: {device}")

    x_train, y_train = load_split(args.data_dir, "train_images.npy", "train_labels.npy")
    try:
        x_val, y_val = load_split(args.data_dir, "test_images.npy", "test_labels.npy")
    except FileNotFoundError:
        # No separate test set -> hold out 10% of train for validation.
        n_val = max(1, len(x_train) // 10)
        perm = torch.randperm(len(x_train))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        x_val, y_val = x_train[val_idx], y_train[val_idx]
        x_train, y_train = x_train[train_idx], y_train[train_idx]
        print("No test split found; held out 10% of train for validation.")

    class_names = [str(i) for i in range(int(y_train.max().item()) + 1)]
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Training images: {len(x_train)} | Validation images: {len(x_val)}")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size)

    model = build_mlp(input_dim=x_train.shape[1], num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        train_loss, train_acc = run_loss / max(total, 1), correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2%} | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "architecture": "mlp-784-16-16-10",
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": 28,
                    "input_dim": int(x_train.shape[1]),
                    "best_val_accuracy": best_val_acc,
                },
                args.output,
            )
            print(f"  Saved new best model -> {args.output}")

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to: {args.output}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
