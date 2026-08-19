"""
Shared model utilities.

Loads a checkpoint produced by train.py and runs a prediction on one image.
Both predict.py (command line) and main.py (web app) import from here, so the
"how do I turn an image into a prediction" logic lives in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import models, transforms


# ImageNet stats -- the values ResNet-18 was pretrained with. Kept here as the
# fallback in case an older checkpoint didn't store its own mean/std.
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    # weights_only=True is the safe way to load checkpoints that only contain
    # tensors and basic Python types. Older PyTorch releases lack the argument.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    required = {"model_state_dict", "class_names"}
    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required fields: {', '.join(sorted(missing))}"
        )
    return checkpoint


def build_model(checkpoint: dict, device: torch.device) -> nn.Module:
    architecture = checkpoint.get("architecture", "resnet18")
    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")

    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def make_transform(checkpoint: dict):
    # Rebuild the exact same preprocessing the model was validated with, using
    # the image_size / mean / std that train.py stored inside the checkpoint.
    image_size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("mean", DEFAULT_MEAN)
    std = checkpoint.get("std", DEFAULT_STD)

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class Classifier:
    """A trained model bundled with its transform and class names."""

    def __init__(self, model, transform, class_names, device):
        self.model = model
        self.transform = transform
        self.class_names = list(class_names)
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, path: Path | str, device: torch.device | None = None
    ) -> "Classifier":
        device = device or choose_device()
        checkpoint = load_checkpoint(Path(path), device)
        model = build_model(checkpoint, device)
        transform = make_transform(checkpoint)
        return cls(model, transform, checkpoint["class_names"], device)

    def predict(self, image: Image.Image, top_k: int = 3):
        """Return a list of (class_name, confidence) sorted best-first."""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        k = max(1, min(top_k, len(self.class_names)))
        confidences, indices = torch.topk(probabilities, k=k)
        return [
            (self.class_names[index.item()], confidence.item())
            for confidence, index in zip(confidences, indices)
        ]
