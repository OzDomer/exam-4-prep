"""
Shared model utilities for the MNIST 784/16/16/10 MLP.

Both train.py and predict.py import from here, so "build the network" and
"turn an image into a prediction" live in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from PIL import Image


def build_mlp(input_dim: int = 784, num_classes: int = 10) -> nn.Module:
    # The exact architecture from the spec: 784 -> 16 -> 16 -> 10.
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path | str, device: torch.device) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

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


def image_to_input(image: Image.Image, image_size: int = 28) -> torch.Tensor:
    # Match MNIST's format: 28x28 grayscale, values in [0, 1], digit drawn
    # white-on-black. Returns a (1, 784) tensor ready for the MLP.
    image = image.convert("L").resize((image_size, image_size))
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32) / 255.0
    # MNIST is white-on-black; if the image looks black-on-white (light
    # background, e.g. a photo or a drawing), invert it so it matches.
    if tensor.mean() > 0.5:
        tensor = 1.0 - tensor
    return tensor.unsqueeze(0)  # (1, 784)


class Classifier:
    """A trained MLP bundled with its class names, for easy reuse."""

    def __init__(self, model, class_names, device, image_size=28):
        self.model = model
        self.class_names = list(class_names)
        self.device = device
        self.image_size = image_size

    @classmethod
    def from_checkpoint(cls, path: Path | str, device: torch.device | None = None):
        device = device or choose_device()
        checkpoint = load_checkpoint(path, device)
        model = build_mlp(
            input_dim=int(checkpoint.get("input_dim", 784)),
            num_classes=len(checkpoint["class_names"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return cls(
            model,
            checkpoint["class_names"],
            device,
            int(checkpoint.get("image_size", 28)),
        )

    def predict(self, image: Image.Image, top_k: int = 3):
        x = image_to_input(image, self.image_size).to(self.device)
        with torch.inference_mode():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)[0]

        k = max(1, min(top_k, len(self.class_names)))
        confidences, indices = torch.topk(probabilities, k=k)
        return [
            (self.class_names[index.item()], confidence.item())
            for confidence, index in zip(confidences, indices)
        ]
