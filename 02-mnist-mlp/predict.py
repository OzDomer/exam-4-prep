#!/usr/bin/env python3
"""
Predict the digit in a given image file, using a checkpoint from train.py.

Example:
    python predict.py --image ./some_digit.png
    python predict.py --model model.pt --image ./some_digit.png --top_k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from model import Classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit.")
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    classifier = Classifier.from_checkpoint(args.model)
    with Image.open(args.image) as image:
        results = classifier.predict(image, top_k=args.top_k)

    best_name, best_confidence = results[0]
    print(f"Prediction: {best_name}")
    print(f"Confidence: {best_confidence:.2%}")

    if len(results) > 1:
        print("\nTop predictions:")
        for rank, (name, confidence) in enumerate(results, start=1):
            print(f"{rank}. {name}: {confidence:.2%}")


if __name__ == "__main__":
    main()
