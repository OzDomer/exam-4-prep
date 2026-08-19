#!/usr/bin/env python3
"""
Predict the class of one image from the command line, using a checkpoint
created by train.py. Same logic the web app uses -- handy for a quick check
before starting the server.

Example:
    python predict.py --image ./some_image.jpg
    python predict.py --model model.pt --image ./some_image.jpg --top_k 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from model import Classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an image class.")
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
