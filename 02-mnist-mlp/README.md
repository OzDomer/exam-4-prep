# MNIST MLP (784/16/16/10) — the "plain neural network" flavor

The second exam template. Where `01-image-classifier-web` is the **ResNet /
transfer-learning** flavor for photos, this is the **plain fully-connected
network** flavor — use it when the task specifies a small N-layer network or a
simple grayscale dataset like MNIST.

This mirrors the teacher's own prompt: *"train.py to train a 784/16/16/10 neural
network to predict handwritten digits; predict.py to predict from a given file."*

```
02-mnist-mlp/
├── model.py     # the 784/16/16/10 MLP + load-checkpoint + predict-one-image
├── train.py     # train on the .npy dataset  ->  model.pt
├── predict.py   # predict the digit in a given image file
└── requirements.txt
```

## Which flavor do I pull on exam day?

| Task says… | Use |
|---|---|
| "train a 784/16/16/10 / N-layer network", MNIST, simple grayscale | **this** (`02-mnist-mlp`) |
| "classify these photos" (animals, objects, etc.) | `01-image-classifier-web` (ResNet) |

## 1. Setup

```bash
cd 02-mnist-mlp
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Git Bash on Windows
```

## 2. Get the dataset

```bash
.venv/Scripts/kaggle.exe datasets download -d hichamachahboun/mnist-handwritten-digits -p ./data --unzip
```

It's stored as NumPy arrays (`train_images.npy`, `train_labels.npy`,
`test_images.npy`, `test_labels.npy`) — `train.py` loads them directly, flattens
each 28×28 image to 784, and scales to [0, 1]. If a dataset has no test split,
`train.py` holds out 10% of train for validation automatically.

## 3. Train

```bash
.venv/Scripts/python.exe train.py --data_dir ./data
# more epochs:
.venv/Scripts/python.exe train.py --data_dir ./data --epochs 30
```

Saves the best checkpoint to **`model.pt`** (weights + class names + image size).
A 16-wide MLP is deliberately tiny, so expect ~92–95% — that's the spec, not a
CNN. Trains from scratch (no pretrained weights) because MNIST is small and
simple enough to learn quickly, even on CPU.

## 4. Predict on an image file

```bash
.venv/Scripts/python.exe predict.py --image ./some_digit.png
```

Give it any image of a single digit. `predict.py` converts it to grayscale
28×28, and auto-inverts if it's black-on-white (a photo/drawing) so it matches
MNIST's white-on-black convention.

## The chain (same shape as the ResNet flavor)

```
.npy arrays ──► train.py ──► model.pt ──► model.py Classifier ──► predict.py
                (784/16/16/10 MLP)         (loads + predicts one image)
```

No web app here (the spec only asked for train + predict). If you need one, it's
a copy of `01-image-classifier-web/main.py` with the import swapped to this
`Classifier` — the endpoint code is identical.
