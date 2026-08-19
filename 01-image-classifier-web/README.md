# Image Classifier + Web App (exam-style project)

Take a dataset → **train** a model on it → **serve** it from a web app so you can
upload an image and get a prediction. This is the two halves of the exam glued
together:

- **Model** (like `2026-08-11-torch-2/04-fish-classification`): `train.py`, `predict.py`, `model.py`
- **Web app** (like `2026-08-13-fastapi`): `main.py`, `config.py`

```
01-image-classifier-web/
├── model.py        # shared: load checkpoint + predict on one image
├── train.py        # train ResNet-18 on your dataset  -> model.pt
├── predict.py      # command-line prediction (quick sanity check)
├── main.py         # FastAPI web app: browser UI + POST /predict
├── config.py       # settings (port, model_path)
└── requirements.txt
```

## 1. Setup

```bash
cd 2026-08-19-exam-prep/01-image-classifier-web
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## 2. Get a dataset

One folder per class. Put a bunch of images in each:

```
data/
  cats/    cat1.jpg cat2.jpg ...
  dogs/    dog1.jpg dog2.jpg ...
```

You can reuse the fish dataset you already have:
`../../2026-08-11-torch-2/04-fish-classification/aquarium-fish-classification`.
A pre-split `data/train/...` + `data/val/...` layout also works automatically.

## 3. Train the model

```bash
python train.py --data_dir ./data
# fewer/more epochs:
python train.py --data_dir ./data --epochs 10 --batch_size 32
```

This writes **`model.pt`** — the checkpoint stores the weights, the class names,
and the exact preprocessing (image size + normalization) so inference matches
training.

Quick check from the command line:

```bash
python predict.py --image ./some_image.jpg
```

## 4. Run the web app

```bash
python main.py
# or: uvicorn main:app --reload --port 9000
```

Open <http://localhost:9000>, choose an image, and press **Predict**.

Or call the API directly:

```bash
curl -F "file=@some_image.jpg" http://localhost:9000/predict
```

```json
{
  "prediction": "cats",
  "confidence": 0.97,
  "top_k": [
    { "label": "cats", "confidence": 0.97 },
    { "label": "dogs", "confidence": 0.03 }
  ]
}
```

## Endpoints

| Method | Path        | What it does                                  |
|--------|-------------|-----------------------------------------------|
| GET    | `/`         | Browser UI (upload an image, see prediction)  |
| GET    | `/health`   | `{status, model_loaded}`                       |
| GET    | `/classes`  | The class names the model knows                |
| POST   | `/predict`  | Upload an image → prediction + top-k           |

## How the two halves connect

`train.py` saves a checkpoint. `model.py`'s `Classifier.from_checkpoint()` loads
it and turns a `PIL` image into `(label, confidence)` pairs. Both `predict.py`
(CLI) and `main.py` (web) call that same `Classifier` — so the web app is just a
thin HTTP wrapper around the exact prediction logic you tested on the command
line. The model is loaded once at startup, not per request.

## Swapping in a different kind of "prediction"

The exam says "an image **or some piece of information**." This project does
images. For tabular/text input instead, keep the same shape:
train a model → save a checkpoint → load it once in `main.py` → add a `/predict`
endpoint that reads the input from the request (JSON body instead of an upload)
and returns the model's output.
```
