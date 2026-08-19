"""
Web app that serves predictions from the model trained by train.py.

Run it:
    python main.py
    (or)  uvicorn main:app --reload --port 9000

Then either:
    - open http://localhost:9000 in a browser, choose an image, get a prediction
    - or POST an image straight to the API:
        curl -F "file=@some_image.jpg" http://localhost:9000/predict
"""

from __future__ import annotations

import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from config import get_settings
from model import Classifier

app = FastAPI()
settings = get_settings()

# Load the trained model ONCE, when the server starts (not on every request).
# If it hasn't been trained yet, keep the server running but explain the
# problem when someone actually calls /predict.
classifier: Classifier | None = None
model_error: str | None = None
try:
    classifier = Classifier.from_checkpoint(settings.model_path)
    print(f"Loaded model '{settings.model_path}' with classes: {classifier.class_names}")
except Exception as error:  # surface any load failure to the client instead of crashing
    model_error = str(error)
    print(f"Could not load model '{settings.model_path}': {model_error}")
    print("Train one first, e.g.:  python train.py --data_dir ./data")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/classes")
def classes():
    if classifier is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {model_error}")
    return {"classes": classifier.class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {model_error}. Run train.py first.",
        )

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    results = classifier.predict(image, top_k=top_k)
    best_name, best_confidence = results[0]
    return {
        "prediction": best_name,
        "confidence": best_confidence,
        "top_k": [
            {"label": name, "confidence": confidence}
            for name, confidence in results
        ],
    }


@app.get("/", response_class=HTMLResponse)
def homepage():
    return HTML_PAGE


# A tiny single-page UI so you can demo "upload an image -> see a prediction"
# in the browser. It just posts to /predict with fetch() and shows the result.
HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Image Classifier</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 3rem auto;
           padding: 0 1rem; line-height: 1.5; }
    h1 { margin-bottom: .25rem; }
    .card { border: 1px solid #8883; border-radius: 12px; padding: 1.25rem; margin-top: 1rem; }
    #preview { max-width: 100%; max-height: 320px; border-radius: 8px; margin-top: 1rem; display: none; }
    button { font-size: 1rem; padding: .6rem 1.2rem; border-radius: 8px; border: 0;
             background: #2563eb; color: #fff; cursor: pointer; margin-top: 1rem; }
    button:disabled { opacity: .5; cursor: default; }
    .result { margin-top: 1rem; font-size: 1.1rem; }
    .bar { height: 10px; background: #2563eb; border-radius: 5px; margin: 4px 0; }
    .row { display: flex; justify-content: space-between; font-size: .9rem; opacity: .85; }
    .muted { opacity: .7; font-size: .9rem; }
  </style>
</head>
<body>
  <h1>Image Classifier</h1>
  <p class="muted">Upload an image and the trained model returns its prediction.</p>

  <div class="card">
    <input type="file" id="file" accept="image/*" />
    <img id="preview" alt="preview" />
    <br />
    <button id="go" disabled>Predict</button>
    <div class="result" id="result"></div>
  </div>

  <script>
    const fileInput = document.getElementById("file");
    const preview = document.getElementById("preview");
    const goButton = document.getElementById("go");
    const result = document.getElementById("result");

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      result.innerHTML = "";
      if (!file) { goButton.disabled = true; preview.style.display = "none"; return; }
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
      goButton.disabled = false;
    });

    goButton.addEventListener("click", async () => {
      const file = fileInput.files[0];
      if (!file) return;
      goButton.disabled = true;
      result.textContent = "Predicting...";

      const body = new FormData();
      body.append("file", file);

      try {
        const response = await fetch("/predict", { method: "POST", body });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Prediction failed");

        let html = `<strong>Prediction: ${data.prediction}</strong> `
                 + `(${(data.confidence * 100).toFixed(1)}%)`;
        for (const item of data.top_k) {
          const pct = (item.confidence * 100).toFixed(1);
          html += `<div class="row"><span>${item.label}</span><span>${pct}%</span></div>`
                + `<div class="bar" style="width:${pct}%"></div>`;
        }
        result.innerHTML = html;
      } catch (error) {
        result.textContent = "Error: " + error.message;
      } finally {
        goButton.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", port=settings.port, reload=True)
