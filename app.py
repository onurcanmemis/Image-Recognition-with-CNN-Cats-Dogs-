"""Pet-breed classifier — FastAPI web app.

Loads three artefacts produced by `model_experiments/train_compare.py`:

  * `cnn_model.keras`       — the winning model (Keras 3 native format)
  * `class_indices.json`    — index → breed name
  * `model_info.json`       — winner metadata: preprocessing tag, Grad-CAM
                              target layer, accuracy, params, etc.

The preprocessing function and Grad-CAM target layer are read from
`model_info.json` so the same `app.py` works no matter which model won the
tournament.

Routes
------
GET  /         — render templates/index.html (upload form + model card)
POST /predict  — multipart file upload; returns JSON with top-5 candidates,
                 base64 PNG of the resized input, and base64 PNG of the
                 Grad-CAM overlay.
GET  /health   — liveness probe; returns {"status": "ok"}.

Local dev:
    uv run uvicorn app:app --reload --port 7860
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# ── Paths ───────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
MODEL_PATH = REPO_ROOT / "cnn_model.keras"
CLASS_INDICES_PATH = REPO_ROOT / "class_indices.json"
MODEL_INFO_PATH = REPO_ROOT / "model_info.json"
STATIC_DIR = REPO_ROOT / "static"
TEMPLATES_DIR = REPO_ROOT / "templates"

INPUT_SIZE = (224, 224)
TOP_K = 5


# ── Preprocessing dispatch (mirrors train_compare.py — preprocessing tag
#    round-trip is the contract between training and inference) ──────────
def preprocess_fn_for(name: str):
    if name == "rescale_1_over_255":
        return lambda x: x / 255.0
    if name == "mobilenet_v2":
        from keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input
    if name == "efficientnet":
        from keras.applications.efficientnet import preprocess_input
        return preprocess_input
    raise ValueError(f"Unknown preprocessing tag: {name}")


# ── Load artefacts once at import time ──────────────────────────────────
def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise RuntimeError(
            f"{path.name} missing — {hint}. "
            "Run `uv run python model_experiments/train_compare.py` first."
        )


_require(MODEL_PATH, "the winning model artefact")
_require(CLASS_INDICES_PATH, "produced by the tournament script")
_require(MODEL_INFO_PATH, "produced by the tournament script")

model: tf.keras.Model = tf.keras.models.load_model(MODEL_PATH)
class_indices: dict[int, str] = {
    int(k): v for k, v in json.loads(CLASS_INDICES_PATH.read_text()).items()
}
model_info: dict = json.loads(MODEL_INFO_PATH.read_text())
preprocess_fn = preprocess_fn_for(model_info["preprocessing"])


# ── Inference helpers ───────────────────────────────────────────────────
def prepare_image(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_resized_uint8, preprocessed_batch_1xHxWx3)."""
    img = pil_image.convert("RGB").resize(INPUT_SIZE, Image.Resampling.LANCZOS)
    raw = np.asarray(img, dtype=np.uint8)
    preprocessed = np.asarray(preprocess_fn(np.expand_dims(raw.astype(np.float32), 0)))
    return raw, preprocessed


def gradcam_heatmap(preprocessed: np.ndarray, class_idx: int, target_layer: str) -> np.ndarray:
    """Return a Grad-CAM heatmap of shape (H, W) normalised to [0, 1]."""
    gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
    score = CategoricalScore([class_idx])
    cam = gradcam(score, preprocessed, penultimate_layer=target_layer)
    return cam[0]


def overlay_heatmap(image_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Resize the heatmap to the image, apply jet colormap, blend at alpha."""
    h, w = image_uint8.shape[:2]
    cam = cv2.resize(heatmap, (w, h))
    cam = (255 * np.clip(cam, 0.0, 1.0)).astype(np.uint8)
    colour = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    return (image_uint8 * (1 - alpha) + colour * alpha).astype(np.uint8)


def _png_b64(image_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(title="Pet Breed Classifier", version=model_info.get("run_date", "0.0.0"))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"model_info": model_info}
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="File is not a valid image.") from exc

    raw, batch = prepare_image(pil_image)
    probs = model.predict(batch, verbose=0)[0]
    top_idx = np.argsort(probs)[-TOP_K:][::-1]
    top_breeds = [class_indices[int(i)] for i in top_idx]
    top_confs = [float(probs[int(i)]) for i in top_idx]
    winner_idx = int(top_idx[0])

    gradcam_b64: str | None = None
    gradcam_failed: str | None = None
    try:
        heat = gradcam_heatmap(batch, winner_idx, model_info["gradcam_target_layer"])
        overlay = overlay_heatmap(raw, heat)
        gradcam_b64 = _png_b64(overlay)
    except Exception as exc:  # tf-keras-vis can fail if the layer name doesn't exist
        gradcam_failed = str(exc)

    return JSONResponse(
        {
            "top_breeds": top_breeds,
            "top_confs": top_confs,
            "raw_b64": _png_b64(raw),
            "gradcam_b64": gradcam_b64,
            "gradcam_failed": gradcam_failed,
        }
    )
