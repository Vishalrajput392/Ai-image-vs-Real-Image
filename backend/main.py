"""
backend/main.py
----------------
FastAPI backend serving the trained AI-vs-Real classifier (SRS Section 14.2).

Endpoint:
    POST /predict — multipart/form-data image upload -> JSON prediction

Responsibilities: validation (format/size/integrity), preprocessing,
model inference, structured JSON response, structured error handling
on failure (SRS 14.3), auto-generated docs at /docs.

Run:
    uvicorn backend.main:app --reload --port 8000

Dependencies:
    fastapi, uvicorn, python-multipart, torch, opencv-python, numpy
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.transforms import EvalTransform, ensure_rgb_3channel
from models.efficientnet import build_model

app = FastAPI(title="AI-Generated vs Real Image Detection API", version="1.0")

_model = None
_transform = EvalTransform()

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MODEL_VERSION = "model_v1"


def get_model():
    """Loads the checkpoint once and caches it (avoids reloading per request)."""
    global _model
    if _model is None:
        ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
        model = build_model(pretrained=False).to(config.DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
        model.eval()
        _model = model
    return _model


@app.on_event("startup")
def load_model_on_startup():
    get_model()
    print("Model loaded and ready.")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ---- Validation (SRS 14.3 / 15) ----
    if file.content_type not in ALLOWED_MIME_TYPES:
        return JSONResponse(status_code=415, content={
            "error_code": "UNSUPPORTED_FORMAT",
            "message": "Unsupported image format",
        })

    raw_bytes = await file.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > config.MAX_FILE_SIZE_MB:
        return JSONResponse(status_code=413, content={
            "error_code": "FILE_TOO_LARGE",
            "message": f"Image exceeds the {config.MAX_FILE_SIZE_MB}MB size limit",
        })

    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return JSONResponse(status_code=422, content={
            "error_code": "CORRUPTED_IMAGE",
            "message": "Unable to process image",
        })

    h, w = img_bgr.shape[:2]
    if min(h, w) < config.MIN_RESOLUTION:
        return JSONResponse(status_code=422, content={
            "error_code": "RESOLUTION_TOO_LOW",
            "message": "Image resolution is too low for reliable prediction",
        })

    # ---- Preprocessing + Inference ----
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = ensure_rgb_3channel(img_rgb)
        tensor = _transform(img_rgb).unsqueeze(0).to(config.DEVICE)

        model = get_model()
        with torch.no_grad():
            logit = model(tensor).squeeze()
            ai_prob = torch.sigmoid(logit).item()

        real_prob = 1.0 - ai_prob
        predicted_label = (
            "AI-Generated" if ai_prob >= config.CLASSIFICATION_THRESHOLD else "Real"
        )

        return {
            "predicted_label": predicted_label,
            "ai_generated_probability": round(ai_prob * 100, 2),
            "real_probability": round(real_prob * 100, 2),
            "model_version": MODEL_VERSION,
            "disclaimer": "This prediction reflects the model's estimate and is not forensic proof.",
        }

    except Exception:
        # Model/API failure -> structured error, app does not crash (SRS 14.3)
        return JSONResponse(status_code=500, content={
            "error_code": "INFERENCE_FAILURE",
            "message": "Something went wrong while processing the image",
        })
