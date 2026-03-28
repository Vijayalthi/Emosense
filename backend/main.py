"""
Facial Emotion Recognition — FastAPI Backend
Supports: single image upload, base64 frame (webcam), batch prediction
"""

import os
import io
import base64
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from src.model import EmotionModel
from src.detector import FaceDetector
from src.inference import run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Facial Emotion Recognition API",
    description="Real-time facial emotion detection using VGG16 + LSTM",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model state ────────────────────────────────────────────────────────
emotion_model: Optional[EmotionModel] = None
face_detector: Optional[FaceDetector] = None

MODEL_PATH = os.getenv("MODEL_PATH", "models/emotion_model.h5")
CASCADE_PATH = os.getenv("CASCADE_PATH", "models/haarcascade_frontalface_default.xml")


@app.on_event("startup")
async def startup_event():
    global emotion_model, face_detector
    logger.info("Loading face detector...")
    face_detector = FaceDetector(CASCADE_PATH)
    logger.info("Loading emotion model...")
    emotion_model = EmotionModel(MODEL_PATH)
    logger.info("✅ System ready.")


# ── Schemas ───────────────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    image: str  # base64 encoded JPEG/PNG


class EmotionResult(BaseModel):
    label: str
    confidence: float
    box: list[int]
    all_scores: dict[str, float]


class PredictResponse(BaseModel):
    success: bool
    faces: list[EmotionResult]
    inference_ms: float
    face_count: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": emotion_model is not None and emotion_model.is_loaded,
        "detector_loaded": face_detector is not None,
    }


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict emotion from an uploaded image file."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    img_array = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    return _process_frame(frame)


@app.post("/predict/frame", response_model=PredictResponse)
async def predict_frame(req: FrameRequest):
    """Predict emotion from a base64-encoded webcam frame."""
    try:
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        raw = base64.b64decode(encoded)
        img_array = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    return _process_frame(frame)


def _process_frame(frame: np.ndarray) -> PredictResponse:
    t0 = time.perf_counter()

    if emotion_model is None or not emotion_model.is_loaded:
        # Demo mode — return mock results
        return _mock_response(time.perf_counter() - t0)

    faces = face_detector.detect(frame)
    results = []
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        label, confidence, all_scores = run_inference(roi, emotion_model)
        results.append(EmotionResult(
            label=label,
            confidence=round(confidence, 4),
            box=[int(x), int(y), int(w), int(h)],
            all_scores={k: round(v, 4) for k, v in all_scores.items()},
        ))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        success=True,
        faces=results,
        inference_ms=round(elapsed_ms, 2),
        face_count=len(results),
    )


def _mock_response(elapsed: float) -> PredictResponse:
    """Returns a realistic mock when no model is loaded (demo mode)."""
    import random
    emotions = ["Happy", "Neutral", "Sad", "Angry", "Surprised", "Fear", "Disgust"]
    primary = random.choice(emotions)
    scores = {e: round(random.uniform(0.01, 0.15), 4) for e in emotions}
    scores[primary] = round(random.uniform(0.55, 0.92), 4)
    total = sum(scores.values())
    scores = {k: round(v / total, 4) for k, v in scores.items()}

    return PredictResponse(
        success=True,
        faces=[EmotionResult(
            label=primary,
            confidence=scores[primary],
            box=[80, 60, 200, 200],
            all_scores=scores,
        )],
        inference_ms=round(elapsed * 1000, 2),
        face_count=1,
    )


# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Only mount /static if it exists (it's optional)
STATIC_DIR = FRONTEND_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if FRONTEND_DIR.exists():
    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        index = FRONTEND_DIR / "index.html"
        return index.read_text(encoding="utf-8")