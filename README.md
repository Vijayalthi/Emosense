# EmoSense — Facial Emotion Recognition System

A production-ready facial emotion recognition system built with **VGG16 + LSTM**, **FastAPI**, and a beautiful dark-mode web interface. Detects 7 emotions in real-time via webcam or static image upload.

![Demo](https://img.shields.io/badge/demo-live-brightgreen) ![Python](https://img.shields.io/badge/python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal) ![Docker](https://img.shields.io/badge/docker-ready-blue)

---

## ✨ What's New (v2.0)

- Full web UI with live webcam streaming
- Image upload with drag-and-drop
- Emotion probability bars with smooth animations
- Face bounding boxes with corner brackets and labels
- Emotion history strip
- FastAPI backend with Swagger docs at `/docs`
- Docker + Docker Compose setup
- GitHub Actions CI/CD
- Demo mode (runs without a trained model — great for UI testing)

---

## 🎭 Emotions Detected

| Emotion   | Emoji |
|-----------|-------|
| Angry     | 😠    |
| Disgust   | 🤢    |
| Fear      | 😨    |
| Happy     | 😄    |
| Neutral   | 😐    |
| Sad       | 😢    |
| Surprised | 😲    |

---

## 🏗️ Architecture

```
User Browser
    │
    │  WebSocket / REST
    ▼
FastAPI (main.py)
    │
    ├── /predict/frame   ← base64 webcam frame
    ├── /predict/image   ← uploaded file
    └── /health
    │
    ├── FaceDetector (OpenCV Haar cascade)
    │       └── returns bounding boxes
    │
    └── EmotionModel (VGG16-inspired CNN + LSTM)
            └── returns 7-class softmax probabilities
```

**Model architecture:**
- VGG16-inspired CNN feature extractor (trained from scratch on grayscale 48×48 input)
- LSTM layer to model temporal/sequential patterns
- Softmax output — 7 emotion classes

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/Vijayalthi/facial-emotion-recognition
cd facial-emotion-recognition

# Build and run (demo mode — no model needed)
docker compose up --build

# Open browser
open http://localhost:8000
```

### Option 2 — Local Python

```bash
git clone https://github.com/Vijayalthi/facial-emotion-recognition
cd facial-emotion-recognition/backend

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --port 8000

# Open browser
open http://localhost:8000
```

> **Note:** The app runs in **demo mode** if no model file is present — the UI and API work fully, with randomised predictions.

---

## 🧠 Training Your Own Model

### 1. Get the FER2013 dataset

```bash
# Install kaggle CLI
pip install kaggle

# Download (requires kaggle API key)
kaggle datasets download -d msambare/fer2013 -p dataset/
unzip dataset/fer2013.zip -d dataset/fer2013
```

Expected layout:
```
dataset/fer2013/
  train/
    angry/    disgust/    fear/    happy/    neutral/    sad/    surprise/
  val/
    angry/    ...
```

### 2. Train

```bash
cd backend
python train.py \
  --data_dir ../dataset/fer2013 \
  --epochs 50 \
  --output models/emotion_model.h5
```

Training uses:
- Data augmentation (rotation, flip, zoom, shift)
- ReduceLROnPlateau + EarlyStopping callbacks
- Best checkpoint saved automatically

### 3. Run with trained model

```bash
# The server automatically picks up models/emotion_model.h5
uvicorn main:app --reload
```

---

## 🌐 API Reference

Full interactive docs at `http://localhost:8000/docs`

### `POST /predict/image`

Upload an image file.

```bash
curl -X POST http://localhost:8000/predict/image \
  -F "file=@face.jpg"
```

Response:
```json
{
  "success": true,
  "face_count": 1,
  "inference_ms": 42.3,
  "faces": [
    {
      "label": "Happy",
      "confidence": 0.8923,
      "box": [120, 80, 200, 200],
      "all_scores": {
        "Angry": 0.012, "Disgust": 0.003, "Fear": 0.008,
        "Happy": 0.892, "Neutral": 0.059, "Sad": 0.018, "Surprised": 0.008
      }
    }
  ]
}
```

### `POST /predict/frame`

Send a base64-encoded webcam frame.

```json
{ "image": "data:image/jpeg;base64,/9j/4AAQ..." }
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "detector_loaded": true }
```

---

## 🚢 Deployment

### Railway

```bash
# Install Railway CLI
npm i -g @railway/cli
railway login
railway init
railway up
```

### Render

1. Fork this repo
2. Create a new Web Service on Render
3. Set **Build Command**: `docker build -t fer .`
4. Set **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

### Hugging Face Spaces

Create a Space with Docker runtime, push this repo, and set `PORT=7860` in the environment variables.

---

## 📁 Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI app
│   ├── train.py             # Training script
│   ├── requirements.txt
│   ├── models/              # .h5 model + cascade XML go here
│   └── src/
│       ├── model.py         # Model loader
│       ├── detector.py      # Face detector
│       └── inference.py     # Inference runner
├── frontend/
│   └── index.html           # Full web UI (single file)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)
