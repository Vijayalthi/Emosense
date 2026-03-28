FROM python:3.11-slim

LABEL maintainer="Vijayalthi"
LABEL description="EmoSense — Facial Emotion Recognition — FastAPI + TensorFlow"

# System deps for OpenCV (libgl1 replaces libgl1-mesa-glx on Debian trixie+)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create empty models folder — main.py downloads files here on first startup
RUN mkdir -p backend/models

WORKDIR /app/backend

ENV MODEL_PATH=models/emotion_model.h5
ENV CASCADE_PATH=models/haarcascade_frontalface_default.xml
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]