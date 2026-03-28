FROM python:3.11-slim

LABEL maintainer="Vijayalthi"
LABEL description="Facial Emotion Recognition — FastAPI + TensorFlow"

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY backend/models/ ./models/ 2>/dev/null || true

WORKDIR /app/backend

# Create model directory
RUN mkdir -p models

ENV MODEL_PATH=/app/models/emotion_model.h5
ENV CASCADE_PATH=/app/models/haarcascade_frontalface_default.xml
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
