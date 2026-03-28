"""
FaceDetector — OpenCV Haar cascade face detection.
Tuned parameters for real webcam conditions:
  - glasses, slight angles, indoor lighting, varying distances.
Automatically downloads the cascade XML if not present.
"""

import logging
import os
import urllib.request
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

HAARCASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)


class FaceDetector:
    def __init__(self, cascade_path: str):
        import cv2
        self.cv2 = cv2
        path = Path(cascade_path)

        if not path.exists():
            logger.info(f"Cascade not found — downloading to '{cascade_path}'...")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(HAARCASCADE_URL, str(path))
                logger.info("Cascade downloaded successfully.")
            except Exception as e:
                logger.warning(f"Download failed ({e}), using OpenCV built-in path.")
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.classifier = cv2.CascadeClassifier(str(cascade_path))
        if self.classifier.empty():
            # Last resort — use OpenCV built-in
            builtin = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.classifier = cv2.CascadeClassifier(builtin)
            if self.classifier.empty():
                raise RuntimeError("Failed to load Haar cascade classifier.")
            logger.info("Using OpenCV built-in cascade.")

        logger.info("Face detector ready.")

    def detect(self, frame: np.ndarray) -> list:
        """
        Returns a list of (x, y, w, h) bounding boxes.

        Uses a multi-pass strategy:
          Pass 1 — standard settings (good lighting, frontal face)
          Pass 2 — relaxed settings (glasses, slight angle, dim light)
        Returns the best result from either pass.
        """
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)

        # Equalise histogram to improve detection in dim / uneven lighting
        gray = self.cv2.equalizeHist(gray)

        # Pass 1: standard — more accurate when it works
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60),
            flags=self.cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) > 0:
            return [tuple(f) for f in faces]

        # Pass 2: relaxed — catches glasses, slight angles, further distances
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40),
            flags=self.cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) > 0:
            return [tuple(f) for f in faces]

        # Pass 3: very relaxed — last resort for difficult conditions
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.03,
            minNeighbors=2,
            minSize=(30, 30),
            flags=self.cv2.CASCADE_SCALE_IMAGE,
        )

        return [tuple(f) for f in faces] if len(faces) > 0 else []