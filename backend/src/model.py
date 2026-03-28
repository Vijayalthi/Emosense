"""
EmotionModel — loads the trained pure CNN model and runs inference.
Input shape: (48, 48, 1) — grayscale face ROI, no time-step dimension.
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
IMG_SIZE = 48


class EmotionModel:
    def __init__(self, model_path: str):
        self.model = None
        self.is_loaded = False
        self.emotions = EMOTIONS
        self._load(model_path)

    def _load(self, path: str):
        if not Path(path).exists():
            logger.warning(f"Model not found at '{path}' — running in DEMO mode.")
            return
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(path, compile=False)
            self.is_loaded = True
            # Log the actual input shape so we can verify
            input_shape = self.model.input_shape
            logger.info(f"Model loaded from '{path}'")
            logger.info(f"Model input shape: {input_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict(self, face_roi: np.ndarray) -> np.ndarray:
        """
        Accepts a BGR uint8 numpy array (any size) and returns
        a (7,) softmax probability vector.

        Pure CNN model expects: (1, 48, 48, 1)
        — no time-step dimension like the old LSTM model had.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded")

        import cv2

        # Convert BGR to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # Resize to 48x48
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

        # Normalise to [0, 1]
        x = resized.astype("float32") / 255.0

        # Shape: (48, 48) → (48, 48, 1) → (1, 48, 48, 1)
        x = np.expand_dims(x, axis=-1)   # (48, 48, 1)
        x = np.expand_dims(x, axis=0)    # (1, 48, 48, 1)  ← CNN expects this

        probs = self.model.predict(x, verbose=0)[0]
        return probs