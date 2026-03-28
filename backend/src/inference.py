"""
run_inference — takes a face ROI numpy array, returns (label, confidence, all_scores).
"""

import numpy as np
from src.model import EmotionModel, EMOTIONS


def run_inference(
    face_roi: np.ndarray,
    model: EmotionModel,
) -> tuple[str, float, dict[str, float]]:
    """
    Args:
        face_roi: BGR uint8 crop of a detected face
        model:    loaded EmotionModel

    Returns:
        (top_label, top_confidence, {emotion: score, ...})
    """
    probs = model.predict(face_roi)  # (7,)
    top_idx = int(np.argmax(probs))
    label = EMOTIONS[top_idx]
    confidence = float(probs[top_idx])
    all_scores = {e: float(probs[i]) for i, e in enumerate(EMOTIONS)}
    return label, confidence, all_scores
