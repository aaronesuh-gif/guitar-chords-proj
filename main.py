"""
main.py
-------
Live chord detection. Loads the trained model + scaler + labels,
opens the webcam, and shows the predicted chord in real time.

Pipeline (shape-based, no fretboard detection):
    frame -> MediaPipe landmarks -> orientation (barre signal)
          -> feature vector -> MLP -> probability-averaged smoothing

Smoothing note: predictions are smoothed by averaging the model's
softmax PROBABILITIES over the recent window, not by modal vote over
argmax labels. Probability averaging responds faster to chord changes
(a confident new chord immediately shifts the average) while still
suppressing single-frame flickers.

Usage:
    python main.py

Requires:
    models/chord_mlp.pt, models/chord_labels.json, models/scaler.joblib
    (all produced by `python -m train.train`)
"""

import os
import json
from collections import deque

import cv2
import numpy as np
import torch
import joblib

from src.vision.hand_tracker import HandTracker
from src.vision.orientation import OrientationAnalyzer
from src.classifier.model import ChordMLP
from src.classifier.feature_builder import build_feature_vector
from src.ui.overlay import OverlayRenderer


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chord_mlp.pt")
LABELS_PATH = os.path.join(MODELS_DIR, "chord_labels.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

SMOOTHING_WINDOW = 6  # frames of probability averaging


class ChordDetector:
    def __init__(self):
        self._check_artifacts_exist()

        with open(LABELS_PATH) as f:
            label_data = json.load(f)
        self.classes = label_data["classes"]
        input_dim = label_data["input_dim"]

        self.model = ChordMLP(input_dim=input_dim, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        self.scaler = joblib.load(SCALER_PATH)

        self.tracker = HandTracker(max_num_hands=1, static_image_mode=False)
        self.orientation = OrientationAnalyzer()

        # Rolling buffer of probability vectors for smoothing
        self.prob_history = deque(maxlen=SMOOTHING_WINDOW)

    def _check_artifacts_exist(self):
        missing = [p for p in [MODEL_PATH, LABELS_PATH, SCALER_PATH] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing trained model artifacts:\n"
                + "\n".join(f"  {p}" for p in missing)
                + "\n\nRun `python -m train.build_dataset` then `python -m train.train` first."
            )

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Run the full pipeline on one frame.

        Returns
        -------
        dict with keys:
            chord             : smoothed predicted chord name, or None
            confidence        : smoothed probability of that chord (0-1)
            raw_chord         : this frame's raw (unsmoothed) prediction
            landmarks         : (21,3) landmark array or None
            orientation_result: OrientationResult or None
        """
        landmarks = self.tracker.process(frame)

        result = {
            "chord": None, "confidence": 0.0, "raw_chord": None,
            "landmarks": landmarks, "orientation_result": None,
        }

        if landmarks is None:
            # No hand — decay the buffer so an old chord doesn't linger
            # on screen after the hand leaves the frame
            if self.prob_history:
                self.prob_history.popleft()
            return result

        orientation_result = self.orientation.analyze_index_finger(landmarks)
        result["orientation_result"] = orientation_result

        feature_vector = build_feature_vector(orientation_result, landmarks)
        scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        scaled_tensor = torch.tensor(scaled, dtype=torch.float32)

        probs = self.model.predict_proba(scaled_tensor)[0].numpy()
        result["raw_chord"] = self.classes[int(np.argmax(probs))]

        # Probability-averaged smoothing
        self.prob_history.append(probs)
        avg_probs = np.mean(self.prob_history, axis=0)
        smoothed_idx = int(np.argmax(avg_probs))

        result["chord"] = self.classes[smoothed_idx]
        result["confidence"] = float(avg_probs[smoothed_idx])
        return result

    def close(self):
        self.tracker.close()


def main():
    detector = ChordDetector()
    renderer = OverlayRenderer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Live chord detection running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.process_frame(frame)
        display = renderer.render(frame, result)

        cv2.imshow("Chord Detector", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()