"""
main.py
-------
The full live inference loop. Loads the trained model + scaler +
labels, opens the webcam, and shows the predicted chord in real time.

This is a minimal first version — full skeleton drawing, fretboard
grid overlay, and confidence-coded styling come later (Phase 6). For
now this is deliberately bare-bones so you can quickly confirm the
whole pipeline + trained model actually works end-to-end.

Usage:
    python main.py

Requires:
    models/chord_mlp.pt, models/chord_labels.json, models/scaler.joblib
    (all produced by `python -m train.train`)
"""

import os
import json
from collections import deque, Counter

import cv2
import numpy as np
import torch
import joblib

from src.localization.edge_detector import EdgeDetector
from src.localization.hough_lines import HoughDetector
from src.localization.homography import HomographyComputer
from src.vision.hand_tracker import HandTracker
from src.vision.coordinate_mapper import CoordinateMapper
from src.vision.orientation import OrientationAnalyzer
from src.classifier.model import ChordMLP
from src.classifier.feature_builder import build_feature_vector
from src.ui.overlay import OverlayRenderer


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "chord_mlp.pt")
LABELS_PATH = os.path.join(MODELS_DIR, "chord_labels.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

N_FRETS = 5
N_STRINGS = 6
SMOOTHING_WINDOW = 5  # frames — modal vote to stabilize predictions.
# Reduced from 8: a smaller window reacts faster when you switch chords,
# at a slight cost in stability. If predictions flicker too much, raise it.


class ChordDetector:
    def __init__(self):
        self._check_artifacts_exist()

        # Load the three saved artifacts
        with open(LABELS_PATH) as f:
            label_data = json.load(f)
        self.classes = label_data["classes"]
        input_dim = label_data["input_dim"]

        self.model = ChordMLP(input_dim=input_dim, num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        self.scaler = joblib.load(SCALER_PATH)

        # Pipeline components — same chain as build_dataset.py, but in
        # VIDEO mode (static_image_mode=False) since this runs on a
        # continuous camera feed rather than isolated photos.
        self.edge_detector = EdgeDetector()
        self.hough_detector = HoughDetector(threshold=80)
        self.homography = HomographyComputer(n_frets=N_FRETS, n_strings=N_STRINGS)
        self.tracker = HandTracker(max_num_hands=1, static_image_mode=False)
        self.mapper = CoordinateMapper(self.homography)
        self.orientation = OrientationAnalyzer()

        # Rolling buffer of recent predictions for temporal smoothing
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)

    def _check_artifacts_exist(self):
        missing = [p for p in [MODEL_PATH, LABELS_PATH, SCALER_PATH] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing trained model artifacts:\n"
                + "\n".join(f"  {p}" for p in missing)
                + "\n\nRun `python -m train.build_dataset` then `python -m train.train` first."
            )

    def process_frame(self, frame: np.ndarray):
        """
        Run the full pipeline on one frame.

        Returns
        -------
        dict with keys:
            chord         : smoothed predicted chord name, or None
            confidence    : confidence of the smoothed prediction (0-1)
            raw_chord     : this frame's raw (unsmoothed) prediction, or None
            frets, strings: detected Line objects, for optional drawing
        """
        edges = self.edge_detector.process(frame)
        frets, strings = self.hough_detector.detect(edges, frame.shape)
        H, H_inv = self.homography.compute(frets, strings, frame.shape)

        landmarks = self.tracker.process(frame)

        result = {
            "chord": None, "confidence": 0.0,
            "raw_chord": None, "frets": frets, "strings": strings,
            "landmarks": landmarks, "positions": None,
            "orientation_result": None, "homography": self.homography,
        }

        if landmarks is None or H is None:
            return result

        positions = self.mapper.map_landmarks(landmarks)
        orientation_result = self.orientation.analyze_index_finger(landmarks, frets)
        result["positions"] = positions
        result["orientation_result"] = orientation_result

        feature_vector = build_feature_vector(positions, orientation_result)
        scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        scaled_tensor = torch.tensor(scaled, dtype=torch.float32)

        probs = self.model.predict_proba(scaled_tensor)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        raw_chord = self.classes[pred_idx]

        self.prediction_history.append(raw_chord)
        smoothed_chord, smoothed_conf = self._smooth_prediction()

        result["chord"] = smoothed_chord
        result["confidence"] = smoothed_conf
        result["raw_chord"] = raw_chord
        return result

    def _smooth_prediction(self) -> tuple[str, float]:
        """Modal vote over the recent prediction buffer."""
        if not self.prediction_history:
            return None, 0.0
        counts = Counter(self.prediction_history)
        chord, count = counts.most_common(1)[0]
        confidence = count / len(self.prediction_history)
        return chord, confidence

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