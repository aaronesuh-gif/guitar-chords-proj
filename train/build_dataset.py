"""
build_dataset.py
-----------------
Offline batch processor. Walks every image in data/<chord_name>/,
runs the full vision pipeline (edges -> hough -> homography ->
landmarks -> coordinate mapping -> orientation) on each one, and
writes the resulting feature vectors + labels to data/dataset.csv.

This is the file that converts your raw photos into the structured
data your classifier actually trains on. Run this any time you add
new photos to data/ and want to rebuild the training set.

Usage:
    python -m train.build_dataset
"""

import os
import sys

# Allow running as `python -m train.build_dataset` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import pandas as pd

from src.localization.edge_detector import EdgeDetector
from src.localization.hough_lines import HoughDetector
from src.localization.homography import HomographyComputer
from src.vision.hand_tracker import HandTracker
from src.vision.coordinate_mapper import CoordinateMapper
from src.vision.orientation import OrientationAnalyzer
from src.classifier.feature_builder import build_landmark_feature_dict


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "dataset.csv")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Number of frets/strings expected in frame — must match how you
# actually shoot your photos (see HomographyComputer in homography.py)
N_FRETS = 5
N_STRINGS = 6


class DatasetBuilder:
    def __init__(self, n_frets: int = N_FRETS, n_strings: int = N_STRINGS):
        # static_image_mode=True is important here — we're processing
        # unrelated still images, not a continuous video stream, so
        # MediaPipe should run full detection on every single one
        # rather than assuming temporal continuity between frames.
        self.edge_detector = EdgeDetector()
        self.hough_detector = HoughDetector(threshold=80)
        self.homography = HomographyComputer(n_frets=n_frets, n_strings=n_strings)
        self.tracker = HandTracker(max_num_hands=1, static_image_mode=True)
        self.mapper = CoordinateMapper(self.homography)
        self.orientation = OrientationAnalyzer()

        self.n_frets = n_frets
        self.n_strings = n_strings

        # Track data quality — no_fretboard images are still INCLUDED
        # (shape features carry them); the counter just tells you how
        # often fretboard detection is failing on your photos.
        self.skip_reasons = {
            "no_fretboard": 0,   # informational — these rows are kept
            "no_hand": 0,        # skipped — no usable signal
            "unreadable": 0,     # skipped — corrupt/unreadable file
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, data_dir: str = DATA_DIR, output_csv: str = OUTPUT_CSV) -> pd.DataFrame:
        """
        Process every chord folder in data_dir and write the combined
        dataset to output_csv.

        Returns
        -------
        The resulting DataFrame (also written to disk as a side effect).
        """
        chord_folders = self._find_chord_folders(data_dir)
        if not chord_folders:
            raise RuntimeError(
                f"No chord folders found in {data_dir}. "
                f"Run capture_images.py first, or add images manually."
            )

        rows = []
        for chord_name, folder_path in chord_folders.items():
            print(f"\nProcessing '{chord_name}'...")
            chord_rows = self._process_folder(chord_name, folder_path)
            rows.extend(chord_rows)
            print(f"  {len(chord_rows)} usable images")

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)

        self._print_summary(df, chord_folders)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_chord_folders(self, data_dir: str) -> dict[str, str]:
        """Find every subfolder of data_dir — each one is a chord label."""
        if not os.path.exists(data_dir):
            return {}
        folders = {}
        for name in sorted(os.listdir(data_dir)):
            path = os.path.join(data_dir, name)
            if os.path.isdir(path):
                folders[name] = path
        return folders

    def _process_folder(self, chord_name: str, folder_path: str) -> list[dict]:
        """Run the full pipeline on every image in one chord's folder."""
        rows = []
        filenames = [
            f for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(VALID_EXTENSIONS)
        ]

        for filename in filenames:
            path = os.path.join(folder_path, filename)
            row = self._process_image(path, chord_name)
            if row is not None:
                rows.append(row)

        return rows

    def _process_image(self, path: str, chord_name: str) -> dict | None:
        """
        Run one image through the full pipeline and build a feature row.

        Returns None (and logs why) if any stage fails — most commonly
        because the fretboard or hand couldn't be detected in that
        particular photo.
        """
        frame = cv2.imread(path)
        if frame is None:
            self.skip_reasons["unreadable"] += 1
            return None

        # Stage 1: hand landmarks — the only hard requirement.
        # No hand = no usable training signal, skip.
        landmarks = self.tracker.process(frame)
        if landmarks is None:
            self.skip_reasons["no_hand"] += 1
            return None

        # Stage 2: fretboard localization — OPTIONAL. If it fails, the
        # fret/string features become zeros and the shape features
        # carry the example. Including these images is important: live
        # inference constantly encounters frames without a detected
        # fretboard, so the model must be trained on that case too.
        # Detection is focused on a ROI around the hand, matching what
        # main.py does live.
        edges = self.edge_detector.process(frame)
        roi = self._hand_roi(landmarks, frame.shape)
        frets, strings = self.hough_detector.detect(edges, frame.shape, roi=roi)
        H, H_inv = self.homography.compute(frets, strings, frame.shape)
        if H is None:
            self.skip_reasons["no_fretboard"] += 1  # counted, but NOT skipped

        # Stage 3: coordinate mapping (fret, string) per fingertip —
        # returns None per finger when H is None, encoded as zeros
        positions = self.mapper.map_landmarks(landmarks)

        # Stage 4: orientation / barre detection — needs at least one
        # fret line for a reference direction; defaults kick in otherwise
        orientation_result = self.orientation.analyze_index_finger(landmarks, frets)

        return self._build_row(chord_name, positions, orientation_result, landmarks)

    @staticmethod
    def _hand_roi(landmarks, frame_shape) -> tuple:
        """Generous box around the hand — same expansion as main.py."""
        h, w = frame_shape[:2]
        xs, ys = landmarks[:, 0], landmarks[:, 1]
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad_x = (x2 - x1) * 1.5 + 40
        pad_y = (y2 - y1) * 0.8 + 40
        return (
            max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y),
        )

    def _build_row(
        self,
        chord_name: str,
        positions: dict,
        orientation_result,
        landmarks,
    ) -> dict:
        """
        Assemble one CSV row from the mapped positions, orientation
        result, and normalized landmark shape features. Missing fingers
        (position=None, e.g. thumb/pinky often aren't on the fretboard)
        are encoded as 0,0 — the classifier learns to treat that as
        "not fretting" through training.
        """
        row = {"label": chord_name}

        for finger_name in HandTracker.FINGERTIP_NAMES:
            pos = positions.get(finger_name)
            fret, string = pos if pos is not None else (0, 0)
            row[f"{finger_name}_fret"] = fret
            row[f"{finger_name}_string"] = string

        if orientation_result is not None:
            row["index_angle"] = round(orientation_result.angle_deg, 2)
            row["index_span"] = round(orientation_result.span_estimate, 2)
            row["is_barre"] = int(orientation_result.is_barre)
        else:
            row["index_angle"] = 90.0  # default: not barre-like
            row["index_span"] = 0.0
            row["is_barre"] = 0

        # Wrist-normalized hand shape features — homography-independent
        row.update(build_landmark_feature_dict(landmarks))

        return row

    def _print_summary(self, df: pd.DataFrame, chord_folders: dict):
        print("\n" + "=" * 50)
        print("Dataset build complete")
        print("=" * 50)
        print(f"Total usable rows: {len(df)}")
        print(f"Saved to: {OUTPUT_CSV}")

        print("\nPer-chord counts:")
        if len(df) > 0:
            print(df["label"].value_counts().to_string())

        total_skipped = sum(self.skip_reasons.values())
        if total_skipped > 0:
            print(f"\nSkipped {total_skipped} images:")
            for reason, count in self.skip_reasons.items():
                if count > 0:
                    print(f"  {reason}: {count}")
        print("=" * 50)


# ------------------------------------------------------------------
# Run: python -m train.build_dataset
# ------------------------------------------------------------------
if __name__ == "__main__":
    builder = DatasetBuilder()
    builder.build()