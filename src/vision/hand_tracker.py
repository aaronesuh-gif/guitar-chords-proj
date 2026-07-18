"""
hand_tracker.py
----------------
Thin wrapper around MediaPipe's HandLandmarker (Tasks API). Takes a raw
camera frame and returns 21 3D hand landmarks (fingertip, knuckle, and
joint positions) in both normalized [0,1] form and pixel form.

NOTE ON API VERSION:
Google removed the legacy `mp.solutions.hands` API in recent MediaPipe
releases (0.10.30+). This file uses the current replacement, the
"Tasks" API (`mp.tasks.vision.HandLandmarker`), which requires a
separate model file (hand_landmarker.task) downloaded once and stored
locally. See the setup instructions below.

SETUP (run once, from your project root):
    curl -o models/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Usage:
    from src.vision.hand_tracker import HandTracker
    tracker = HandTracker()
    landmarks = tracker.process(frame)
    if landmarks is not None:
        fingertip = landmarks[HandTracker.INDEX_TIP]
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../../models/hand_landmarker.task"
)


class HandTracker:
    # MediaPipe's 21-point hand landmark indices.
    # Naming these as class constants avoids "magic numbers" scattered
    # throughout the rest of the codebase.
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # Convenience groupings — useful in coordinate_mapper.py and orientation.py
    FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    FINGERTIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

    # Connections between landmark indices, for drawing the hand skeleton.
    # Same topology MediaPipe's old drawing_utils used internally.
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (5, 9), (9, 10), (10, 11), (11, 12),     # middle
        (9, 13), (13, 14), (14, 15), (15, 16),   # ring
        (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (0, 17),                                  # palm base
    ]

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: str = DEFAULT_MODEL_PATH,
    ):
        """
        Parameters
        ----------
        max_num_hands : only need 1 for fretting-hand detection. Keeping
                        this at 1 also improves speed.
        min_detection_confidence : threshold for the initial hand detection
                        model. Raise if you get false positives (detecting
                        a "hand" where there isn't one).
        min_tracking_confidence  : threshold for the frame-to-frame tracking
                        model. Lower values track through motion blur better
                        but may drift; raise if landmarks jitter.
        static_image_mode : set True only when running on individual still
                        images (e.g. in build_dataset.py). Uses IMAGE running
                        mode (full detection every call). False for video —
                        uses VIDEO running mode, which is faster because it
                        tracks between frames instead of re-detecting fully.
        model_path : path to the downloaded hand_landmarker.task file.
                        See the module docstring for the download command.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}\n"
                f"Download it with:\n"
                f"  curl -o {model_path} "
                f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                f"hand_landmarker/float16/1/hand_landmarker.task"
            )

        self.static_image_mode = static_image_mode
        running_mode = RunningMode.IMAGE if static_image_mode else RunningMode.VIDEO

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # VIDEO running mode requires monotonically increasing timestamps
        # per call. We generate our own since frames don't carry one.
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Run hand detection on a single BGR frame.

        Parameters
        ----------
        frame : BGR image from cv2.VideoCapture or cv2.imread

        Returns
        -------
        landmarks : (21, 3) numpy array of (x, y, z) in PIXEL coordinates
                    (x, y in pixels; z is MediaPipe's relative depth,
                    roughly in the same scale as x).
                    Returns None if no hand was detected in this frame.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detect(mp_image)

        if not result.hand_landmarks:
            return None

        # Only take the first detected hand (max_num_hands=1 by default)
        hand = result.hand_landmarks[0]
        landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand])
        return landmarks

    def process_normalized(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Same as process(), but returns landmarks in MediaPipe's native
        normalized [0, 1] coordinate space instead of pixels.

        Returns
        -------
        landmarks : (21, 3) numpy array of (x, y, z) normalized to [0, 1]
                    Returns None if no hand was detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._detect(mp_image)

        if not result.hand_landmarks:
            return None

        hand = result.hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand])
        return landmarks

    def get_fingertip_positions(
        self, landmarks: np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """
        Convenience method — extract just the 5 fingertip (x, y) pixel
        positions as a labeled dict.

        Parameters
        ----------
        landmarks : (21, 3) array from process()

        Returns
        -------
        dict like {"thumb": (x, y), "index": (x, y), ...}
        """
        return {
            name: (landmarks[idx][0], landmarks[idx][1])
            for name, idx in zip(self.FINGERTIP_NAMES, self.FINGERTIPS)
        }

    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw the hand skeleton on a copy of the frame using pixel-space
        landmarks from process(). Replaces the old drawing_utils helper,
        which was tied to the removed solutions API.
        """
        out = frame.copy()
        for a, b in self.CONNECTIONS:
            pt_a = tuple(landmarks[a][:2].astype(int))
            pt_b = tuple(landmarks[b][:2].astype(int))
            cv2.line(out, pt_a, pt_b, (0, 200, 0), 2)
        for x, y, _ in landmarks:
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
        return out

    def close(self):
        """Release MediaPipe resources. Call when done, e.g. on app exit."""
        self.landmarker.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect(self, mp_image: "mp.Image"):
        """
        Run detection using the correct method for the configured
        running mode. IMAGE mode is stateless (one-off images). VIDEO
        mode requires an increasing millisecond timestamp per call.
        """
        if self.static_image_mode:
            return self.landmarker.detect(mp_image)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)


# ------------------------------------------------------------------
# Standalone test — python -m src.vision.hand_tracker
# Shows live hand landmark detection with the skeleton drawn.
# Prints fingertip pixel positions to terminal.
# ------------------------------------------------------------------
if __name__ == "__main__":
    tracker = HandTracker(max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Showing hand landmarks. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        landmarks = tracker.process(frame)

        if landmarks is not None:
            display = tracker.draw_landmarks(display, landmarks)
            tips = tracker.get_fingertip_positions(landmarks)
            y_offset = 20
            for name, (x, y) in tips.items():
                text = f"{name}: ({int(x)}, {int(y)})"
                cv2.putText(display, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20

        cv2.imshow("HandTracker", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()