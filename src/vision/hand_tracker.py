"""
hand_tracker.py
----------------
Thin wrapper around MediaPipe Hands. Takes a raw camera frame and
returns 21 3D hand landmarks (fingertip, knuckle, and joint positions)
in both normalized [0,1] form and pixel form.

Usage:
    from src.vision.hand_tracker import HandTracker
    tracker = HandTracker()
    landmarks = tracker.process(frame)
    if landmarks is not None:
        fingertip = landmarks[HandTracker.INDEX_TIP]
"""

import cv2
import mediapipe as mp
import numpy as np


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

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
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
                        images (e.g. in build_dataset.py). False for video —
                        it enables MediaPipe's faster tracking-based mode
                        instead of running full detection every frame.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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

        # MediaPipe expects RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # Only take the first detected hand (max_num_hands=1 by default)
        hand = results.multi_hand_landmarks[0]

        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]  # scale z by width, MediaPipe convention
            for lm in hand.landmark
        ])

        return landmarks

    def process_normalized(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Same as process(), but returns landmarks in MediaPipe's native
        normalized [0, 1] coordinate space instead of pixels.

        Useful when you want coordinates independent of frame resolution
        (e.g. for the feature vector fed to the classifier, since pixel
        coordinates would make the model resolution-dependent).

        Returns
        -------
        landmarks : (21, 3) numpy array of (x, y, z) normalized to [0, 1]
                    Returns None if no hand was detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        return landmarks

    def get_fingertip_positions(
        self, landmarks: np.ndarray
    ) -> dict[str, tuple[float, float]]:
        """
        Convenience method — extract just the 5 fingertip (x, y) pixel
        positions as a labeled dict. Drops z since most downstream code
        (coordinate_mapper.py) only needs 2D pixel position to project
        through the homography.

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

    def draw_landmarks(
        self, frame: np.ndarray, landmarks_raw
    ) -> np.ndarray:
        """
        Draw the MediaPipe hand skeleton on a copy of the frame using
        MediaPipe's built-in drawing utility. Requires the raw MediaPipe
        result object, not the numpy array from process().

        This is mainly for quick visual debugging — the real overlay
        (Phase 6) draws a custom skeleton instead so it can be styled
        and combined with the fretboard grid.
        """
        out = frame.copy()
        self.mp_drawing.draw_landmarks(
            out,
            landmarks_raw,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )
        return out

    def close(self):
        """Release MediaPipe resources. Call when done, e.g. on app exit."""
        self.hands.close()


# ------------------------------------------------------------------
# Standalone test — python src/vision/hand_tracker.py
# Shows live hand landmark detection with the skeleton drawn.
# Prints fingertip pixel positions to terminal.
# ------------------------------------------------------------------
if __name__ == "__main__":
    tracker = HandTracker(max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Showing hand landmarks. Press Q to quit.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run raw MediaPipe here too, just for the built-in drawing utility
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.hands.process(rgb)

        display = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        landmarks = tracker.process(frame)
        if landmarks is not None:
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