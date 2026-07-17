"""
orientation.py
---------------
Solves the barre-chord ambiguity problem.

The problem: a single-note fretting finger and a barring finger can
both produce the same fingertip (fret, string) position — the index
fingertip at "fret 1, string 6" says nothing about whether the rest of
the finger is lying flat across all 6 strings (a barre) or pressing
just that one string (a single note).

Position alone can't distinguish these. The signal that can is
orientation: a barring finger lies nearly PARALLEL to the fret wire
(across the strings), while a fretting finger presses roughly
PERPENDICULAR to the fret wire (down onto one string).

This file computes that angle and uses it to flag barre chords.

Usage:
    from src.vision.orientation import OrientationAnalyzer
    analyzer = OrientationAnalyzer()
    result = analyzer.analyze_index_finger(landmarks, frets)
"""

import numpy as np
from src.vision.hand_tracker import HandTracker
from src.localization.hough_lines import Line


class OrientationResult:
    """Container for the orientation analysis of one finger in one frame."""

    def __init__(
        self,
        angle_deg: float,
        is_barre: bool,
        span_estimate: float,
    ):
        """
        Parameters
        ----------
        angle_deg     : angle in degrees between the finger's long axis
                         (MCP → fingertip) and the fret wire direction.
                         0° = perfectly parallel to fret (barre-like).
                         90° = perfectly perpendicular (single-note-like).
        is_barre      : True if this finger's orientation and estimated
                         span suggest a barre.
        span_estimate : rough estimate of how many string-widths the
                         finger's projected length covers. Larger values
                         support the barre classification.
        """
        self.angle_deg = angle_deg
        self.is_barre = is_barre
        self.span_estimate = span_estimate

    def __repr__(self):
        return (
            f"OrientationResult(angle={self.angle_deg:.1f}°, "
            f"is_barre={self.is_barre}, span={self.span_estimate:.1f})"
        )


class OrientationAnalyzer:
    def __init__(
        self,
        barre_angle_threshold: float = 20.0,
        barre_span_threshold: float = 3.0,
    ):
        """
        Parameters
        ----------
        barre_angle_threshold : maximum angle (degrees) from the fret-wire
                                 direction for a finger to be considered
                                 "lying flat" (barre-like). Lower = stricter.
        barre_span_threshold  : minimum estimated string-span for a finger
                                 to be considered a barre. A single-note
                                 press spans ~1 string; a barre spans most
                                 or all 6. 3.0 is a reasonable middle ground
                                 that tolerates partial (2-3 string) barres.
        """
        self.barre_angle_threshold = barre_angle_threshold
        self.barre_span_threshold = barre_span_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_index_finger(
        self,
        landmarks: np.ndarray,
        frets: list[Line],
    ) -> OrientationResult | None:
        """
        Analyze the index finger's orientation to detect a barre.

        The index finger is the one that performs barre chords, so this
        is the primary check. (Occasionally the ring or pinky finger
        barres a partial chord, but index-finger barres cover the vast
        majority of common chord shapes — F, B, Bm, F#m, etc.)

        Parameters
        ----------
        landmarks : (21, 3) array from HandTracker.process()
        frets     : detected fret Line objects from HoughDetector,
                    used to establish the "fret wire direction" reference

        Returns
        -------
        OrientationResult, or None if there aren't enough fret lines to
        establish a reference direction.
        """
        if len(frets) < 1:
            return None

        fret_direction = self._fret_direction_vector(frets)

        mcp = landmarks[HandTracker.INDEX_MCP][:2]
        tip = landmarks[HandTracker.INDEX_TIP][:2]
        finger_vector = tip - mcp

        angle = self._angle_between(finger_vector, fret_direction)
        span = self._estimate_span(mcp, tip)

        is_barre = (
            angle <= self.barre_angle_threshold
            and span >= self.barre_span_threshold
        )

        return OrientationResult(angle_deg=angle, is_barre=is_barre, span_estimate=span)

    def analyze_finger(
        self,
        landmarks: np.ndarray,
        finger_mcp_idx: int,
        finger_tip_idx: int,
        frets: list[Line],
    ) -> OrientationResult | None:
        """
        Generalized version of analyze_index_finger for any finger.
        Useful if you later want to detect partial barres performed by
        the ring or pinky finger.

        Parameters
        ----------
        landmarks       : (21, 3) array from HandTracker.process()
        finger_mcp_idx  : landmark index of that finger's MCP joint
                          (e.g. HandTracker.RING_MCP)
        finger_tip_idx  : landmark index of that finger's tip
                          (e.g. HandTracker.RING_TIP)
        frets           : detected fret lines for the reference direction

        Returns
        -------
        OrientationResult, or None if not enough fret lines detected.
        """
        if len(frets) < 1:
            return None

        fret_direction = self._fret_direction_vector(frets)
        mcp = landmarks[finger_mcp_idx][:2]
        tip = landmarks[finger_tip_idx][:2]
        finger_vector = tip - mcp

        angle = self._angle_between(finger_vector, fret_direction)
        span = self._estimate_span(mcp, tip)
        is_barre = (
            angle <= self.barre_angle_threshold
            and span >= self.barre_span_threshold
        )

        return OrientationResult(angle_deg=angle, is_barre=is_barre, span_estimate=span)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fret_direction_vector(self, frets: list[Line]) -> np.ndarray:
        """
        Compute a unit vector representing the direction fret wires run.
        Uses the first detected fret line as the reference — in a
        correctly-warped frame all fret wires should be roughly parallel,
        so any one of them is a valid reference.

        Returns
        -------
        2D unit vector (dx, dy) along the fret wire's direction.
        """
        line = frets[0]
        vec = np.array([line.x2 - line.x1, line.y2 - line.y1], dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([0.0, 1.0])  # fallback: assume vertical
        return vec / norm

    def _angle_between(
        self, finger_vector: np.ndarray, fret_direction: np.ndarray
    ) -> float:
        """
        Compute the acute angle (0-90°) between the finger's long axis
        and the fret wire direction.

        A barre finger lies ALONG the strings, which run PERPENDICULAR
        to the fret wires. So:
          - finger_vector parallel to fret_direction  → finger is vertical
            (pressing straight down on one fret) → single-note-like
          - finger_vector perpendicular to fret_direction → finger lies
            across the strings → barre-like

        We report the angle relative to fret_direction, then interpret:
        small angle = single-note-like, large angle (~90°) = barre-like.
        NOTE: this is inverted from the docstring framing above for
        clarity in the actual comparison — see is_barre logic below,
        which checks against fret_direction rotated 90° (i.e. the
        STRING direction) so that "small angle from string direction"
        correctly means "barre-like".
        """
        finger_norm = np.linalg.norm(finger_vector)
        if finger_norm < 1e-6:
            return 90.0  # degenerate case, treat as non-barre

        finger_unit = finger_vector / finger_norm

        # Rotate fret_direction by 90° to get the STRING direction —
        # a barre finger should align with strings, not frets.
        string_direction = np.array([-fret_direction[1], fret_direction[0]])

        cos_angle = np.clip(np.dot(finger_unit, string_direction), -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_angle)))  # abs -> angle in [0,90]
        return angle

    def _estimate_span(
        self, mcp: np.ndarray, tip: np.ndarray
    ) -> float:
        """
        Rough proxy for how many strings the finger covers, based on
        the pixel distance between MCP and fingertip landmarks.

        This is a coarse heuristic, not a precise measurement — a longer
        visible finger segment is more consistent with a flattened barre
        posture than a curled single-note press. Calibrate the divisor
        against your own hand/camera setup by printing span_estimate
        values for known barre vs. non-barre frames.

        Returns
        -------
        float — larger values suggest more string coverage.
        """
        distance = np.linalg.norm(tip - mcp)
        # Rough normalization: divide by an approximate single-string-width
        # in pixels. This constant should be tuned per camera setup —
        # start here and adjust based on printed debug values.
        approx_string_width_px = 25.0
        return distance / approx_string_width_px


# ------------------------------------------------------------------
# Standalone test — python src/vision/orientation.py
# Full pipeline test: hold a normal chord vs. an F barre chord and
# watch the angle / is_barre flag change live.
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

    import cv2
    from src.localization.edge_detector import EdgeDetector
    from src.localization.hough_lines import HoughDetector
    from src.localization.homography import HomographyComputer
    from src.vision.hand_tracker import HandTracker

    edge_det = EdgeDetector()
    hough_det = HoughDetector(threshold=80)
    homog = HomographyComputer(n_frets=5, n_strings=6)
    tracker = HandTracker(max_num_hands=1)
    analyzer = OrientationAnalyzer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Try a normal chord, then an F barre chord. Watch is_barre flip.")
    print("Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = edge_det.process(frame)
        frets, strings = hough_det.detect(edges, frame.shape)
        homog.compute(frets, strings, frame.shape)

        display = frame.copy()
        landmarks = tracker.process(frame)

        if landmarks is not None and len(frets) >= 1:
            result = analyzer.analyze_index_finger(landmarks, frets)
            if result is not None:
                color = (0, 0, 255) if result.is_barre else (0, 255, 0)
                text = f"angle:{result.angle_deg:.1f} span:{result.span_estimate:.1f} barre:{result.is_barre}"
                cv2.putText(display, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw the index finger vector for visual reference
                mcp = landmarks[HandTracker.INDEX_MCP][:2].astype(int)
                tip = landmarks[HandTracker.INDEX_TIP][:2].astype(int)
                cv2.line(display, tuple(mcp), tuple(tip), color, 2)

        cv2.imshow("OrientationAnalyzer", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()