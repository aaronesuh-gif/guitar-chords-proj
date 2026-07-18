"""
coordinate_mapper.py
---------------------
Bridges the classical CV layer (Phase 1 homography) and the landmark
layer (Phase 2 MediaPipe). Takes raw fingertip pixel positions and
projects them through the homography matrix H to produce interpretable
(fret, string) coordinates.

This is the file that turns "fingertip at pixel (412, 288)" into
"index finger is on fret 2, string 3" — which is what makes the whole
system geometrically grounded rather than just a black-box classifier
on raw pixel positions.

Usage:
    from src.vision.coordinate_mapper import CoordinateMapper
    mapper = CoordinateMapper(homography_computer)
    finger_positions = mapper.map_landmarks(landmarks, fingertip_dict)
"""

import numpy as np
from src.localization.homography import HomographyComputer
from src.vision.hand_tracker import HandTracker


class CoordinateMapper:
    def __init__(self, homography_computer: HomographyComputer):
        """
        Parameters
        ----------
        homography_computer : an already-computed HomographyComputer
                               instance (from Phase 1). Must have .H set
                               via .compute() before mapping will work.
        """
        self.homography = homography_computer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_point(self, pixel_point: tuple[float, float]) -> tuple[int, int] | None:
        """
        Map a single pixel coordinate to a (fret, string) pair.

        Parameters
        ----------
        pixel_point : (x, y) in the original camera frame's pixel space

        Returns
        -------
        (fret, string) as 1-indexed integers, or None if no homography
        has been computed yet (e.g. fretboard not detected this frame).
        """
        if self.homography.H is None:
            return None

        warped_x, warped_y = self.homography.warp_point(
            pixel_point, self.homography.H
        )
        return self.homography.fretboard_to_fret_string(warped_x, warped_y)

    def map_landmarks(
        self,
        landmarks: np.ndarray,
        fingertip_indices: list[int] = None,
        fingertip_names: list[str] = None,
    ) -> dict[str, tuple[int, int] | None]:
        """
        Map all fingertip landmarks to (fret, string) coordinates at once.

        Parameters
        ----------
        landmarks         : (21, 3) array from HandTracker.process()
        fingertip_indices : which landmark indices to map. Defaults to
                             HandTracker.FINGERTIPS (all 5 fingertips).
        fingertip_names   : matching labels for the output dict. Defaults
                             to HandTracker.FINGERTIP_NAMES.

        Returns
        -------
        dict like {"thumb": (fret, string), "index": (2, 3), ...}
        Values are None for any finger that fails to map (e.g. off the
        fretboard region, or homography not yet computed).
        """
        if fingertip_indices is None:
            fingertip_indices = HandTracker.FINGERTIPS
        if fingertip_names is None:
            fingertip_names = HandTracker.FINGERTIP_NAMES

        result = {}
        for name, idx in zip(fingertip_names, fingertip_indices):
            x, y = landmarks[idx][0], landmarks[idx][1]
            result[name] = self.map_point((x, y))

        return result

    def map_all_joints(
        self, landmarks: np.ndarray
    ) -> list[tuple[int, int] | None]:
        """
        Map all 21 landmarks (not just fingertips) to (fret, string).
        Useful if you later want knuckle positions too — e.g. for a
        richer feature vector, or for drawing the full hand skeleton
        in fretboard-aligned space.

        Returns
        -------
        list of 21 (fret, string) tuples (or None), indexed the same
        way as the input landmarks array.
        """
        return [
            self.map_point((landmarks[i][0], landmarks[i][1]))
            for i in range(len(landmarks))
        ]

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    def format_positions(
        self, positions: dict[str, tuple[int, int] | None]
    ) -> str:
        """
        Format a fingertip position dict as a readable string for
        printing/debugging, e.g. "index:F2S3 middle:F2S4 ring:F2S5".
        """
        parts = []
        for name, pos in positions.items():
            if pos is None:
                parts.append(f"{name}:--")
            else:
                fret, string = pos
                parts.append(f"{name}:F{fret:.1f}S{string:.1f}")
        return " ".join(parts)


# ------------------------------------------------------------------
# Standalone test — python src/vision/coordinate_mapper.py
# Full pipeline: edges -> hough -> homography -> landmarks -> mapping.
# Prints live (fret, string) per fingertip to the terminal and overlay.
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
    mapper = CoordinateMapper(homog)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Full pipeline test. Hold a chord shape. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phase 1: fretboard localization
        edges = edge_det.process(frame)
        frets, strings = hough_det.detect(edges, frame.shape)
        homog.compute(frets, strings, frame.shape)

        display = frame.copy()

        # Phase 2: hand landmarks + mapping
        landmarks = tracker.process(frame)
        if landmarks is not None:
            positions = mapper.map_landmarks(landmarks)

            # Draw fingertip dots + labels
            for name, idx in zip(HandTracker.FINGERTIP_NAMES, HandTracker.FINGERTIPS):
                x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                pos = positions[name]
                label = f"F{pos[0]}S{pos[1]}" if pos else "--"
                cv2.putText(display, label, (x + 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            print(mapper.format_positions(positions))

        cv2.imshow("CoordinateMapper", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()