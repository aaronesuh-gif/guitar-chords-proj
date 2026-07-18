"""
homography.py
-------------
Uses the fret wire and string lines from HoughDetector to compute a
perspective homography matrix H that warps the camera frame into a
consistent top-down fretboard coordinate space.

Once H is computed, any pixel coordinate (fingertip, knuckle) can be
projected into the normalized fretboard space and mapped to a
(fret, string) integer pair.

Usage:
    from src.localization.homography import HomographyComputer
    computer = HomographyComputer(n_frets=5, n_strings=6)
    H, H_inv = computer.compute(frets, strings, frame_shape)
    fretboard_pt = computer.warp_point(pixel_pt, H)
    pixel_pt     = computer.warp_point(fretboard_pt, H_inv)
"""

import cv2
import numpy as np
from src.localization.hough_lines import Line


# Output canvas size for the normalized fretboard view (pixels)
# These are arbitrary — they just define the coordinate space you
# map into. Larger = more resolution in fret/string space.
WARP_WIDTH = 400
WARP_HEIGHT = 300


class HomographyComputer:
    def __init__(self, n_frets: int = 5, n_strings: int = 6):
        """
        Parameters
        ----------
        n_frets   : number of frets visible in frame. Used to compute
                    fret spacing in the normalized output space.
        n_strings : number of guitar strings (always 6 for standard guitar).
        """
        self.n_frets = n_frets
        self.n_strings = n_strings
        self.H: np.ndarray | None = None
        self.H_inv: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        frets: list[Line],
        strings: list[Line],
        frame_shape: tuple,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Compute the homography H from the detected fret and string lines.

        Requires at least 2 fret lines and 2 string lines to define
        the four corners of the fretboard region.

        Parameters
        ----------
        frets       : fret wire Line objects, sorted left-to-right
        strings     : string Line objects, sorted top-to-bottom
        frame_shape : (height, width) of the original camera frame

        Returns
        -------
        H     : 3x3 homography matrix (frame → normalized fretboard)
        H_inv : 3x3 inverse homography (normalized fretboard → frame)
        Both are None if not enough lines were detected.
        """
        if len(frets) < 2 or len(strings) < 2:
            return None, None

        corners_src = self._extract_corners(frets, strings)
        if corners_src is None:
            return None, None

        corners_dst = self._destination_corners()

        H, _ = cv2.findHomography(corners_src, corners_dst)
        if H is None:
            return None, None

        H_inv = np.linalg.inv(H)
        self.H = H
        self.H_inv = H_inv
        return H, H_inv

    def warp_point(
        self, point: tuple[float, float], H: np.ndarray
    ) -> tuple[float, float]:
        """
        Project a single (x, y) point through the homography matrix H.

        Works in both directions:
          - H     → frame pixel to normalized fretboard space
          - H_inv → normalized fretboard space back to frame pixel

        Parameters
        ----------
        point : (x, y) coordinate to transform
        H     : 3x3 homography matrix

        Returns
        -------
        (x', y') transformed coordinate
        """
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, H)
        return float(warped[0][0][0]), float(warped[0][0][1])

    def warp_frame(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Warp the entire camera frame into the normalized fretboard view.
        Useful for debugging — lets you see what the model "sees".

        Returns None if H has not been computed yet.
        """
        if self.H is None:
            return None
        return cv2.warpPerspective(frame, self.H, (WARP_WIDTH, WARP_HEIGHT))

    def fretboard_to_fret_string(
        self, warped_x: float, warped_y: float
    ) -> tuple[float, float]:
        """
        Convert a point in normalized fretboard space to a (fret, string)
        pair — CONTINUOUS, not snapped to integers.

        Returning fractional values (e.g. string 2.6) instead of hard
        integer bins preserves fine positional differences. This matters
        most for chords that differ by a single string shift (A vs D):
        integer snapping forces a borderline finger to one bin or the
        other, destroying exactly the signal that separates those chords.
        The MLP learns finer boundaries from continuous inputs.

        Returns
        -------
        (fret, string) — 1-indexed floats, clamped to valid range.
        """
        fret = warped_x / WARP_WIDTH * self.n_frets + 1
        string = warped_y / WARP_HEIGHT * self.n_strings + 1

        fret = max(1.0, min(float(fret), float(self.n_frets)))
        string = max(1.0, min(float(string), float(self.n_strings)))

        return round(fret, 2), round(string, 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_corners(
        self, frets: list[Line], strings: list[Line]
    ) -> np.ndarray | None:
        """
        Find the four corners of the fretboard by intersecting:
          - leftmost fret  with top string    → top-left corner
          - rightmost fret with top string    → top-right corner
          - leftmost fret  with bottom string → bottom-left corner
          - rightmost fret with bottom string → bottom-right corner

        Returns a (4, 2) float32 array of corner pixel coordinates,
        or None if any intersection fails.
        """
        left_fret = frets[0]
        right_fret = frets[-1]
        top_string = strings[0]
        bottom_string = strings[-1]

        tl = self._intersect(left_fret, top_string)
        tr = self._intersect(right_fret, top_string)
        bl = self._intersect(left_fret, bottom_string)
        br = self._intersect(right_fret, bottom_string)

        if any(pt is None for pt in [tl, tr, bl, br]):
            return None

        return np.float32([tl, tr, bl, br])

    def _destination_corners(self) -> np.ndarray:
        """
        The four corners of the output (normalized) fretboard canvas.
        Always the same — top-left, top-right, bottom-left, bottom-right
        of the WARP_WIDTH x WARP_HEIGHT rectangle.
        """
        return np.float32([
            [0, 0],
            [WARP_WIDTH, 0],
            [0, WARP_HEIGHT],
            [WARP_WIDTH, WARP_HEIGHT],
        ])

    def _intersect(
        self, line_a: Line, line_b: Line
    ) -> tuple[float, float] | None:
        """
        Find the pixel intersection of two lines using the cross-product
        method on their homogeneous representations.

        Two lines in homogeneous form: l = (a, b, c) where ax + by + c = 0.
        Intersection point p = l1 × l2 (cross product), then dehomogenize.

        Returns None if lines are parallel (no intersection).
        """
        a1, b1, c1 = self._line_to_homogeneous(line_a)
        a2, b2, c2 = self._line_to_homogeneous(line_b)

        # Cross product gives the intersection in homogeneous coords
        x = b1 * c2 - b2 * c1
        y = a2 * c1 - a1 * c2
        w = a1 * b2 - a2 * b1

        if abs(w) < 1e-6:
            # Lines are parallel — no intersection
            return None

        return x / w, y / w

    @staticmethod
    def _line_to_homogeneous(line: Line) -> tuple[float, float, float]:
        """
        Convert a Line (two endpoints) to homogeneous form (a, b, c)
        where ax + by + c = 0.

        Derived from the two-point form of a line:
          (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x1 = 0
        """
        x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
        a = float(y2 - y1)
        b = float(x1 - x2)
        c = float(x2 * y1 - x1 * y2)
        return a, b, c

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def draw_corners(
        self, frame: np.ndarray, frets: list[Line], strings: list[Line]
    ) -> np.ndarray:
        """
        Draw the four computed fretboard corners on the frame.
        Green circles = corners, yellow lines = fretboard boundary.
        """
        out = frame.copy()
        if len(frets) < 2 or len(strings) < 2:
            return out

        corners = self._extract_corners(frets, strings)
        if corners is None:
            return out

        pts = corners.astype(int)
        # Draw boundary
        cv2.polylines(out, [pts[[0, 1, 3, 2]]], isClosed=True, color=(0, 255, 255), thickness=2)
        # Draw corner dots
        for pt in pts:
            cv2.circle(out, tuple(pt), 6, (0, 255, 0), -1)

        return out

    def debug_view(
        self,
        frame: np.ndarray,
        frets: list[Line],
        strings: list[Line],
    ) -> np.ndarray:
        """
        Side-by-side: annotated frame with corners | warped fretboard view.
        """
        corner_frame = self.draw_corners(frame, frets, strings)
        warped = self.warp_frame(frame)

        if warped is None:
            warped = np.zeros((WARP_HEIGHT, WARP_WIDTH, 3), dtype=np.uint8)
            cv2.putText(warped, "No H computed", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize warped to match frame height for side-by-side display
        h = frame.shape[0]
        scale = h / WARP_HEIGHT
        warped_resized = cv2.resize(
            warped, (int(WARP_WIDTH * scale), h)
        )
        return np.hstack([corner_frame, warped_resized])


# ------------------------------------------------------------------
# Standalone test — python src/localization/homography.py
# Shows detected corners and the warped top-down fretboard view.
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

    from src.localization.edge_detector import EdgeDetector
    from src.localization.hough_lines import HoughDetector

    edge_det = EdgeDetector()
    hough_det = HoughDetector(threshold=80)
    homog = HomographyComputer(n_frets=5, n_strings=6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Yellow box = detected fretboard region | Right panel = warped view")
    print("Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = edge_det.process(frame)
        frets, strings = hough_det.detect(edges, frame.shape)
        homog.compute(frets, strings, frame.shape)

        display = homog.debug_view(frame, frets, strings)
        cv2.imshow("Homography", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()