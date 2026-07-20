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
    def __init__(self, n_frets: int = 5, n_strings: int = 6, corner_smoothing: float = 0.6):
        """
        Parameters
        ----------
        n_frets   : number of frets visible in frame. Used to compute
                    fret spacing in the normalized output space.
        n_strings : number of guitar strings (always 6 for standard guitar).
        corner_smoothing : EMA factor for corner stabilization, 0-1.
                    Higher = trust new detections more (less lag, more
                    jitter). Lower = smoother but slower to follow
                    guitar movement. 0.6 is a reasonable middle.
        """
        self.n_frets = n_frets
        self.n_strings = n_strings
        self.corner_smoothing = corner_smoothing
        self.H: np.ndarray | None = None
        self.H_inv: np.ndarray | None = None

        # Smoothed corner state (EMA across frames) — reduces the
        # frame-to-frame homography jitter that made coordinates wobble
        self._smoothed_corners: np.ndarray | None = None
        self.last_corners: np.ndarray | None = None

        # Actual detected line positions in warped space — used for
        # accurate (non-uniform) fret/string mapping
        self._fret_xs: np.ndarray | None = None
        self._string_ys: np.ndarray | None = None

        # Persistence: when detection fails for a frame, keep serving
        # the last good H for up to this many frames instead of
        # immediately dropping to None. Hough detection flickers
        # frame-to-frame under real lighting; the guitar itself barely
        # moves in 15 frames (~0.5s), so a briefly stale H is far
        # better than no H.
        self.max_stale_frames = 15
        self._stale_count = 0

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
            return self._handle_detection_failure()

        corners_src = self._extract_corners(frets, strings)
        if corners_src is None or not self._corners_valid(corners_src, frame_shape):
            return self._handle_detection_failure()

        self._stale_count = 0

        # EMA smoothing on corners — stabilizes the homography across
        # frames so mapped coordinates don't jitter with Hough noise.
        if self._smoothed_corners is not None:
            a = self.corner_smoothing
            corners_src = a * corners_src + (1 - a) * self._smoothed_corners
        self._smoothed_corners = corners_src
        self.last_corners = corners_src

        corners_dst = self._destination_corners()

        H, _ = cv2.findHomography(corners_src, corners_dst)
        if H is None:
            return None, None

        H_inv = np.linalg.inv(H)
        self.H = H
        self.H_inv = H_inv

        # Store the ACTUAL warped positions of each detected fret and
        # string line — real fret spacing is geometric, not uniform, so
        # mapping against true line positions is more accurate than
        # dividing the warped space into equal bins.
        self._compute_reference_positions(frets, strings)

        return H, H_inv

    def _corners_valid(self, corners: np.ndarray, frame_shape: tuple) -> bool:
        """
        Sanity-check an extracted corner quad before trusting it.
        Rejects degenerate results (tiny slivers, corners far outside
        the frame, near-zero area) that would otherwise get blended
        into the EMA and poison the smoothed homography for many
        frames afterward.
        """
        h, w = frame_shape[:2]

        # Corners shouldn't be wildly outside the frame — allow some
        # margin since the neck can extend past the frame edge
        margin = 0.5
        xs, ys = corners[:, 0], corners[:, 1]
        if (xs < -w * margin).any() or (xs > w * (1 + margin)).any():
            return False
        if (ys < -h * margin).any() or (ys > h * (1 + margin)).any():
            return False

        # Quad area must be a meaningful fraction of the frame —
        # shoelace formula on [tl, tr, br, bl] ordering
        tl, tr, bl, br = corners
        quad = np.array([tl, tr, br, bl])
        x, y = quad[:, 0], quad[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        if area < 0.01 * w * h:  # less than 1% of the frame = sliver
            return False

        return True

    def _handle_detection_failure(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Called when this frame's line detection wasn't good enough to
        compute a fresh H. Serves the previous H for up to
        max_stale_frames, then invalidates everything so we don't keep
        mapping against a fretboard position that's no longer accurate.
        """
        if self.H is not None and self._stale_count < self.max_stale_frames:
            self._stale_count += 1
            return self.H, self.H_inv

        # Too stale — full reset
        self.H = None
        self.H_inv = None
        self._smoothed_corners = None
        self.last_corners = None
        self._fret_xs = None
        self._string_ys = None
        return None, None

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
        Convert a point in normalized fretboard space to a continuous
        (fret, string) pair.

        When actual detected line positions are available, interpolates
        against them (accurate — respects real geometric fret spacing).
        Falls back to uniform division otherwise.

        Returns
        -------
        (fret, string) — 1-indexed floats, clamped to valid range.
        """
        if self._fret_xs is not None and len(self._fret_xs) >= 2:
            # Interpolate against real detected fret line x positions:
            # a point at fret line i maps to i+1; between lines maps
            # to a fractional value respecting actual spacing.
            indices = np.arange(1, len(self._fret_xs) + 1, dtype=float)
            fret = float(np.interp(warped_x, self._fret_xs, indices))
            max_fret = float(len(self._fret_xs))
        else:
            fret = warped_x / WARP_WIDTH * self.n_frets + 1
            max_fret = float(self.n_frets)

        if self._string_ys is not None and len(self._string_ys) >= 2:
            indices = np.arange(1, len(self._string_ys) + 1, dtype=float)
            string = float(np.interp(warped_y, self._string_ys, indices))
            max_string = float(len(self._string_ys))
        else:
            string = warped_y / WARP_HEIGHT * self.n_strings + 1
            max_string = float(self.n_strings)

        fret = max(1.0, min(fret, max_fret))
        string = max(1.0, min(string, max_string))

        return round(fret, 2), round(string, 2)

    def _compute_reference_positions(self, frets: list[Line], strings: list[Line]):
        """
        Warp each detected fret line and string line into normalized
        space and store their sorted x (frets) / y (strings) positions.

        For each fret line: intersect with the top and bottom strings,
        warp both intersection points, average their x. For each string
        line: intersect with the left/right frets, warp, average y.
        """
        top_string, bottom_string = strings[0], strings[-1]
        left_fret, right_fret = frets[0], frets[-1]

        fret_xs = []
        for fret_line in frets:
            pts = [
                self._intersect(fret_line, top_string),
                self._intersect(fret_line, bottom_string),
            ]
            xs = []
            for pt in pts:
                if pt is not None:
                    wx, _ = self.warp_point(pt, self.H)
                    xs.append(wx)
            if xs:
                fret_xs.append(float(np.mean(xs)))

        string_ys = []
        for string_line in strings:
            pts = [
                self._intersect(string_line, left_fret),
                self._intersect(string_line, right_fret),
            ]
            ys = []
            for pt in pts:
                if pt is not None:
                    _, wy = self.warp_point(pt, self.H)
                    ys.append(wy)
            if ys:
                string_ys.append(float(np.mean(ys)))

        self._fret_xs = np.array(sorted(fret_xs)) if len(fret_xs) >= 2 else None
        self._string_ys = np.array(sorted(string_ys)) if len(string_ys) >= 2 else None

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