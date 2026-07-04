"""
hough_lines.py
--------------
Takes the binary edge map from EdgeDetector and finds the dominant
straight lines corresponding to fret wires and guitar strings.

Raw Hough output is noisy — a single fret wire often produces 10-20
near-duplicate lines. This file clusters those duplicates and returns
one clean line per physical fret wire / string.

Usage:
    from src.localization.hough_lines import HoughDetector
    detector = HoughDetector()
    frets, strings = detector.detect(edges, frame_shape)
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class Line:
    """
    A line in (rho, theta) polar form as returned by HoughLines.

    rho   : perpendicular distance from origin to the line (pixels)
    theta : angle of the perpendicular in radians [0, pi)
    x1, y1, x2, y2 : two endpoint pixels for drawing
    """
    rho: float
    theta: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def angle_deg(self) -> float:
        """Angle of the LINE itself (not the perpendicular) in degrees."""
        return np.degrees(self.theta) - 90


class HoughDetector:
    def __init__(
        self,
        rho: float = 1.0,
        theta_deg: float = 0.5,
        threshold: int = 80,
        cluster_rho_px: float = 15.0,
        cluster_theta_deg: float = 5.0,
        fret_angle_tolerance: float = 20.0,
        string_angle_tolerance: float = 20.0,
    ):
        """
        Parameters
        ----------
        rho               : Hough accumulator resolution in pixels. 1.0 is standard.
        theta_deg         : Hough accumulator angular resolution in degrees.
                            0.5 gives finer angle discrimination than the default 1.0.
        threshold         : Minimum accumulator votes for a line to be returned.
                            Lower = more lines detected (noisier). Raise if you get
                            too many spurious lines; lower if frets are missed.
        cluster_rho_px    : Two lines are in the same cluster if their rho values
                            are within this many pixels. Controls how aggressively
                            near-duplicate lines are merged.
        cluster_theta_deg : Two lines are in the same cluster if their theta values
                            are within this many degrees.
        fret_angle_tolerance  : Fret wires run roughly vertical (90° in line-angle
                                space). Lines within ±this many degrees of 90° are
                                classified as frets.
        string_angle_tolerance: String lines run roughly horizontal (0° or 180°).
                                Lines within ±this many degrees of 0/180° are
                                classified as strings.
        """
        self.rho = rho
        self.theta = np.deg2rad(theta_deg)
        self.threshold = threshold
        self.cluster_rho_px = cluster_rho_px
        self.cluster_theta_deg = cluster_theta_deg
        self.fret_angle_tolerance = fret_angle_tolerance
        self.string_angle_tolerance = string_angle_tolerance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, edges: np.ndarray, frame_shape: tuple
    ) -> tuple[list[Line], list[Line]]:
        """
        Run Hough detection, cluster duplicates, and split into frets/strings.

        Parameters
        ----------
        edges       : binary edge map from EdgeDetector.process()
        frame_shape : (height, width[, channels]) of the original frame —
                      used to compute line endpoints for drawing

        Returns
        -------
        frets   : list of Line objects for fret wires (roughly vertical)
        strings : list of Line objects for guitar strings (roughly horizontal)
        """
        raw_lines = self._hough(edges)
        if raw_lines is None:
            return [], []

        lines = [self._to_line(r, t, frame_shape) for r, t in raw_lines]
        clustered = self._cluster(lines)
        frets, strings = self._split_frets_strings(clustered)

        # Sort frets left-to-right, strings top-to-bottom
        frets.sort(key=lambda l: l.x1)
        strings.sort(key=lambda l: l.y1)

        return frets, strings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hough(self, edges: np.ndarray) -> np.ndarray | None:
        """Run cv2.HoughLines and return raw (rho, theta) pairs or None."""
        result = cv2.HoughLines(
            edges,
            self.rho,
            self.theta,
            self.threshold,
        )
        if result is None:
            return None
        # cv2.HoughLines returns shape (N, 1, 2) — squeeze to (N, 2)
        return result[:, 0, :]

    def _to_line(self, rho: float, theta: float, frame_shape: tuple) -> Line:
        """
        Convert a (rho, theta) Hough result into a Line with pixel endpoints.
        Standard conversion from OpenCV docs.
        """
        h, w = frame_shape[:2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x0 = cos_t * rho
        y0 = sin_t * rho
        # Extend the line far enough to always cross the full frame
        scale = max(h, w) * 2
        x1 = int(x0 + scale * (-sin_t))
        y1 = int(y0 + scale * cos_t)
        x2 = int(x0 - scale * (-sin_t))
        y2 = int(y0 - scale * cos_t)
        return Line(rho=rho, theta=theta, x1=x1, y1=y1, x2=x2, y2=y2)

    def _cluster(self, lines: list[Line]) -> list[Line]:
        """
        Merge near-duplicate lines into one representative line per cluster.

        Algorithm:
          1. Sort lines by rho.
          2. Walk through — if the current line is within (cluster_rho_px,
             cluster_theta_deg) of the current cluster's running mean, add it.
          3. Otherwise, close the current cluster and start a new one.
          4. Each cluster is averaged to produce one output Line.

        This is a greedy single-pass approach — fast and sufficient for the
        typically small number of lines returned by HoughLines on a fretboard.
        """
        if not lines:
            return []

        lines_sorted = sorted(lines, key=lambda l: l.rho)
        clusters: list[list[Line]] = [[lines_sorted[0]]]

        for line in lines_sorted[1:]:
            # Compare against the mean rho/theta of the last open cluster
            last_cluster = clusters[-1]
            mean_rho = np.mean([l.rho for l in last_cluster])
            mean_theta = np.mean([l.theta for l in last_cluster])

            rho_close = abs(line.rho - mean_rho) < self.cluster_rho_px
            theta_close = (
                abs(np.degrees(line.theta - mean_theta)) < self.cluster_theta_deg
            )

            if rho_close and theta_close:
                last_cluster.append(line)
            else:
                clusters.append([line])

        # Average each cluster into one representative Line
        merged = []
        for cluster in clusters:
            avg_rho = float(np.mean([l.rho for l in cluster]))
            avg_theta = float(np.mean([l.theta for l in cluster]))
            # Use the frame shape from the first member to recompute endpoints
            rep = cluster[0]
            merged_line = self._to_line(
                avg_rho, avg_theta,
                (max(l.y1 for l in cluster), max(l.x1 for l in cluster))
            )
            merged_line.rho = avg_rho
            merged_line.theta = avg_theta
            merged.append(merged_line)

        return merged

    def _split_frets_strings(
        self, lines: list[Line]
    ) -> tuple[list[Line], list[Line]]:
        """
        Split lines into fret wires (vertical) and strings (horizontal)
        based on their angle.

        In OpenCV Hough space:
          - theta ≈ 0 or pi → line is vertical  → fret wire
          - theta ≈ pi/2    → line is horizontal → guitar string

        We use angle_deg (the line's own angle, not the perpendicular)
        to make this more intuitive:
          - fret wires:    angle near 90° (vertical)
          - guitar strings: angle near 0° or 180° (horizontal)
        """
        frets = []
        strings = []

        for line in lines:
            angle = abs(line.angle_deg)

            # Vertical = fret wire
            if abs(angle - 90) <= self.fret_angle_tolerance:
                frets.append(line)
            # Horizontal = guitar string (angle near 0 or near 180)
            elif angle <= self.string_angle_tolerance or angle >= (
                180 - self.string_angle_tolerance
            ):
                strings.append(line)
            # Diagonal lines are ignored — not frets or strings

        return frets, strings

    # ------------------------------------------------------------------
    # Debug / visualisation helpers
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        frets: list[Line],
        strings: list[Line],
    ) -> np.ndarray:
        """
        Draw detected fret lines (green) and string lines (blue) on a copy
        of the frame. Does not modify the original.
        """
        out = frame.copy()
        for line in frets:
            cv2.line(out, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 1)
        for line in strings:
            cv2.line(out, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 1)
        return out

    def debug_view(
        self,
        frame: np.ndarray,
        edges: np.ndarray,
        frets: list[Line],
        strings: list[Line],
    ) -> np.ndarray:
        """
        Side-by-side: original | edge map | detected lines.
        Use during tuning — not in the main loop.
        """
        annotated = self.draw(frame, frets, strings)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Label counts
        cv2.putText(
            annotated,
            f"frets:{len(frets)} strings:{len(strings)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        return np.hstack([frame, edges_bgr, annotated])


# ------------------------------------------------------------------
# Standalone test — python src/localization/hough_lines.py
# Shows fret lines (green) and string lines (blue) live.
# Tune threshold up if too many lines, down if frets are missed.
# ------------------------------------------------------------------
if __name__ == "__main__":
    from edge_detector import EdgeDetector

    edge_det = EdgeDetector()
    hough_det = HoughDetector(threshold=80)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Green = fret wires | Blue = strings | Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        edges = edge_det.process(frame)
        frets, strings = hough_det.detect(edges, frame.shape)
        display = hough_det.debug_view(frame, edges, frets, strings)

        cv2.imshow("HoughDetector", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()