"""
overlay.py
----------
The full visual overlay for live chord detection. Replaces main.py's
bare-bones dots-and-text with:

  - Full hand skeleton (all 21 landmarks + connections)
  - Fretboard grid locked to the detected homography region
  - Per-finger (fret, string) coordinate labels at each fingertip
  - Chord name + confidence, color-coded
  - A stats panel: index finger angle/span/barre flag, FPS, fret/string
    detection counts

Usage:
    from src.ui.overlay import OverlayRenderer
    renderer = OverlayRenderer()
    display = renderer.render(frame, pipeline_result)
"""

import time

import cv2
import numpy as np

from src.vision.hand_tracker import HandTracker


class OverlayRenderer:
    def __init__(self):
        # For FPS calculation — tracks time between render() calls
        self._last_frame_time = time.time()
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, frame: np.ndarray, pipeline_result: dict) -> np.ndarray:
        """
        Draw the full overlay on a copy of the frame.

        Parameters
        ----------
        frame : raw BGR camera frame
        pipeline_result : dict produced by ChordDetector.process_frame(),
                           expected keys: chord, confidence, raw_chord,
                           frets, strings, landmarks, positions,
                           orientation_result, homography (HomographyComputer)

        Returns
        -------
        Annotated BGR frame, ready for cv2.imshow()
        """
        display = frame.copy()
        self._update_fps()

        homography = pipeline_result.get("homography")
        landmarks = pipeline_result.get("landmarks")
        positions = pipeline_result.get("positions")
        frets = pipeline_result.get("frets", [])
        strings = pipeline_result.get("strings", [])

        # Layer 1 — subtle fretboard boundary (thin outline only, no grid)
        if homography is not None and getattr(homography, "last_corners", None) is not None:
            display = self._draw_fretboard_outline(display, homography.last_corners)

        # Layer 2 — full hand skeleton
        if landmarks is not None:
            display = self._draw_skeleton(display, landmarks)
            display = self._draw_fingertip_labels(display, landmarks, positions)

        # Layer 3 — chord name + confidence banner
        display = self._draw_chord_banner(display, pipeline_result)

        # Layer 4 — stats panel (bottom-left)
        display = self._draw_stats_panel(display, pipeline_result, frets, strings)

        return display

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_fretboard_outline(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Thin, dim outline of the detected fretboard region — enough to
        confirm tracking is working, deliberately NOT a full grid.
        Corners order from homography: [tl, tr, bl, br].
        """
        out = frame.copy()
        pts = corners.astype(int)[[0, 1, 3, 2]]  # reorder to draw a closed quad
        cv2.polylines(out, [pts], isClosed=True, color=(120, 200, 120), thickness=2, lineType=cv2.LINE_AA)
        return out

    def _draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Full 21-point hand skeleton with connections, brighter than the grid."""
        out = frame.copy()
        for a, b in HandTracker.CONNECTIONS:
            pt_a = self._to_pt(landmarks[a][:2])
            pt_b = self._to_pt(landmarks[b][:2])
            cv2.line(out, pt_a, pt_b, (0, 220, 0), 2)

        for i, (x, y, _) in enumerate(landmarks):
            is_tip = i in HandTracker.FINGERTIPS
            radius = 6 if is_tip else 3
            color = (0, 255, 255) if is_tip else (0, 255, 0)
            cv2.circle(out, self._to_pt((x, y)), radius, color, -1)

        return out

    def _draw_fingertip_labels(
        self, frame: np.ndarray, landmarks: np.ndarray, positions: dict | None
    ) -> np.ndarray:
        """Draw (fret, string) text next to each fingertip."""
        if positions is None:
            return frame

        out = frame.copy()
        for name, idx in zip(HandTracker.FINGERTIP_NAMES, HandTracker.FINGERTIPS):
            x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
            pos = positions.get(name)
            label = f"F{pos[0]:.1f}S{pos[1]:.1f}" if pos is not None else "--"
            cv2.putText(
                out, label, (x + 8, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
            )
        return out

    def _draw_chord_banner(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Top banner: predicted chord name + confidence, color-coded."""
        out = frame.copy()
        chord = result.get("chord")
        confidence = result.get("confidence", 0.0)

        if chord is None:
            text = "No hand / fretboard detected"
            color = (150, 150, 150)
        else:
            text = f"{chord}   {confidence*100:.0f}%"
            if confidence >= 0.85:
                color = (0, 200, 0)
            elif confidence >= 0.6:
                color = (0, 200, 255)
            else:
                color = (0, 0, 255)

        cv2.rectangle(out, (0, 0), (out.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(out, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Thin confidence meter bar under the banner
        bar_width = int(out.shape[1] * confidence) if chord else 0
        cv2.rectangle(out, (0, 50), (bar_width, 54), color, -1)

        return out

    def _draw_stats_panel(
        self, frame: np.ndarray, result: dict, frets: list, strings: list
    ) -> np.ndarray:
        """Bottom-left stats: raw vs smoothed chord, orientation data, FPS."""
        out = frame.copy()
        h = out.shape[0]

        orientation_result = result.get("orientation_result")
        raw_chord = result.get("raw_chord")

        lines = [
            f"FPS: {self._fps:.1f}",
            f"frets:{len(frets)} strings:{len(strings)}",
            f"raw: {raw_chord if raw_chord else '--'}",
        ]

        if orientation_result is not None:
            lines.append(
                f"idx angle:{orientation_result.angle_deg:.0f} "
                f"span:{orientation_result.span_estimate:.1f} "
                f"barre:{orientation_result.is_barre}"
            )

        panel_height = 20 * len(lines) + 10
        y0 = h - panel_height
        cv2.rectangle(out, (0, y0), (260, h), (30, 30, 30), -1)

        for i, line in enumerate(lines):
            y = y0 + 20 * (i + 1)
            cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return out

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _to_pt(self, xy) -> tuple[int, int]:
        return (int(xy[0]), int(xy[1]))

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            # Light smoothing so the FPS number doesn't jitter wildly
            instant_fps = 1.0 / dt
            self._fps = self._fps * 0.9 + instant_fps * 0.1 if self._fps > 0 else instant_fps