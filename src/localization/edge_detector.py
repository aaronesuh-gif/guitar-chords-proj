"""
edge_detector.py
----------------
Converts a raw BGR camera frame into a clean binary edge map
suitable for Hough line detection on guitar fret wires.
"""


import cv2
import numpy as np


class EdgeDetector:
    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        clahe_clip: float = 2.0,
        clahe_grid: tuple = (8, 8),
    ):
        """
        Parameters
        ----------
        blur_kernel  : Gaussian blur kernel size (must be odd). Larger = more noise
                       suppression but softer edges. Start at 5, raise if Hough is
                       noisy.
        canny_low    : Lower hysteresis threshold. Edges with gradient below this
                       are discarded.
        canny_high   : Upper hysteresis threshold. Edges above this are kept; edges
                       between low/high are kept only if connected to a strong edge.
                       Ratio of ~1:3 (low:high) is a good starting point.
        clahe_clip   : Contrast limit for CLAHE. Prevents over-amplification of
                       noise in uniform regions.
        clahe_grid   : Tile grid size for CLAHE. (8,8) is standard.
        """
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high

        # CLAHE equalizes contrast locally — helps with glare on metal fret wires
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing + edge detection pipeline.

        Parameters
        ----------
        frame : BGR image from cv2.VideoCapture

        Returns
        -------
        edges : single-channel uint8 binary edge map (0 or 255)
        """
        gray = self._to_gray(frame)
        equalized = self._equalize(gray)
        blurred = self._blur(equalized)
        edges = self._canny(blurred)
        return edges


    # Private steps — split out so you can inspect each stage easily


    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _equalize(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE on the grayscale image. Improves fret wire contrast under glare."""
        return self.clahe.apply(gray)

    def _blur(self, gray: np.ndarray) -> np.ndarray:
        """Gaussian blur to suppress high-frequency noise before Canny."""
        return cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

    def _canny(self, blurred: np.ndarray) -> np.ndarray:
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

  
    # Debug helper — call this during tuning, not in the main loop


    def debug_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a side-by-side BGR image: original | gray | equalized | edges.
        Useful for tuning thresholds — run this on a still photo first.

        Example:
            debug = detector.debug_view(frame)
            cv2.imshow("debug", debug)
            cv2.waitKey(0)
        """
        gray = self._to_gray(frame)
        equalized = self._equalize(gray)
        blurred = self._blur(equalized)
        edges = self._canny(blurred)

        # Convert single-channel stages to BGR for side-by-side display
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        eq_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return np.hstack([frame, gray_bgr, eq_bgr, edges_bgr])



# Quick standalone test — run: python src/localization/edge_detector.py
# Point your webcam at the guitar and press Q to quit.
# Tune canny_low / canny_high until fret wires are clean lines.

if __name__ == "__main__":
    detector = EdgeDetector(
        blur_kernel=5,
        canny_low=50,
        canny_high=150,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Showing edge detection. Press Q to quit, D for debug view.")
    debug_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug_mode = not debug_mode

        if debug_mode:
            display = detector.debug_view(frame)
        else:
            edges = detector.process(frame)
            display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cv2.imshow("EdgeDetector", display)

    cap.release()
    cv2.destroyAllWindows()