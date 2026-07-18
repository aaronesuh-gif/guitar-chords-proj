"""
capture_images.py
------------------
Standalone tool for collecting your chord training dataset. Opens the
webcam, lets you switch between chord labels, and saves frames to
data/<chord_name>/ on keypress.

This is a data collection tool, not part of the inference pipeline —
it doesn't use the localization or vision modules at all. Its only
job is getting clean labeled photos onto disk.

Usage:
    python capture_images.py

Controls:
    SPACE    - save current frame to the active chord's folder
    N        - cycle to the next chord in CHORDS list
    P        - cycle to the previous chord
    Q        - quit
"""

import os
import time

import cv2


# Edit this list to match the chords you're training on.
# Folder names come directly from these strings, so keep them simple —
# no spaces or special characters.
CHORDS = ["A", "Am", "C", "D", "E", "Em", "G", "F", "Bm"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class ImageCapture:
    def __init__(self, chords: list[str] = CHORDS, data_dir: str = DATA_DIR):
        self.chords = chords
        self.data_dir = data_dir
        self.current_idx = 0

        self._ensure_folders_exist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_chord(self) -> str:
        return self.chords[self.current_idx]

    def next_chord(self):
        self.current_idx = (self.current_idx + 1) % len(self.chords)

    def previous_chord(self):
        self.current_idx = (self.current_idx - 1) % len(self.chords)

    def save_frame(self, frame) -> str:
        """
        Save a frame to the current chord's folder with a timestamped
        filename so repeated captures never overwrite each other.

        Returns
        -------
        The full path the image was saved to.
        """
        folder = os.path.join(self.data_dir, self.current_chord)
        timestamp = int(time.time() * 1000)  # milliseconds — avoids collisions
        filename = f"{self.current_chord.lower()}_{timestamp}.jpg"
        path = os.path.join(folder, filename)
        cv2.imwrite(path, frame)
        return path

    def count_images(self, chord: str) -> int:
        """Count how many images currently exist for a given chord."""
        folder = os.path.join(self.data_dir, chord)
        if not os.path.exists(folder):
            return 0
        return len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    def all_counts(self) -> dict[str, int]:
        """Return image counts for every chord — useful for a progress overview."""
        return {chord: self.count_images(chord) for chord in self.chords}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_folders_exist(self):
        """Create data/<chord>/ for every chord if it doesn't exist yet."""
        for chord in self.chords:
            folder = os.path.join(self.data_dir, chord)
            os.makedirs(folder, exist_ok=True)


# ------------------------------------------------------------------
# Standalone tool — python capture_images.py
# ------------------------------------------------------------------
if __name__ == "__main__":
    capture = ImageCapture()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("=" * 50)
    print("Chord image capture tool")
    print("=" * 50)
    print("SPACE - save frame   N - next chord   P - previous chord   Q - quit")
    print()

    # Flash effect state — briefly whiten the frame after a save so you
    # get visual confirmation without needing to watch the terminal.
    flash_until = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Header bar with current chord + running count
        count = capture.count_images(capture.current_chord)
        header = f"Chord: {capture.current_chord}   Saved: {count}"
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (30, 30, 30), -1)
        cv2.putText(display, header, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Brief white flash to confirm a save happened
        if time.time() < flash_until:
            display = cv2.addWeighted(
                display, 0.5, 255 * (display * 0 + 1), 0.5, 0
            )

        cv2.imshow("Chord Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            path = capture.save_frame(frame)
            print(f"Saved: {path}")
            flash_until = time.time() + 0.15
        elif key == ord("n"):
            capture.next_chord()
            print(f"Switched to: {capture.current_chord}")
        elif key == ord("p"):
            capture.previous_chord()
            print(f"Switched to: {capture.current_chord}")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 50)
    print("Final counts:")
    for chord, n in capture.all_counts().items():
        print(f"  {chord}: {n} images")
    print("=" * 50)