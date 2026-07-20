"""
feature_builder.py
-------------------
Single source of truth for turning (fret, string) positions + an
orientation result into the exact numeric feature vector the model
expects — in the exact same order every time.

This file exists specifically to prevent a very common and very
annoying bug: training code and inference code independently building
"the same" feature vector but in a slightly different column order.
That mismatch silently produces garbage predictions with no error
message. By importing FEATURE_COLUMNS and build_feature_vector() in
both train.py and main.py, that whole category of bug is impossible.

Usage:
    from src.classifier.feature_builder import FEATURE_COLUMNS, build_feature_vector
    vector = build_feature_vector(positions, orientation_result)
"""

import numpy as np

from src.vision.hand_tracker import HandTracker


# Landmark shape features: all 21 landmarks' (x, y), normalized relative
# to the wrist and scaled by hand size. These capture the raw hand SHAPE
# independent of the fretboard homography — critical because D, A, and Am
# have distinctly different finger arrangements even when fret/string
# mapping is noisy or the fretboard isn't detected well.
LANDMARK_FEATURE_COLUMNS = [
    f"lm{i}_{axis}" for i in range(21) for axis in ("x", "y")
]

# Pairwise fingertip distances (normalized by hand scale) — the most
# directly discriminative shape signal for similar chords: D's tight
# triangle, A's three-in-a-row, and Am's offset cluster produce clearly
# different distance patterns between fingertips.
_TIP_PAIRS = [
    (a, b)
    for i, a in enumerate(HandTracker.FINGERTIP_NAMES)
    for b in HandTracker.FINGERTIP_NAMES[i + 1:]
]
PAIRWISE_DIST_COLUMNS = [f"dist_{a}_{b}" for a, b in _TIP_PAIRS]

# The canonical feature order. Both build_dataset.py's CSV columns and
# this list must match exactly — see build_dataset.py's _build_row()
# for where the CSV columns are created.
FEATURE_COLUMNS = (
    [f"{name}_fret" for name in HandTracker.FINGERTIP_NAMES]
    + [f"{name}_string" for name in HandTracker.FINGERTIP_NAMES]
    + ["index_angle", "index_span", "is_barre"]
    + LANDMARK_FEATURE_COLUMNS
    + PAIRWISE_DIST_COLUMNS
)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Make landmark coordinates translation- and scale-invariant:

      1. Subtract the wrist position (landmark 0) — so the features
         describe hand SHAPE, not where the hand is in the frame.
      2. Divide by hand size (wrist -> middle-finger MCP distance) —
         so the features don't change when the hand is closer to or
         further from the camera.

    Parameters
    ----------
    landmarks : (21, 3) array from HandTracker.process()

    Returns
    -------
    (21, 2) array of normalized (x, y) coordinates
    """
    xy = landmarks[:, :2].astype(np.float64)
    wrist = xy[HandTracker.WRIST]
    centered = xy - wrist

    hand_scale = np.linalg.norm(xy[HandTracker.MIDDLE_MCP] - wrist)
    if hand_scale < 1e-6:
        hand_scale = 1.0  # degenerate frame — avoid division by zero

    return centered / hand_scale


def compute_pairwise_distances(landmarks: np.ndarray) -> list[float]:
    """
    Distances between every pair of fingertips, computed on the
    wrist-normalized coordinates so they're scale-invariant.

    Returns
    -------
    list of 10 floats, in PAIRWISE_DIST_COLUMNS order
    """
    normalized = normalize_landmarks(landmarks)
    tip_positions = {
        name: normalized[idx]
        for name, idx in zip(HandTracker.FINGERTIP_NAMES, HandTracker.FINGERTIPS)
    }
    return [
        float(np.linalg.norm(tip_positions[a] - tip_positions[b]))
        for a, b in _TIP_PAIRS
    ]


def build_landmark_feature_dict(landmarks: np.ndarray) -> dict:
    """
    Landmark shape features as a {column_name: value} dict — used by
    build_dataset.py when writing CSV rows. Includes both the raw
    normalized coordinates and the pairwise fingertip distances.
    """
    normalized = normalize_landmarks(landmarks)
    features = {}
    for i in range(21):
        features[f"lm{i}_x"] = round(float(normalized[i][0]), 4)
        features[f"lm{i}_y"] = round(float(normalized[i][1]), 4)

    distances = compute_pairwise_distances(landmarks)
    for col, dist in zip(PAIRWISE_DIST_COLUMNS, distances):
        features[col] = round(dist, 4)

    return features


def build_feature_vector(
    positions: dict,
    orientation_result,
    landmarks: np.ndarray,
) -> np.ndarray:
    """
    Build a feature vector from live pipeline output — used at
    inference time in main.py.

    Parameters
    ----------
    positions : dict from CoordinateMapper.map_landmarks(), e.g.
                {"thumb": (2,3), "index": None, ...}
    orientation_result : OrientationResult from OrientationAnalyzer,
                          or None if unavailable this frame
    landmarks : (21, 3) array from HandTracker.process() — required for
                the shape features

    Returns
    -------
    (55,) numpy array in FEATURE_COLUMNS order:
        10 fret/string values + 3 orientation values + 42 shape values
    """
    frets = []
    strings = []
    for name in HandTracker.FINGERTIP_NAMES:
        pos = positions.get(name)
        fret, string = pos if pos is not None else (0, 0)
        frets.append(fret)
        strings.append(string)

    if orientation_result is not None:
        angle = orientation_result.angle_deg
        span = orientation_result.span_estimate
        is_barre = int(orientation_result.is_barre)
    else:
        angle, span, is_barre = 90.0, 0.0, 0

    normalized = normalize_landmarks(landmarks)
    landmark_values = normalized.flatten().tolist()  # 21 x (x,y) = 42 values
    distance_values = compute_pairwise_distances(landmarks)  # 10 values

    vector = np.array(
        frets + strings + [angle, span, is_barre] + landmark_values + distance_values,
        dtype=np.float32,
    )
    assert len(vector) == len(FEATURE_COLUMNS), (
        f"Feature vector length {len(vector)} doesn't match "
        f"FEATURE_COLUMNS length {len(FEATURE_COLUMNS)} — check for a "
        f"mismatch between build_feature_vector() and FEATURE_COLUMNS."
    )
    return vector


def build_feature_matrix_from_dataframe(df) -> np.ndarray:
    """
    Build the feature matrix from a pandas DataFrame — used at
    training time in train.py, where features come from dataset.csv
    rather than a live camera frame.

    Parameters
    ----------
    df : pandas DataFrame loaded from dataset.csv, must contain all
         columns listed in FEATURE_COLUMNS

    Returns
    -------
    (n_rows, 13) numpy array, columns in FEATURE_COLUMNS order
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"dataset.csv is missing expected columns: {missing}. "
            f"Did build_dataset.py run to completion? "
            f"Expected columns: {FEATURE_COLUMNS}"
        )
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)