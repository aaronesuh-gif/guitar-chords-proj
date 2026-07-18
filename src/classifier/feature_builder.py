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


# The canonical feature order. Both build_dataset.py's CSV columns and
# this list must match exactly — see build_dataset.py's _build_row()
# for where the CSV columns are created.
FEATURE_COLUMNS = (
    [f"{name}_fret" for name in HandTracker.FINGERTIP_NAMES]
    + [f"{name}_string" for name in HandTracker.FINGERTIP_NAMES]
    + ["index_angle", "index_span", "is_barre"]
)


def build_feature_vector(
    positions: dict,
    orientation_result,
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

    Returns
    -------
    (13,) numpy array in FEATURE_COLUMNS order:
        [thumb_fret, index_fret, middle_fret, ring_fret, pinky_fret,
         thumb_string, index_string, middle_string, ring_string, pinky_string,
         index_angle, index_span, is_barre]
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

    vector = np.array(frets + strings + [angle, span, is_barre], dtype=np.float32)
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