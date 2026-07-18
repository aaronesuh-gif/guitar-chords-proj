"""
train.py
--------
Loads data/dataset.csv (built by build_dataset.py), trains the MLP
classifier, and saves three artifacts to models/:

  chord_mlp.pt       - trained PyTorch model weights
  chord_labels.json  - index -> chord name mapping
  scaler.joblib       - fitted StandardScaler for feature normalization

All three are required at inference time (main.py loads all of them).

Usage:
    python -m train.train
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.classifier.model import ChordMLP
from src.classifier.feature_builder import FEATURE_COLUMNS, build_feature_matrix_from_dataframe


DATA_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_PATH = os.path.join(MODELS_DIR, "chord_mlp.pt")
LABELS_PATH = os.path.join(MODELS_DIR, "chord_labels.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

# Training hyperparameters — reasonable defaults for a small dataset
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
TEST_SIZE = 0.15    # held out entirely, never seen until final evaluation
VAL_SIZE = 0.15      # held out from remaining train set, used during training


def load_dataset(csv_path: str = DATA_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m train.build_dataset` first."
        )
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"{csv_path} is empty — no usable images were processed.")
    return df


def prepare_data(df: pd.DataFrame):
    """
    Split into train/val/test, encode labels, and standardize features.

    Returns
    -------
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test,
              label_encoder, scaler)
    """
    X = build_feature_matrix_from_dataframe(df)
    y_raw = df["label"].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # First split off the test set — never touched until final eval
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    # Then split remaining into train/val
    val_fraction_of_temp = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=42, stratify=y_temp
    )

    # Fit scaler on TRAIN ONLY — fitting on val/test would leak
    # information from data the model shouldn't have "seen" yet
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler


def train_model(
    X_train, y_train, X_val, y_val,
    num_classes: int,
    epochs: int = EPOCHS,
) -> ChordMLP:
    """
    Standard PyTorch training loop with CrossEntropyLoss + Adam.
    Tracks the best validation accuracy and keeps those weights,
    since the final epoch isn't always the best one on a small dataset.
    """
    model = ChordMLP(input_dim=X_train.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_val_acc = 0.0
    best_state = None

    n_samples = len(X_train_t)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(n_samples)
        epoch_loss = 0.0

        for i in range(0, n_samples, BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(idx)

        epoch_loss /= n_samples

        # Validation pass
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  loss={epoch_loss:.4f}  val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    return model


def evaluate_test_set(model: ChordMLP, X_test, y_test) -> float:
    """Final, held-out evaluation — the number that goes in your resume bullet."""
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():
        logits = model(X_test_t)
        preds = logits.argmax(dim=1)
        acc = (preds == y_test_t).float().mean().item()

    print(f"Test set accuracy: {acc:.3f}")
    return acc


def save_artifacts(model: ChordMLP, label_encoder: LabelEncoder, scaler: StandardScaler):
    os.makedirs(MODELS_DIR, exist_ok=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Save input_dim/num_classes alongside labels so main.py can
    # reconstruct the exact same architecture before loading weights
    label_data = {
        "classes": label_encoder.classes_.tolist(),
        "input_dim": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
    }
    with open(LABELS_PATH, "w") as f:
        json.dump(label_data, f, indent=2)
    print(f"Saved labels to {LABELS_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")


def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Loaded {len(df)} rows across {df['label'].nunique()} chords")
    print(df["label"].value_counts().to_string())

    print("\nPreparing train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, scaler = prepare_data(df)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    print("\nTraining...")
    model = train_model(
        X_train, y_train, X_val, y_val,
        num_classes=len(label_encoder.classes_),
    )

    print("\nEvaluating on held-out test set...")
    evaluate_test_set(model, X_test, y_test)

    print("\nSaving artifacts...")
    save_artifacts(model, label_encoder, scaler)

    print("\nDone. Run main.py to test live chord detection.")


if __name__ == "__main__":
    main()