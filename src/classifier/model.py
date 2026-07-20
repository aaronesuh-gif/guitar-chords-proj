"""
model.py
--------
The classifier itself. A small multilayer perceptron (MLP) that takes
the feature vector built from build_dataset.py's CSV columns (fret,
string per finger, plus orientation features) and predicts a chord
label.

This is intentionally simple — an MLP rather than a CNN or anything
heavier — because the input is already a compact, geometrically
meaningful feature vector (10 fret/string values + 3 orientation
values = 13 numbers), not raw pixels. A small network is plenty.

Usage:
    from src.classifier.model import ChordMLP
    model = ChordMLP(input_dim=13, num_classes=6)
    logits = model(feature_tensor)
"""

import torch
import torch.nn as nn


class ChordMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ):
        """
        Parameters
        ----------
        input_dim    : size of the input feature vector. For 5 fingers
                        x (fret, string) = 10, plus 3 orientation
                        features (angle, span, is_barre) = 13 total.
                        Must match feature_builder.py's output exactly.
        num_classes  : number of chord labels the model predicts.
        hidden_dims  : sizes of the hidden layers. Two layers of
                        (64, 32) is intentionally small — this dataset
                        is small (hundreds, not thousands, of rows) and
                        the features are already low-dimensional and
                        structured, so a large network would just overfit.
        dropout      : dropout probability between hidden layers, to
                        reduce overfitting on a small dataset.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final layer outputs raw logits — softmax is applied by the
        # loss function (CrossEntropyLoss) during training, and
        # explicitly at inference time when we want probabilities.
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch_size, input_dim) tensor of feature vectors

        Returns
        -------
        (batch_size, num_classes) tensor of raw logits
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference — returns softmax probabilities
        instead of raw logits. Puts the model in eval mode and disables
        gradient tracking.

        Parameters
        ----------
        x : (batch_size, input_dim) tensor, or (input_dim,) for a
            single sample (will be unsqueezed automatically)

        Returns
        -------
        (batch_size, num_classes) tensor of class probabilities
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)