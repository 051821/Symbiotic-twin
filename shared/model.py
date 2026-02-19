"""
shared/model.py
Neural network architecture shared by edge nodes and the server.

Input  : 7 IoT features (co, humidity, light, lpg, motion, smoke, temp)
Output : 3 classes     (0=Normal, 1=Warning, 2=Critical)
"""

import torch
import torch.nn as nn
from config.loader import get_config


class IoTClassifier(nn.Module):
    """Feedforward neural network for IoT environmental classification."""

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_classes: int = None,
        dropout: float = None,
    ):
        super(IoTClassifier, self).__init__()

        cfg = get_config().get("model", {})
        input_size  = input_size  or cfg.get("input_size", 7)
        hidden_size = hidden_size or cfg.get("hidden_size", 64)
        num_classes = num_classes or cfg.get("num_classes", 3)
        dropout     = dropout     if dropout is not None else cfg.get("dropout", 0.3)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model() -> IoTClassifier:
    """Convenience factory that builds a model from config."""
    return IoTClassifier()
