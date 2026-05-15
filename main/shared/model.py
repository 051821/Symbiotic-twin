"""
shared/model.py
Neural network architecture shared by edge nodes and the server.

Input  : 7 IoT features (co, humidity, light, lpg, motion, smoke, temp)
Output : 3 classes     (0=Normal, 1=Warning, 2=Critical)
"""

import torch
import torch.nn as nn
import joblib
from pathlib import Path
from config.loader import get_config


FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]


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
        self.rule_prior_enabled = bool(cfg.get("rule_prior_enabled", False))
        self.rule_prior_strength = float(cfg.get("rule_prior_strength", 0.0) or 0.0)
        self.rule_prior_margins = cfg.get("rule_prior_margins", [0.0, 0.0, 0.0, 0.0])

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, num_classes),
        )
        thresholds = self._load_scaled_rule_thresholds()
        self.register_buffer("rule_thresholds", thresholds)
        margins = self._load_rule_margins()
        self.register_buffer("rule_margins", margins)

    def _load_rule_margins(self) -> torch.Tensor:
        margins = list(self.rule_prior_margins or [0.0, 0.0, 0.0, 0.0])
        margins = (margins + [0.0, 0.0, 0.0, 0.0])[:4]
        return torch.tensor(margins, dtype=torch.float32)

    def _load_scaled_rule_thresholds(self) -> torch.Tensor:
        """
        Return label thresholds in the model's feature space.
        Current processed data is raw-valued; scaled mode is kept for datasets
        regenerated with StandardScaler output.
        """
        if not self.rule_prior_enabled:
            return torch.full((4,), float("nan"), dtype=torch.float32)

        cfg = get_config()
        if not bool(cfg.get("model", {}).get("rule_prior_scaled_input", False)):
            return torch.tensor([0.005, 0.10, 0.007, 90.0], dtype=torch.float32)

        scaler_path = Path(cfg["data"]["processed_path"]) / "scaler.pkl"
        if not scaler_path.exists():
            self.rule_prior_enabled = False
            return torch.full((4,), float("nan"), dtype=torch.float32)

        scaler = joblib.load(scaler_path)
        means = dict(zip(FEATURE_COLS, scaler.mean_))
        scales = dict(zip(FEATURE_COLS, scaler.scale_))

        def scaled(feature: str, raw_value: float) -> float:
            return (raw_value - means[feature]) / scales[feature]

        return torch.tensor(
            [
                scaled("co", 0.005),
                scaled("smoke", 0.10),
                scaled("lpg", 0.007),
                scaled("temp", 90.0),
            ],
            dtype=torch.float32,
        )

    def _rule_prior_logits(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self.rule_prior_enabled
            or self.rule_prior_strength <= 0.0
            or torch.isnan(self.rule_thresholds).any()
        ):
            return torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)

        co_t, smoke_t, lpg_t, temp_t = self.rule_thresholds.to(x.device, x.dtype)
        co_m, smoke_m, lpg_m, temp_m = self.rule_margins.to(x.device, x.dtype)
        co = x[:, 0]
        lpg = x[:, 3]
        smoke = x[:, 5]
        temp = x[:, 6]

        critical = (smoke > smoke_t + smoke_m) | (co > co_t + co_m)
        warning = ((temp > temp_t + temp_m) | (lpg > lpg_t + lpg_m)) & ~critical
        near_boundary = (
            ((smoke - smoke_t).abs() <= smoke_m)
            | ((co - co_t).abs() <= co_m)
            | ((lpg - lpg_t).abs() <= lpg_m)
            | ((temp - temp_t).abs() <= temp_m)
        )
        normal = ~(critical | warning | near_boundary)

        prior = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
        prior[:, 0] = normal.to(x.dtype) * self.rule_prior_strength
        prior[:, 1] = warning.to(x.dtype) * self.rule_prior_strength
        prior[:, 2] = critical.to(x.dtype) * self.rule_prior_strength
        return prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) + self._rule_prior_logits(x)


def build_model() -> IoTClassifier:
    """Convenience factory that builds a model from config."""
    return IoTClassifier()
