"""
shared/utils.py
Common utility functions used across edge and server modules.
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return GPU device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(Path(path), "r") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_weights(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    """Flatten all model weights into a single 1D numpy array."""
    return np.concatenate([v.cpu().numpy().flatten() for v in state_dict.values()])
