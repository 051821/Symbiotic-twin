"""
shared/serialization.py
Converts PyTorch model state_dicts to/from JSON-serializable dicts
for safe HTTP communication between edges and server.
"""

import torch
import numpy as np
from typing import Dict, Any


def serialize_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Convert a PyTorch state_dict to a JSON-serializable dict.

    Args:
        state_dict: Model state dictionary from model.state_dict()

    Returns:
        Dict with tensor values converted to nested Python lists.
    """
    serialized = {}
    for key, tensor in state_dict.items():
        serialized[key] = tensor.cpu().detach().numpy().tolist()
    return serialized


def deserialize_weights(serialized: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Convert a serialized weight dict back to a PyTorch state_dict.

    Args:
        serialized: Dict with lists as values (from serialize_weights)

    Returns:
        Dict with torch.Tensor values.
    """
    state_dict = {}
    for key, value in serialized.items():
        state_dict[key] = torch.tensor(np.array(value, dtype=np.float32))
    return state_dict


def weights_to_numpy(state_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert state_dict tensors to numpy arrays (for aggregation)."""
    return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}


def numpy_to_weights(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert numpy arrays back to torch tensors."""
    return {k: torch.from_numpy(v.astype(np.float32)) for k, v in numpy_dict.items()}
