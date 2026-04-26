"""
metrics/latency.py
Inference latency measurement utilities.
"""

import time
import torch
from typing import Callable, Any, Tuple


def measure_latency(fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Measure wall-clock latency of any callable.

    Args:
        fn    : Function to time.
        *args : Positional arguments passed to fn.

    Returns:
        (result, latency_ms) tuple.
    """
    start  = time.perf_counter()
    result = fn(*args, **kwargs)
    end    = time.perf_counter()
    latency_ms = (end - start) * 1000.0
    return result, latency_ms


def measure_inference_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device = None,
) -> Tuple[torch.Tensor, float]:
    """
    Measure model inference latency on a single batch.

    Args:
        model        : Trained PyTorch model.
        input_tensor : Input tensor of shape (N, features).
        device       : Torch device (defaults to CPU).

    Returns:
        (output_logits, latency_ms) tuple.
    """
    device = device or torch.device("cpu")
    model  = model.to(device).eval()
    x      = input_tensor.to(device)

    with torch.no_grad():
        start   = time.perf_counter()
        outputs = model(x)
        end     = time.perf_counter()

    latency_ms = (end - start) * 1000.0
    return outputs, latency_ms


class LatencyTimer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
