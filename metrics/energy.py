"""
metrics/energy.py
Simulated energy consumption estimation for edge training.

Formula:  Energy (J) ≈ computation_time (s) × model_complexity_factor
"""

import time
from typing import Optional
from shared.utils import count_parameters
import torch.nn as nn


# Baseline power draw in Watts (simulated IoT edge device)
BASELINE_POWER_W = 2.5


def estimate_energy(
    computation_time_s: float,
    model: Optional[nn.Module] = None,
    power_w: float = BASELINE_POWER_W,
) -> float:
    """
    Estimate energy consumption during a training or inference phase.

    Args:
        computation_time_s : Duration of computation in seconds.
        model              : If provided, scales power by parameter count.
        power_w            : Baseline device power draw in Watts.

    Returns:
        Estimated energy in Joules.
    """
    if model is not None:
        param_count = count_parameters(model)
        # Scale: every 100k params adds ~0.5W (simulated)
        scale_factor = 1.0 + (param_count / 100_000) * 0.5
        power_w = power_w * scale_factor

    energy_j = power_w * computation_time_s
    return round(energy_j, 6)


class EnergyMonitor:
    """Context manager that tracks energy during a code block."""

    def __init__(self, model: Optional[nn.Module] = None, power_w: float = BASELINE_POWER_W):
        self.model    = model
        self.power_w  = power_w
        self.energy_j = 0.0
        self._start   = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self.energy_j = estimate_energy(elapsed, self.model, self.power_w)
