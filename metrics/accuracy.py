"""
metrics/accuracy.py
Accuracy computation utilities.
"""

import torch
from typing import Tuple


def compute_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.

    Args:
        outputs : Raw logits tensor of shape (N, num_classes)
        labels  : Ground truth tensor of shape (N,)

    Returns:
        Accuracy as a float percentage (0–100).
    """
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        correct     = (predictions == labels).sum().item()
        total       = labels.size(0)
    return (correct / total) * 100.0 if total > 0 else 0.0


def compute_batch_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[int, int]:
    """
    Return (correct_count, total_count) for accumulating over batches.

    Usage:
        correct, total = 0, 0
        for batch in loader:
            c, t = compute_batch_accuracy(model(x), y)
            correct += c; total += t
        accuracy = correct / total * 100
    """
    with torch.no_grad():
        predictions = torch.argmax(outputs, dim=1)
        correct     = (predictions == labels).sum().item()
        total       = labels.size(0)
    return correct, total
