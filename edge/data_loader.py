"""
edge/data_loader.py
Wraps data/partition.py for use inside edge containers.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple

from config.loader import get_config
from data.partition import get_edge_partition


def load_edge_data(
    device_id: str,
    batch_size: int = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return DataLoaders for the given device's train and test partitions.

    Args:
        device_id  : MAC address of the IoT device assigned to this edge.
        batch_size : Override config batch size if provided.

    Returns:
        (train_loader, test_loader)
    """
    cfg        = get_config()
    batch_size = batch_size or cfg["system"]["batch_size"]

    train_ds, test_ds = get_edge_partition(device_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader


def get_sample_count(device_id: str) -> int:
    """Return number of training samples for the given device."""
    train_ds, _ = get_edge_partition(device_id)
    return len(train_ds)
