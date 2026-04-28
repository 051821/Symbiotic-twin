"""
edge/data_loader.py
Wraps data/partition.py for use inside edge containers.
Passes round_num so the temporal sliding window advances each round.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple

from config.loader import get_config
from data.partition import get_edge_partition


def load_edge_data(
    device_id: str,
    round_num: int = 0,
    batch_size: int = None,
    max_train_samples: int = None,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, int]:
    """
    Return DataLoaders for the given device's current-round window.

    Args:
        device_id  : MAC address of the IoT device.
        round_num  : Current federated round (0-indexed). Controls window position.
        batch_size : Override config batch size if provided.

    Returns:
        (train_loader, test_loader, class_weights, sample_count)
    """
    cfg        = get_config()
    batch_size = batch_size or cfg["system"]["batch_size"]
    num_workers = int(cfg["system"].get("dataloader_workers", 0) or 0)
    pin_memory = torch.cuda.is_available()

    train_ds, test_ds, class_weights = get_edge_partition(
        device_id,
        round_num=round_num,
        max_train_samples_override=max_train_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
    )

    return train_loader, test_loader, class_weights, len(train_ds)


def get_sample_count(device_id: str, round_num: int = 0) -> int:
    """Return number of training samples for the given device at given round."""
    train_ds, _, _ = get_edge_partition(device_id, round_num=round_num)
    return len(train_ds)
