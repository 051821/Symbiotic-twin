"""
data/partition.py
Splits the processed dataset by device ID (natural Non-IID partitioning).
Each edge receives data only from its assigned device.
Saves train/test tensors as .pt files per edge.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

from config.loader import get_config
from config.logging_config import setup_logger

logger = setup_logger("partition")

FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
LABEL_COL    = "label"


def get_edge_partition(
    device_id: str,
    processed_path: str = None,
    partitions_path: str = None,
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """
    Load (or create) train/test TensorDatasets for a specific device.

    Args:
        device_id       : MAC address of the IoT device.
        processed_path  : Path to processed.csv
        partitions_path : Directory to cache/load .pt partition files.

    Returns:
        (train_dataset, test_dataset) as TensorDatasets.
    """
    cfg = get_config()
    processed_path  = Path(processed_path  or cfg["data"]["processed_path"]) / "processed.csv"
    partitions_path = Path(partitions_path or cfg["data"]["partitions_path"])
    test_split      = cfg["data"]["test_split"]
    partitions_path.mkdir(parents=True, exist_ok=True)

    safe_id   = device_id.replace(":", "_")
    train_pt  = partitions_path / f"{safe_id}_train.pt"
    test_pt   = partitions_path / f"{safe_id}_test.pt"

    # Return cached tensors if available
    if train_pt.exists() and test_pt.exists():
        logger.info(f"[{device_id}] Loading cached partition from {partitions_path}")
        train_data = torch.load(train_pt, weights_only=False)
        test_data  = torch.load(test_pt, weights_only=False)
        return train_data, test_data

    logger.info(f"[{device_id}] Building partition from {processed_path}")
    df = pd.read_csv(processed_path)

    device_df = df[df["device"] == device_id].reset_index(drop=True)
    if device_df.empty:
        raise ValueError(f"No data found for device: {device_id}")

    logger.info(f"[{device_id}] Samples: {len(device_df)}")

    X = device_df[FEATURE_COLS].values.astype(np.float32)
    y = device_df[LABEL_COL].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, shuffle=False
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test), torch.tensor(y_test)
    )

    torch.save(train_dataset, train_pt)
    torch.save(test_dataset,  test_pt)
    logger.info(f"[{device_id}] Partition saved → train={len(X_train)}, test={len(X_test)}")

    return train_dataset, test_dataset


def partition_all() -> Dict[str, int]:
    """Partition data for all configured edges. Returns sample counts per edge."""
    cfg = get_config()
    sample_counts = {}
    for edge in cfg["edges"]:
        device_id = edge["device"]
        train_ds, _ = get_edge_partition(device_id)
        sample_counts[edge["id"]] = len(train_ds)
    return sample_counts


if __name__ == "__main__":
    counts = partition_all()
    for edge_id, count in counts.items():
        print(f"{edge_id}: {count} training samples")
