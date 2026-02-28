"""
data/partition.py
Non-IID partitioning by device ID with Temporal Sliding Window support.

Strategy: Each federated round trains on the NEXT time window of that
device's data — simulating real IoT streaming where new data continuously
arrives at the edge and the model must adapt to it.

Window modes (set in config.yaml):
  sliding   : fixed-size window slides forward each round
              Round 1: [0%  – 30%], Round 2: [10% – 40%], ...
  expanding : window grows from start each round (cumulative)
              Round 1: [0%  – 30%], Round 2: [0%  – 40%], ...
  full      : use all data every round (original behaviour)
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from config.loader import get_config
from config.logging_config import setup_logger

logger = setup_logger("partition")

FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
LABEL_COL    = "label"


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute per-class weights to handle class imbalance.
    Normal readings dominate the dataset — this prevents the model
    from ignoring Warning / Critical classes.
    """
    classes, counts = np.unique(y, return_counts=True)
    total   = len(y)
    weights = np.ones(3, dtype=np.float32)
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = total / (len(classes) * cnt)
    return torch.tensor(weights, dtype=torch.float32)


def get_edge_partition(
    device_id: str,
    round_num: int = 0,
    processed_path: str = None,
    partitions_path: str = None,
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset, torch.Tensor]:
    """
    Load train/test TensorDatasets for a specific device and round.

    The TEST set is always the fixed last 20% of the device's data (held-out).
    The TRAIN set shifts each round according to the window strategy.

    Args:
        device_id       : MAC address of the IoT device.
        round_num       : Current federated round (0-indexed). Used for windowing.
        processed_path  : Path to processed.csv
        partitions_path : Directory to cache the fixed test partition.

    Returns:
        (train_dataset, test_dataset, class_weights_tensor)
    """
    cfg             = get_config()
    processed_path  = Path(processed_path  or cfg["data"]["processed_path"]) / "processed.csv"
    partitions_path = Path(partitions_path or cfg["data"]["partitions_path"])
    test_split      = cfg["data"]["test_split"]
    strategy        = cfg["data"].get("window_strategy", "sliding")
    win_frac        = cfg["data"].get("window_fraction", 0.3)
    win_step        = cfg["data"].get("window_step", 0.1)

    partitions_path.mkdir(parents=True, exist_ok=True)

    # ── Load device data ──────────────────────────────────────────────────
    logger.info(f"[{device_id}] Loading partition from {processed_path} | round={round_num} | strategy={strategy}")
    df = pd.read_csv(processed_path)

    device_df = df[df["device"] == device_id].reset_index(drop=True)
    if device_df.empty:
        raise ValueError(f"No data found for device: {device_id}")

    # Data is already sorted by timestamp from preprocess.py
    X_all = device_df[FEATURE_COLS].values.astype(np.float32)
    y_all = device_df[LABEL_COL].values.astype(np.int64)
    n     = len(X_all)

    # ── Fixed test set (last test_split of all data) ───────────────────────
    test_start = int(n * (1 - test_split))
    X_test     = X_all[test_start:]
    y_test     = y_all[test_start:]

    # ── Training window selection ─────────────────────────────────────────
    train_end_idx = test_start   # training data is everything before test set

    if strategy == "full":
        # Original behaviour — all training data every round
        X_train = X_all[:train_end_idx]
        y_train = y_all[:train_end_idx]

    elif strategy == "expanding":
        # Cumulative: start small, grow each round
        initial_frac = win_frac
        step_frac    = win_step
        end_frac     = min(1.0, initial_frac + round_num * step_frac)
        end_idx      = int(train_end_idx * end_frac)
        end_idx      = max(end_idx, cfg["system"]["batch_size"] * 2)   # safety min
        X_train = X_all[:end_idx]
        y_train = y_all[:end_idx]
        logger.info(f"[{device_id}] Expanding window → [0, {end_idx}] ({end_frac*100:.1f}% of train data)")

    else:
        # SLIDING (default) — fixed window moves forward each round
        start_frac = round_num * win_step
        end_frac   = start_frac + win_frac

        # Wrap around when window reaches end (data keeps arriving)
        if end_frac > 1.0:
            start_frac = start_frac % 1.0
            end_frac   = start_frac + win_frac
            if end_frac > 1.0:
                end_frac = 1.0

        start_idx = int(train_end_idx * start_frac)
        end_idx   = int(train_end_idx * end_frac)
        end_idx   = max(end_idx, start_idx + cfg["system"]["batch_size"] * 2)

        X_train = X_all[start_idx:end_idx]
        y_train = y_all[start_idx:end_idx]
        logger.info(
            f"[{device_id}] Sliding window → [{start_idx}, {end_idx}] "
            f"({start_frac*100:.1f}%–{end_frac*100:.1f}%) | "
            f"samples={len(X_train)}"
        )

    # ── Class weights to fix imbalance ────────────────────────────────────
    class_weights = _compute_class_weights(y_train)
    logger.info(
        f"[{device_id}] Class weights: "
        f"Normal={class_weights[0]:.3f}, Warning={class_weights[1]:.3f}, Critical={class_weights[2]:.3f}"
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test), torch.tensor(y_test)
    )

    logger.info(
        f"[{device_id}] Round {round_num} | "
        f"train={len(X_train)} | test={len(X_test)} | "
        f"labels: Normal={int((y_train==0).sum())}, "
        f"Warning={int((y_train==1).sum())}, "
        f"Critical={int((y_train==2).sum())}"
    )

    return train_dataset, test_dataset, class_weights


def partition_all(round_num: int = 0) -> Dict[str, int]:
    """Partition data for all configured edges. Returns sample counts per edge."""
    cfg = get_config()
    sample_counts = {}
    for edge in cfg["edges"]:
        device_id = edge["device"]
        train_ds, _, _ = get_edge_partition(device_id, round_num=round_num)
        sample_counts[edge["id"]] = len(train_ds)
    return sample_counts


if __name__ == "__main__":
    counts = partition_all(round_num=0)
    for edge_id, count in counts.items():
        print(f"{edge_id}: {count} training samples")
