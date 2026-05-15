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
from sklearn.model_selection import train_test_split

from config.loader import get_config
from config.logging_config import setup_logger

logger = setup_logger("partition")

FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
LABEL_COL    = "label"
_DATAFRAME_CACHE: Dict[Path, pd.DataFrame] = {}
_DEVICE_CACHE: Dict[Tuple[Path, str, str], Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]] = {}
_TEST_DATASET_CACHE: Dict[Tuple[Path, str, str], torch.utils.data.TensorDataset] = {}


def _expand_window_for_class_diversity(
    X_all: np.ndarray,
    y_all: np.ndarray,
    start_idx: int,
    end_idx: int,
    train_end_idx: int,
    min_classes: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Expand a sliding window until it contains at least `min_classes` labels.
    Keeps temporal ordering and only expands bounds (does not shuffle).
    """
    left = max(0, start_idx)
    right = min(train_end_idx, end_idx)
    if right <= left:
        return X_all[left:right], y_all[left:right], left, right

    step = max(128, int(0.05 * train_end_idx))
    for _ in range(12):
        labels = np.unique(y_all[left:right])
        if len(labels) >= min_classes:
            break
        grew = False
        if left > 0:
            left = max(0, left - step)
            grew = True
        if right < train_end_idx:
            right = min(train_end_idx, right + step)
            grew = True
        if not grew:
            break
    return X_all[left:right], y_all[left:right], left, right


def _compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute per-class weights to handle class imbalance.
    Normal readings dominate the dataset — this prevents the model
    from ignoring Warning / Critical classes.
    """
    classes, counts = np.unique(y, return_counts=True)
    cfg = get_config()
    total = len(y)
    weights = np.ones(3, dtype=np.float32)
    for cls, cnt in zip(classes, counts):
        raw_weight = total / (len(classes) * max(cnt, 1))
        weights[int(cls)] = np.sqrt(raw_weight)

    multipliers = cfg["data"].get("class_weight_multipliers", [1.0, 1.0, 1.0])
    if multipliers:
        weights *= np.array(multipliers[:3], dtype=np.float32)

    # Prevent instability when sampling has already boosted rare classes.
    weights = np.clip(weights, 0.5, 5.0)
    return torch.tensor(weights, dtype=torch.float32)


def get_edge_partition(
    device_id: str,
    round_num: int = 0,
    processed_path: str = None,
    partitions_path: str = None,
    max_train_samples_override: int = None,
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
    test_split_mode = cfg["data"].get("test_split_mode", "temporal")
    strategy        = cfg["data"].get("window_strategy", "sliding")
    win_frac        = cfg["data"].get("window_fraction", 0.3)
    win_step        = cfg["data"].get("window_step", 0.1)

    partitions_path.mkdir(parents=True, exist_ok=True)

    # ── Load device data ──────────────────────────────────────────────────
    logger.info(f"[{device_id}] Loading partition from {processed_path} | round={round_num} | strategy={strategy}")
    if processed_path not in _DATAFRAME_CACHE:
        _DATAFRAME_CACHE[processed_path] = pd.read_csv(processed_path)
    df = _DATAFRAME_CACHE[processed_path]

    cache_key = (processed_path, device_id, test_split_mode)
    cached = _DEVICE_CACHE.get(cache_key)
    if cached is None:
        device_df = df[df["device"] == device_id].reset_index(drop=True)
        if device_df.empty:
            raise ValueError(f"No data found for device: {device_id}")

        # Data is already sorted by timestamp from preprocess.py
        X_all = device_df[FEATURE_COLS].values.astype(np.float32)
        y_all = device_df[LABEL_COL].values.astype(np.int64)
        n = len(X_all)
        if test_split_mode == "stratified":
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X_all,
                y_all,
                test_size=test_split,
                random_state=cfg["system"].get("seed", 42),
                stratify=y_all,
            )
            X_all = X_train_full
            y_all = y_train_full
            test_start = len(X_all)
        else:
            test_start = int(n * (1 - test_split))
            X_test = X_all[test_start:]
            y_test = y_all[test_start:]
        _DEVICE_CACHE[cache_key] = (X_all, y_all, test_start, X_test, y_test)
    else:
        X_all, y_all, test_start, X_test, y_test = cached

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

        min_classes = int(cfg["data"].get("min_classes_per_window", 2) or 2)
        if len(np.unique(y_train)) < min_classes:
            X_train, y_train, expanded_start, expanded_end = _expand_window_for_class_diversity(
                X_all=X_all,
                y_all=y_all,
                start_idx=start_idx,
                end_idx=end_idx,
                train_end_idx=train_end_idx,
                min_classes=min_classes,
            )
            logger.info(
                f"[{device_id}] Sliding window expanded for class diversity "
                f"→ [{expanded_start}, {expanded_end}] | classes={len(np.unique(y_train))}"
            )
        logger.info(
            f"[{device_id}] Sliding window → [{start_idx}, {end_idx}] "
            f"({start_frac*100:.1f}%–{end_frac*100:.1f}%) | "
            f"samples={len(X_train)}"
        )

    # Optional cap to keep per-round compute bounded and comparable.
    # This helps reduce latency/energy without changing test-set evaluation.
    calibration_n = int(cfg["data"].get("calibration_samples_per_class", 0) or 0)
    if calibration_n > 0:
        rng = np.random.default_rng(cfg["system"].get("seed", 42) + 10_000 + round_num)
        cal_X = []
        cal_y = []
        for cls in range(3):
            cls_df = df[df[LABEL_COL] == cls]
            if cls_df.empty:
                continue
            take_n = min(calibration_n, len(cls_df))
            sample_df = cls_df.sample(
                n=take_n,
                replace=False,
                random_state=cfg["system"].get("seed", 42) + round_num + cls,
            )
            cal_X.append(sample_df[FEATURE_COLS].values.astype(np.float32))
            cal_y.append(sample_df[LABEL_COL].values.astype(np.int64))
        if cal_X:
            X_train = np.concatenate([X_train, *cal_X], axis=0)
            y_train = np.concatenate([y_train, *cal_y], axis=0)
            order = rng.permutation(len(y_train))
            X_train = X_train[order]
            y_train = y_train[order]
            logger.info(
                f"[{device_id}] Added calibration samples: "
                f"{sum(len(y) for y in cal_y)} ({calibration_n}/class target)"
            )

    if max_train_samples_override is not None:
        max_train = int(max_train_samples_override)
    else:
        max_train = int(cfg["system"].get("max_train_samples_per_round", 0) or 0)
    if max_train > 0 and len(X_train) > max_train:
        # Preserve rare Warning/Critical examples inside the cap. This keeps
        # latency bounded without letting majority Normal samples dominate.
        rng = np.random.default_rng(cfg["system"].get("seed", 42) + round_num)
        classes, counts = np.unique(y_train, return_counts=True)
        min_per_class = int(cfg["data"].get("min_samples_per_class_cap", 0) or 0)
        min_fraction = float(cfg["data"].get("class_balance_cap_fraction", 0.0) or 0.0)
        balance_floor = max(min_per_class, int(round(max_train * min_fraction)))
        oversample_rare = bool(cfg["data"].get("oversample_rare_classes", False))
        selected_idx = []
        total = len(y_train)
        selected_real = set()
        targets = {}

        for cls, cnt in zip(classes, counts):
            targets[int(cls)] = max(balance_floor, int(round(max_train * (cnt / total))))

        excess = sum(targets.values()) - max_train
        if excess > 0:
            for cls in sorted(targets, key=lambda c: targets[c], reverse=True):
                if excess <= 0:
                    break
                reducible = max(0, targets[cls] - balance_floor)
                cut = min(excess, reducible)
                targets[cls] -= cut
                excess -= cut
        if excess > 0:
            for cls in sorted(targets, key=lambda c: targets[c], reverse=True):
                if excess <= 0:
                    break
                cut = min(excess, max(0, targets[cls] - 1))
                targets[cls] -= cut
                excess -= cut

        for cls, _cnt in zip(classes, counts):
            cls_idx = np.where(y_train == cls)[0]
            target = min(targets[int(cls)], max_train)
            replace = oversample_rare and len(cls_idx) < target
            if not replace:
                target = min(target, len(cls_idx))
            pick = rng.choice(cls_idx, size=target, replace=replace)
            selected_idx.append(pick)
            selected_real.update(int(i) for i in np.unique(pick))

        if selected_idx:
            idx = np.concatenate(selected_idx)
            if len(idx) > max_train:
                idx = rng.choice(idx, size=max_train, replace=False)
            elif len(idx) < max_train:
                remaining = np.setdiff1d(
                    np.arange(len(y_train)),
                    np.fromiter(selected_real, dtype=np.int64),
                    assume_unique=False,
                )
                if len(remaining) > 0:
                    add_n = min(max_train - len(idx), len(remaining))
                    add_idx = rng.choice(remaining, size=add_n, replace=False)
                    idx = np.concatenate([idx, add_idx])

            rng.shuffle(idx)
            X_train = X_train[idx]
            y_train = y_train[idx]
            logger.info(
                f"[{device_id}] Balanced cap applied: {len(X_train)} samples | "
                f"floor={balance_floor} | oversample_rare={oversample_rare}"
            )

    # ── Class weights to fix imbalance ────────────────────────────────────
    class_weights = _compute_class_weights(y_train)
    logger.info(
        f"[{device_id}] Class weights: "
        f"Normal={class_weights[0]:.3f}, Warning={class_weights[1]:.3f}, Critical={class_weights[2]:.3f}"
    )

    # Build train tensors with zero-copy path when possible.
    X_train_t = torch.from_numpy(np.ascontiguousarray(X_train)).float()
    y_train_t = torch.from_numpy(np.ascontiguousarray(y_train)).long()
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)

    # Test partition is fixed per device/split mode, so cache the tensor dataset once.
    test_dataset = _TEST_DATASET_CACHE.get(cache_key)
    if test_dataset is None:
        X_test_t = torch.from_numpy(np.ascontiguousarray(X_test)).float()
        y_test_t = torch.from_numpy(np.ascontiguousarray(y_test)).long()
        test_dataset = torch.utils.data.TensorDataset(X_test_t, y_test_t)
        _TEST_DATASET_CACHE[cache_key] = test_dataset

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
