from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
LABEL_COL = "label"

ROOT_DIR = Path(__file__).resolve().parents[1]
CANONICAL_DATA_DIR = ROOT_DIR / "main" / "data" / "processed"
BASELINE_DATA_PATH = CANONICAL_DATA_DIR / "processed.csv"
CANONICAL_SCALER_PATH = CANONICAL_DATA_DIR / "scaler.pkl"
CANONICAL_RAW_PATH = ROOT_DIR / "main" / "data" / "iot_telemetry_data.csv"
BASELINE_META_PATH = CANONICAL_DATA_DIR / "baseline_meta.json"
BASELINE_DATA_VERSION = "v3_train_only_scaling_temporal_split"


def create_labels(df: pd.DataFrame) -> pd.Series:
    conditions = [
        (df["smoke"] > 0.10) | (df["co"] > 0.005),
        (df["temp"] > 90.0) | (df["lpg"] > 0.007),
    ]
    return np.select(conditions, [2, 1], default=0).astype(int)


def ensure_baseline_data(force: bool = False) -> Path:
    CANONICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if BASELINE_DATA_PATH.exists() and not force and BASELINE_META_PATH.exists():
        try:
            meta = json.loads(BASELINE_META_PATH.read_text(encoding="utf-8"))
            if meta.get("version") == BASELINE_DATA_VERSION:
                return BASELINE_DATA_PATH
        except Exception:
            # If metadata is corrupted, regenerate to ensure fair preprocessing.
            pass

    if not CANONICAL_RAW_PATH.exists():
        raise FileNotFoundError(f"Canonical raw IoT dataset not found: {CANONICAL_RAW_PATH}")

    df = pd.read_csv(CANONICAL_RAW_PATH)
    df["light"] = df["light"].astype(int)
    df["motion"] = df["motion"].astype(int)
    df[LABEL_COL] = create_labels(df)
    # Keep canonical baseline data unscaled to avoid train-test leakage.
    # Scaling is applied in load_baseline_split() with train-only fit.
    keep_cols = [*FEATURE_COLS, LABEL_COL]
    if "ts" in df.columns:
        keep_cols.append("ts")
    if "device" in df.columns:
        keep_cols.append("device")
    df[keep_cols].to_csv(BASELINE_DATA_PATH, index=False)
    BASELINE_META_PATH.write_text(
        json.dumps({"version": BASELINE_DATA_VERSION}, indent=2),
        encoding="utf-8",
    )
    return BASELINE_DATA_PATH


def load_baseline_dataframe() -> pd.DataFrame:
    return pd.read_csv(ensure_baseline_data())


def load_baseline_split(test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_baseline_dataframe()
    if "ts" in df.columns:
        # Harder and more realistic evaluation: train on earlier data, test on later data.
        df = df.sort_values("ts").reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        X_train = train_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_train = train_df[LABEL_COL].to_numpy(dtype=np.int64)
        X_test = test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y_test = test_df[LABEL_COL].to_numpy(dtype=np.int64)
    else:
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        y = df[LABEL_COL].to_numpy(dtype=np.int64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test


def split_non_iid(X: np.ndarray, y: np.ndarray, n_clients: int = 4, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(y).tolist())
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for class_pos, cls in enumerate(classes):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)

        # Keep the split non-IID by giving each class a few preferred clients,
        # while still reserving a tiny amount for everyone so local training remains valid.
        primary_client = class_pos % n_clients
        weights = np.full(n_clients, 0.15, dtype=np.float64)
        weights[primary_client] = 1.0
        if n_clients > 1:
            weights[(primary_client + 1) % n_clients] = 0.55
        weights = weights / weights.sum()

        base_counts = np.ones(n_clients, dtype=int)
        remaining = len(cls_idx) - base_counts.sum()
        if remaining < 0:
            raise ValueError(f"Not enough samples in class {cls} for {n_clients} clients.")

        counts = base_counts + rng.multinomial(remaining, weights)
        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            client_indices[client_id].extend(cls_idx[start:end].tolist())
            start = end

    clients: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in client_indices:
        shuffled = np.array(idx, dtype=np.int64)
        rng.shuffle(shuffled)
        clients.append((X[shuffled], y[shuffled]))
    return clients
