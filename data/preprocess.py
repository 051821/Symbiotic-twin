"""
data/preprocess.py
Loads raw IoT telemetry CSV, cleans it, creates risk labels,
normalises features, and saves the processed DataFrame.

Label logic
-----------
  Critical : smoke > 0.1  OR  co > 0.005
  Warning  : temp  > 90   OR  lpg > 0.007
  Normal   : everything else
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

from config.loader import get_config
from config.logging_config import setup_logger

logger = setup_logger("preprocess")

FEATURE_COLS = ["co", "humidity", "light", "lpg", "motion", "smoke", "temp"]
LABEL_COL    = "label"
DEVICE_COL   = "device"


def create_labels(df: pd.DataFrame) -> pd.Series:
    """Assign risk labels: 2=Critical, 1=Warning, 0=Normal."""
    conditions = [
        (df["smoke"] > 0.10) | (df["co"] > 0.005),          # Critical
        (df["temp"]  > 90.0) | (df["lpg"] > 0.007),         # Warning
    ]
    choices = [2, 1]
    return np.select(conditions, choices, default=0).astype(int)


def preprocess(raw_path: str = None, out_dir: str = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Args:
        raw_path : Path to raw CSV file.
        out_dir  : Directory to write processed data and scaler.

    Returns:
        Processed DataFrame with features, label, device columns.
    """
    cfg = get_config()
    raw_path = Path(raw_path or cfg["data"]["raw_path"])
    out_dir  = Path(out_dir  or cfg["data"]["processed_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    # --- Timestamp ---
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values("ts").reset_index(drop=True)

    # --- Boolean → int ---
    df["light"]  = df["light"].astype(int)
    df["motion"] = df["motion"].astype(int)

    # --- Labels ---
    df[LABEL_COL] = create_labels(df)
    logger.info(f"Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")

    # --- Normalise features ---
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    scaler_path = out_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    out_path = out_dir / "processed.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Processed data saved to {out_path}  shape={df.shape}")

    return df


if __name__ == "__main__":
    preprocess()
