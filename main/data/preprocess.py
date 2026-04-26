"""
data/preprocess.py
Loads raw IoT telemetry CSV, cleans it, creates 3-class risk labels,
normalises features, and saves the processed DataFrame.

Label logic (3 classes, checked in order):
  Critical (2) : smoke > 0.10  OR  co > 0.005
  Warning  (1) : temp  > 90    OR  lpg > 0.007
  Normal   (0) : everything else
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
    """
    Assign 3 risk labels using np.select (checked top-to-bottom):
      2 = Critical, 1 = Warning, 0 = Normal
    """
    conditions = [
        (df["smoke"] > 0.10) | (df["co"] > 0.005),
        (df["temp"]  > 90.0) | (df["lpg"] > 0.007),
    ]
    choices = [2, 1]
    return np.select(conditions, choices, default=0).astype(int)


def preprocess(raw_path: str = None, out_dir: str = None) -> pd.DataFrame:
    cfg      = get_config()
    raw_path = Path(raw_path or cfg["data"]["raw_path"])
    out_dir  = Path(out_dir  or cfg["data"]["processed_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Raw shape: {df.shape}")

    df["ts"]     = pd.to_datetime(df["ts"], unit="s")
    df           = df.sort_values("ts").reset_index(drop=True)
    df["light"]  = df["light"].astype(int)
    df["motion"] = df["motion"].astype(int)

    df[LABEL_COL] = create_labels(df)
    dist = df[LABEL_COL].value_counts().to_dict()
    logger.info(
        f"Label distribution — Normal: {dist.get(0,0)} "
        f"| Warning: {dist.get(1,0)} | Critical: {dist.get(2,0)}"
    )

    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    scaler_path = out_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved → {scaler_path}")

    out_path = out_dir / "processed.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Processed data saved → {out_path}  shape={df.shape}")

    return df


if __name__ == "__main__":
    preprocess()
