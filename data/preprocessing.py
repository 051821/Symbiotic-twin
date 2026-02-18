import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys

# Allow imports from root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import config_loader


def create_label(row):
    """
    Create risk classification label:
    0 -> Normal
    1 -> Warning
    2 -> Critical
    """

    if row["smoke"] > 0.8 or row["co"] > 0.7 or row["lpg"] > 0.7:
        return 2  # Critical
    elif row["humidity"] > 75 or row["temp"] > 90:
        return 1  # Warning
    else:
        return 0  # Normal


def preprocess_data():
    print("Starting preprocessing...")

    raw_data_path = config_loader.get("data", "raw_data_path")
    processed_data_path = config_loader.get("data", "processed_data_path")
    feature_columns = config_loader.get("data", "feature_columns")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")

    print("Loading dataset...")
    df = pd.read_csv(raw_data_path)

    print(f"Dataset shape: {df.shape}")

    # Convert timestamp
    print("Converting timestamp...")
    df["ts"] = pd.to_datetime(df["ts"], unit="s")

    # Create label
    print("Creating classification labels...")
    df["label"] = df.apply(create_label, axis=1)

    # Convert boolean columns
    if "light" in df.columns:
        df["light"] = df["light"].astype(int)

    if "motion" in df.columns:
        df["motion"] = df["motion"].astype(int)

    # Normalize continuous features
    continuous_features = ["co", "humidity", "lpg", "smoke", "temp"]

    print("Normalizing continuous features...")
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    # Save processed dataset
    print("Saving processed dataset...")
    df.to_csv(processed_data_path, index=False)

    print("Preprocessing completed successfully.")
    print(f"Processed dataset saved at: {processed_data_path}")


if __name__ == "__main__":
    preprocess_data()
