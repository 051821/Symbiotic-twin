import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import sys

# Allow root imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import config_loader


def preprocess_data():
    print("Starting preprocessing...")

    raw_data_path = config_loader.get("data", "raw_data_path")
    processed_data_path = config_loader.get("data", "processed_data_path")

    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_data_path}")

    print("Loading dataset...")
    df = pd.read_csv(raw_data_path)

    print(f"Dataset shape: {df.shape}")

    # Convert timestamp
    df["ts"] = pd.to_datetime(df["ts"], unit="s")

    # Convert booleans to int
    df["light"] = df["light"].astype(int)
    df["motion"] = df["motion"].astype(int)

    # -----------------------------
    # Step 1: Create Risk Score
    # -----------------------------
    print("Creating risk score...")

    df["risk_score"] = (
        0.3 * df["smoke"] +
        0.3 * df["co"] +
        0.2 * df["lpg"] +
        0.1 * df["humidity"] +
        0.1 * df["temp"]
    )

    # -----------------------------
    # Step 2: Convert Risk Score to 3 Balanced Classes
    # -----------------------------
    print("Generating balanced 3-class labels using quantiles...")

    df["label"] = pd.qcut(
        df["risk_score"],
        q=3,
        labels=[0, 1, 2]
    ).astype(int)

    # Drop risk score (not needed for training)
    df.drop(columns=["risk_score"], inplace=True)

    # -----------------------------
    # Step 3: Normalize Continuous Features
    # -----------------------------
    continuous_features = ["co", "humidity", "lpg", "smoke", "temp"]

    print("Normalizing continuous features...")
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    # Save processed dataset
    print("Saving processed dataset...")
    df.to_csv(processed_data_path, index=False)

    print("Preprocessing completed successfully.")
    print("Label distribution:")
    print(df["label"].value_counts(normalize=True))


if __name__ == "__main__":
    preprocess_data()

