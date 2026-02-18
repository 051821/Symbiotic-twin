import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
import sys

# Allow import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import config_loader


def load_processed_dataset():
    path = config_loader.get("data", "processed_data_path")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}")

    df = pd.read_csv(path)
    return df


def get_edge_partition(device_id, test_size=0.2, random_state=42):
    """
    Returns train/test tensors for a specific edge device.
    """

    df = load_processed_dataset()

    # Filter by device
    df_device = df[df["device"] == device_id].copy()

    if df_device.empty:
        raise ValueError(f"No data found for device: {device_id}")

    # Drop non-training columns
    label_column = config_loader.get("data", "label_column")
    feature_columns = config_loader.get("data", "feature_columns")

    X = df_device[feature_columns].values
    y = df_device[label_column].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    sample_count = len(X_train)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "sample_count": sample_count
    }


def get_all_devices():
    """
    Returns list of unique device IDs.
    """
    df = load_processed_dataset()
    return df["device"].unique().tolist()


if __name__ == "__main__":
    devices = get_all_devices()
    print("Available devices:", devices)

    for device in devices:
        partition = get_edge_partition(device)
        print(f"\nDevice: {device}")
        print("Training samples:", partition["sample_count"])
