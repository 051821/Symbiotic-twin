"""
Centralized Training Script
--------------------------------
- Uses the shared baseline dataset file also consumed by FedAvg
- Trains a simple MLP
- Uses energy estimation
- Saves metrics.csv for dashboard
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_shared.data_utils import BASELINE_DATA_PATH, ensure_baseline_data, load_baseline_split
from baseline_shared.metrics_utils import build_metrics_payload, save_metrics_payload

BASELINE_POWER_W = 2.5
METRICS_JSON_PATH = Path(__file__).with_name("metrics.json")
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
METRICS_CSV_PATH = SCRIPT_DIR / "metrics.csv"
FAIR_CFG_PATH = ROOT / "main" / "config" / "config.yaml"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_energy(computation_time_s, model=None, power_w=BASELINE_POWER_W):
    if model is not None:
        param_count = count_parameters(model)
        scale_factor = 1.0 + (param_count / 100_000) * 0.5
        power_w = power_w * scale_factor

    energy_j = power_w * computation_time_s
    return round(energy_j, 6)


class EnergyMonitor:
    def __init__(self, model=None, power_w=BASELINE_POWER_W):
        self.model = model
        self.power_w = power_w
        self.energy_j = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self.energy_j = estimate_energy(elapsed, self.model, self.power_w)


with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

fair_cfg = {}
if FAIR_CFG_PATH.exists():
    with FAIR_CFG_PATH.open("r", encoding="utf-8") as f:
        fair_cfg = yaml.safe_load(f) or {}

input_size = config["model"]["input_size"]
hidden_size = config["model"]["hidden_size"]
num_classes = config["model"]["num_classes"]
dropout = config["model"]["dropout"]
test_split = config["data"]["test_split"]

ensure_baseline_data()
X_train, X_test, y_train, y_test = load_baseline_split(test_size=test_split, seed=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = int((fair_cfg.get("system", {}) or {}).get("num_rounds", 10))
metrics = []

print("Starting Centralized Training...\n")
print(f"Canonical shared preprocessed CSV: {BASELINE_DATA_PATH}")

for epoch in range(1, epochs + 1):
    epoch_start = time.perf_counter()
    with EnergyMonitor(model=model) as monitor:
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    energy_j = monitor.energy_j
    latency_ms = (time.perf_counter() - epoch_start) * 1000

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = accuracy_score(y_test, preds.numpy())

    metrics.append([epoch, acc, latency_ms, energy_j])

    print(
        f"Epoch {epoch:02d} | "
        f"Accuracy: {acc:.4f} | "
        f"Latency: {latency_ms:.2f} ms | "
        f"Energy: {energy_j:.6f} J"
    )

metrics_df = pd.DataFrame(metrics, columns=["round", "accuracy", "latency", "energy"])
metrics_df.to_csv(METRICS_CSV_PATH, index=False)

rounds = [row[0] for row in metrics]
accuracy_pct = [round(row[1] * 100, 4) for row in metrics]
latency_ms = [round(row[2], 4) for row in metrics]
energy_j = [round(row[3], 6) for row in metrics]
metrics_payload = build_metrics_payload(
    rounds=rounds,
    global_acc=accuracy_pct,
    edge_acc={"centralized": accuracy_pct},
    latency={"centralized": latency_ms},
    energy={"centralized": energy_j},
    agg_weights=[{"centralized": 1.0} for _ in rounds],
    reputation={"centralized": [1.0 for _ in rounds]},
    metadata={
        "model": "centralized",
        "baseline_data_path": str(BASELINE_DATA_PATH),
        "metrics_csv": str(METRICS_CSV_PATH),
        "epochs": epochs,
    },
)
save_metrics_payload(METRICS_JSON_PATH, metrics_payload)

print("\nTraining completed.")
print(f"Metrics saved to {METRICS_CSV_PATH}")
print(f"Metrics JSON saved to {METRICS_JSON_PATH}")
