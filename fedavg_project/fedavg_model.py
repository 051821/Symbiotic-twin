import sys
import time
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline_shared.data_utils import BASELINE_DATA_PATH, ensure_baseline_data, load_baseline_split, split_non_iid
from baseline_shared.metrics_utils import build_metrics_payload, save_metrics_payload

METRICS_JSON_PATH = Path(__file__).with_name("metrics.json")
SYMBIOTIC_METRICS_PATH = ROOT / "main" / "logs" / "metrics.json"
BASELINE_POWER_W = 2.5


# ✅ SIMPLE LOCAL TRAIN
def local_train(global_weights, X_local, y_local, classes):
    start = time.time()

    coef, intercept = global_weights

    # Use a moderate iteration budget for stable convergence while staying fast.
    model = LogisticRegression(
        max_iter=12,
        tol=1e-2,
        solver="lbfgs",
        C=0.2,
        warm_start=True,
    )

    # Initialize from global model
    model.classes_ = classes
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()

    model.fit(X_local, y_local)

    preds = model.predict(X_local)
    accuracy = accuracy_score(y_local, preds) * 100

    elapsed = time.time() - start

    return {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "samples": len(X_local),
        "accuracy": round(accuracy, 4),
        "latency_ms": round(elapsed * 1000, 4),
        "energy_j": round(BASELINE_POWER_W * elapsed, 4),
    }


# ✅ FEDAVG
def fedavg_aggregate(updates, total_samples):
    coef_avg = np.zeros_like(updates[0]["coef"])
    int_avg = np.zeros_like(updates[0]["intercept"])

    for update in updates:
        w = update["samples"] / total_samples
        coef_avg += w * update["coef"]
        int_avg += w * update["intercept"]

    return coef_avg, int_avg


def _load_symbiotic_summary():
    if not SYMBIOTIC_METRICS_PATH.exists():
        return None

    with SYMBIOTIC_METRICS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    global_acc = payload.get("global_acc", []) or []
    latency = payload.get("latency", {}) or {}
    energy = payload.get("energy", {}) or {}

    latency_values = []
    for series in latency.values():
        latency_values.extend(series or [])

    energy_values = []
    for series in energy.values():
        energy_values.extend(series or [])

    return {
        "accuracy": round(float(global_acc[-1]), 2) if global_acc else 0.0,
        "latency_ms": round(float(np.mean(latency_values)), 1) if latency_values else 0.0,
        "energy_j": round(float(np.mean(energy_values)), 4) if energy_values else 0.0,
        "source": str(SYMBIOTIC_METRICS_PATH),
    }


def run_fedavg(n_rounds=10, n_clients=4, seed=42):
    ensure_baseline_data()

    X_train, X_test, y_train, y_test = load_baseline_split(test_size=0.2, seed=seed)
    clients = split_non_iid(X_train, y_train, n_clients=n_clients, seed=seed)

    # ✅ Random initialization (NO pretraining)
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    global_coef = np.zeros((n_classes, n_features))
    global_inter = np.zeros(n_classes)

    classes = np.unique(y_train)

    round_metrics = []
    energy_per_round = []

    edge_acc = {f"edge{i+1}": [] for i in range(n_clients)}
    latency = {f"edge{i+1}": [] for i in range(n_clients)}
    edge_energy = {f"edge{i+1}": [] for i in range(n_clients)}
    agg_weights = []
    reputation = {f"edge{i+1}": [] for i in range(n_clients)}

    start_total = time.time()

    for r in range(1, n_rounds + 1):
        print(f"🚀 Round {r} starting...")

        updates = []
        round_start = time.time()

        for idx, (X_c, y_c) in enumerate(clients, start=1):
            result = local_train((global_coef, global_inter), X_c, y_c, classes)

            edge_id = f"edge{idx}"
            edge_acc[edge_id].append(result["accuracy"])
            latency[edge_id].append(result["latency_ms"])
            edge_energy[edge_id].append(result["energy_j"])
            reputation[edge_id].append(1.0)

            updates.append(result)

        total_n = sum(u["samples"] for u in updates)

        agg_weights.append({
            f"edge{i+1}": round(updates[i]["samples"] / total_n, 4)
            for i in range(len(updates))
        })

        # ✅ Aggregate
        global_coef, global_inter = fedavg_aggregate(updates, total_n)

        # ✅ Evaluate (NO training here)
        eval_model = LogisticRegression()
        eval_model.classes_ = classes
        eval_model.coef_ = global_coef.copy()
        eval_model.intercept_ = global_inter.copy()

        preds = eval_model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100

        print(f"Round {r} Accuracy: {round(acc, 2)}%")

        round_elapsed = time.time() - round_start

        round_metrics.append({
            "round": r,
            "accuracy": round(acc, 2),
            "latency_ms": round(round_elapsed * 1000, 1),
            "energy_j": round(BASELINE_POWER_W * round_elapsed, 4),
        })

        energy_per_round.append(BASELINE_POWER_W * round_elapsed)

    total_time = (time.time() - start_total) * 1000
    final = round_metrics[-1]

    symbiotic = _load_symbiotic_summary()

    metrics_payload = build_metrics_payload(
        rounds=[m["round"] for m in round_metrics],
        global_acc=[m["accuracy"] for m in round_metrics],
        edge_acc=edge_acc,
        latency=latency,
        energy=edge_energy,
        agg_weights=agg_weights,
        reputation=reputation,
        metadata={
            "model": "fedavg",
            "baseline_data_path": str(BASELINE_DATA_PATH),
            "n_rounds": n_rounds,
            "n_clients": n_clients,
        },
    )

    save_metrics_payload(METRICS_JSON_PATH, metrics_payload)

    return {
        "fedavg_accuracy": final["accuracy"],
        "fedavg_latency": round(total_time / n_rounds, 1),
        "fedavg_energy": round(sum(energy_per_round) / n_rounds, 4),
        "metrics_path": str(METRICS_JSON_PATH),
    }


if __name__ == "__main__":
    result = run_fedavg()

    print(f" FINAL FedAvg Accuracy: {result['fedavg_accuracy']}%")
    print(f" Latency: {result['fedavg_latency']} ms")
    print(f" Energy: {result['fedavg_energy']} J")