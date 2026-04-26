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


def local_train(global_weights, X_local, y_local):
    start = time.time()
    model = LogisticRegression(max_iter=200, warm_start=False, random_state=42)
    model.fit(X_local, y_local)
    preds = model.predict(X_local)
    accuracy = accuracy_score(y_local, preds) * 100
    elapsed = time.time() - start
    energy = round((0.9 + 0.15 * len(np.unique(y_local))) * elapsed, 4)
    return {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "samples": len(X_local),
        "accuracy": round(accuracy, 4),
        "latency_ms": round(elapsed * 1000, 4),
        "energy_j": energy,
    }


def fedavg_aggregate(updates, total_samples):
    coef_avg = np.zeros_like(updates[0]["coef"])
    int_avg = np.zeros_like(updates[0]["intercept"])
    for update in updates:
        w = update["samples"] / total_samples
        coef_avg += w * update["coef"]
        int_avg += w * update["intercept"]
    return coef_avg, int_avg


def _load_symbiotic_summary():
    """
    Load real Symbiotic-Twin metrics if available.
    Returns None when the Symbiotic pipeline has not been run yet.
    """
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

    latency_mean = round(float(np.mean(latency_values)), 1) if latency_values else 0.0
    energy_mean = round(float(np.mean(energy_values)), 4) if energy_values else 0.0

    return {
        "accuracy": round(float(global_acc[-1]), 2) if global_acc else 0.0,
        "latency_ms": latency_mean,
        "energy_j": energy_mean,
        "source": str(SYMBIOTIC_METRICS_PATH),
    }


def run_fedavg(n_rounds=20, n_clients=4, seed=42):
    ensure_baseline_data()
    X_train, X_test, y_train, y_test = load_baseline_split(test_size=0.2, seed=seed)
    clients = split_non_iid(X_train, y_train, n_clients=n_clients, seed=seed)

    global_model = LogisticRegression(max_iter=200, random_state=42)
    global_model.fit(X_train, y_train)
    global_coef = global_model.coef_.copy()
    global_inter = global_model.intercept_.copy()

    round_metrics = []
    energy_per_round = []
    edge_acc = {f"edge{i+1}": [] for i in range(n_clients)}
    latency = {f"edge{i+1}": [] for i in range(n_clients)}
    edge_energy = {f"edge{i+1}": [] for i in range(n_clients)}
    agg_weights = []
    reputation = {f"edge{i+1}": [] for i in range(n_clients)}
    start_total = time.time()

    for r in range(1, n_rounds + 1):
        updates = []
        round_start = time.time()
        for idx, (X_c, y_c) in enumerate(clients, start=1):
            result = local_train((global_coef, global_inter), X_c, y_c)
            edge_id = f"edge{idx}"
            edge_acc[edge_id].append(result["accuracy"])
            latency[edge_id].append(result["latency_ms"])
            edge_energy[edge_id].append(result["energy_j"])
            reputation[edge_id].append(1.0)
            updates.append(result)
        total_n = sum(u["samples"] for u in updates)
        weight_map = {
            f"edge{i+1}": round(updates[i]["samples"] / total_n, 4)
            for i in range(len(updates))
        }
        agg_weights.append(weight_map)
        global_coef, global_inter = fedavg_aggregate(updates, total_n)

        eval_model = LogisticRegression(max_iter=500, random_state=42)
        eval_model.fit(X_train, y_train)
        eval_model.coef_ = global_coef.copy()
        eval_model.intercept_ = global_inter.copy()
        preds = eval_model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100

        round_elapsed = time.time() - round_start
        lat = round_elapsed * 1000
        round_energy = round((0.9 + 0.15 * n_clients) * round_elapsed, 4)

        round_metrics.append(
            {
                "round": r,
                "accuracy": round(acc, 2),
                "latency_ms": round(lat, 1),
                "energy_j": round(round_energy, 4),
            }
        )
        energy_per_round.append(round_energy)

    total_time = (time.time() - start_total) * 1000

    final = round_metrics[-1]
    central_start = time.time()
    centralized_model = LogisticRegression(max_iter=500, random_state=seed)
    centralized_model.fit(X_train, y_train)
    centralized_preds = centralized_model.predict(X_test)
    centralized_elapsed = time.time() - central_start
    centralized = {
        "accuracy": round(accuracy_score(y_test, centralized_preds) * 100, 2),
        "latency_ms": round(centralized_elapsed * 1000, 1),
        "energy_j": round(2.5 * centralized_elapsed, 4),
    }
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

    summary = {
        "fedavg_accuracy": final["accuracy"],
        "fedavg_latency": round(total_time / n_rounds, 1),
        "fedavg_energy": round(sum(energy_per_round) / n_rounds, 4),
        "centralized": centralized,
        "symbiotic": symbiotic,
        "symbiotic_available": symbiotic is not None,
        "round_metrics": round_metrics,
        "n_rounds": n_rounds,
        "n_clients": n_clients,
        "baseline_data_path": str(BASELINE_DATA_PATH),
        "metrics_payload": metrics_payload,
        "metrics_path": str(METRICS_JSON_PATH),
    }
    return summary


if __name__ == "__main__":
    result = run_fedavg()
    print(f"FedAvg  -> Accuracy: {result['fedavg_accuracy']}%  "
          f"Latency: {result['fedavg_latency']} ms  "
          f"Energy: {result['fedavg_energy']} J")
    print(f"Canonical shared preprocessed CSV: {result['baseline_data_path']}")
    print(f"Metrics JSON saved to {result['metrics_path']}")
