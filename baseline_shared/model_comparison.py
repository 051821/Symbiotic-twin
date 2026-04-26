from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _list_mean(values: List[float]) -> float:
    return float(mean(values)) if values else 0.0


def _summary_from_payload(payload: Dict[str, object]) -> Dict[str, float]:
    global_acc = payload.get("global_acc", []) or []
    latency = payload.get("latency", {}) or {}
    energy = payload.get("energy", {}) or {}

    latency_values: List[float] = []
    for series in latency.values():
        latency_values.extend(series or [])

    energy_values: List[float] = []
    for series in energy.values():
        energy_values.extend(series or [])

    return {
        "accuracy": float(global_acc[-1]) if global_acc else 0.0,
        "latency_ms": _list_mean(latency_values),
        "energy_j": _list_mean(energy_values),
    }


def _normalize(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    acc_vals = [m["accuracy"] for m in metrics.values()]
    lat_vals = [m["latency_ms"] for m in metrics.values()]
    eng_vals = [m["energy_j"] for m in metrics.values()]

    max_acc = max(acc_vals) if max(acc_vals) > 0 else 1.0
    max_lat = max(lat_vals) if max(lat_vals) > 0 else 1.0
    max_eng = max(eng_vals) if max(eng_vals) > 0 else 1.0

    normalized: Dict[str, Dict[str, float]] = {}
    for model_name, m in metrics.items():
        normalized[model_name] = {
            # Accuracy: higher is better
            "accuracy": m["accuracy"] / max_acc,
            # Latency/Energy: lower is better, so invert after scaling
            "latency_ms": 1.0 - (m["latency_ms"] / max_lat),
            "energy_j": 1.0 - (m["energy_j"] / max_eng),
        }
    return normalized


def load_model_summaries(repo_root: Path) -> Dict[str, Dict[str, float]]:
    paths = {
        "Symbiotic-Twin": repo_root / "main" / "logs" / "metrics.json",
        "FedAvg": repo_root / "fedavg_project" / "metrics.json",
        "Centralized": repo_root / "Centrelized symbiotic twin" / "metrics.json",
    }

    summaries: Dict[str, Dict[str, float]] = {}
    missing: List[str] = []
    for name, path in paths.items():
        if path.exists():
            payload = _read_metrics(path)
            summaries[name] = _summary_from_payload(payload)
            continue

        # Fallback for centralized baseline when only metrics.csv exists.
        if name == "Centralized":
            csv_path = repo_root / "Centrelized symbiotic twin" / "metrics.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                acc = float(df["accuracy"].iloc[-1]) if not df.empty else 0.0
                if acc <= 1.0:
                    acc *= 100.0
                summaries[name] = {
                    "accuracy": acc,
                    "latency_ms": float(df["latency"].mean()) if "latency" in df else 0.0,
                    "energy_j": float(df["energy"].mean()) if "energy" in df else 0.0,
                }
                continue

        missing.append(str(path))

    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Missing required metrics files. Run each model first to generate metrics:\n"
            f"{missing_text}"
        )
    return summaries


def build_comparison_figures(
    summaries: Dict[str, Dict[str, float]]
) -> Tuple[plt.Figure, plt.Figure]:
    models = list(summaries.keys())
    colors = ["#2F6CAD", "#56A67A", "#F2C230"]

    raw_accuracy = [summaries[m]["accuracy"] for m in models]
    raw_latency = [summaries[m]["latency_ms"] for m in models]
    raw_energy = [summaries[m]["energy_j"] for m in models]

    normalized = _normalize(summaries)
    norm_accuracy = [normalized[m]["accuracy"] for m in models]
    norm_latency = [normalized[m]["latency_ms"] for m in models]
    norm_energy = [normalized[m]["energy_j"] for m in models]

    # Figure 1: Raw metrics (same chart structure for all metrics)
    fig_raw, axes_raw = plt.subplots(1, 3, figsize=(14, 4.5))
    fig_raw.suptitle("Model Comparison (Raw Metrics)", fontsize=13, fontweight="bold")
    raw_sets = [
        ("Accuracy (%)", raw_accuracy),
        ("Latency (ms)", raw_latency),
        ("Energy (J)", raw_energy),
    ]
    for ax, (title, values) in zip(axes_raw, raw_sets):
        bars = ax.bar(models, values, color=colors, width=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", rotation=10)
        upper = max(values) * 1.2 if max(values) > 0 else 1
        ax.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig_raw.tight_layout()

    # Figure 2: Fair normalized comparison
    fig_norm, ax_norm = plt.subplots(figsize=(9, 5))
    fig_norm.suptitle(
        "Fair Comparison (Normalized to 0-1, Higher is Better)",
        fontsize=13,
        fontweight="bold",
    )
    x = np.arange(len(models))
    width = 0.25

    b1 = ax_norm.bar(x - width, norm_accuracy, width=width, label="Accuracy", color="#2F6CAD")
    b2 = ax_norm.bar(x, norm_latency, width=width, label="Latency (inverted)", color="#56A67A")
    b3 = ax_norm.bar(x + width, norm_energy, width=width, label="Energy (inverted)", color="#F2C230")

    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(models)
    ax_norm.set_ylim(0, 1.05)
    ax_norm.set_ylabel("Normalized Score")
    ax_norm.set_axisbelow(True)
    ax_norm.grid(axis="y", linestyle="--", alpha=0.3)
    ax_norm.legend(loc="upper right")

    for bars in (b1, b2, b3):
        for bar in bars:
            value = bar.get_height()
            ax_norm.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig_norm.tight_layout()
    return fig_raw, fig_norm


def print_metric_table(summaries: Dict[str, Dict[str, float]]) -> None:
    print("\n=== Unified Comparison Table ===")
    print(f"{'Model':<16} {'Accuracy(%)':>12} {'Latency(ms)':>12} {'Energy(J)':>12}")
    print("-" * 56)
    for model, m in summaries.items():
        print(f"{model:<16} {m['accuracy']:>12.2f} {m['latency_ms']:>12.2f} {m['energy_j']:>12.4f}")

