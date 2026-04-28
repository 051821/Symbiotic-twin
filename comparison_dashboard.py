from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="All Model Comparison", page_icon="📊", layout="wide")
st.title("📊 Centralized Comparison Dashboard")
st.caption("Fair comparison across Centralized, FedAvg, and SYMBIOTIC-TWIN using stored metrics files.")


REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATHS = {
    "Centralized": REPO_ROOT / "Centrelized symbiotic twin" / "metrics.json",
    "FedAvg": REPO_ROOT / "fedavg_project" / "metrics.json",
    "SYMBIOTIC-TWIN": REPO_ROOT / "main" / "logs" / "metrics.json",
}


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean_per_round(series_by_edge: Dict[str, List[float]], round_count: int) -> List[float]:
    if not series_by_edge:
        return [0.0] * round_count
    values = []
    for i in range(round_count):
        row = []
        for edge_series in series_by_edge.values():
            if i < len(edge_series):
                row.append(float(edge_series[i]))
        values.append(float(np.mean(row)) if row else 0.0)
    return values


def _summarize_payload(payload: Dict[str, object]) -> Dict[str, object]:
    rounds = payload.get("rounds", []) or []
    global_acc = [float(x) for x in (payload.get("global_acc", []) or [])]
    latency = payload.get("latency", {}) or {}
    energy = payload.get("energy", {}) or {}

    n_rounds = len(rounds)
    lat_per_round = _mean_per_round(latency, n_rounds)
    eng_per_round = _mean_per_round(energy, n_rounds)

    return {
        "rounds": rounds,
        "accuracy_series": global_acc,
        "latency_series": lat_per_round,
        "energy_series": eng_per_round,
        "final_accuracy": global_acc[-1] if global_acc else 0.0,
        "avg_latency": float(np.mean(lat_per_round)) if lat_per_round else 0.0,
        "avg_energy": float(np.mean(eng_per_round)) if eng_per_round else 0.0,
    }


def _truncate_summary(summary: Dict[str, object], n_rounds: int) -> Dict[str, object]:
    rounds = list(summary.get("rounds", []))[:n_rounds]
    acc = list(summary.get("accuracy_series", []))[:n_rounds]
    lat = list(summary.get("latency_series", []))[:n_rounds]
    eng = list(summary.get("energy_series", []))[:n_rounds]
    return {
        "rounds": rounds,
        "accuracy_series": acc,
        "latency_series": lat,
        "energy_series": eng,
        "final_accuracy": acc[-1] if acc else 0.0,
        "avg_latency": float(np.mean(lat)) if lat else 0.0,
        "avg_energy": float(np.mean(eng)) if eng else 0.0,
    }


available: Dict[str, Dict[str, object]] = {}
missing = []
for model_name, path in MODEL_PATHS.items():
    if path.exists():
        payload = _load_json(path)
        available[model_name] = _summarize_payload(payload)
    else:
        missing.append(f"{model_name}: {path}")

if missing:
    st.warning("Some model metrics are missing. Run these first:\n\n" + "\n".join(missing))

if len(available) < 2:
    st.error("Need at least two model metrics files for comparison.")
    st.stop()

st.subheader("Model Metric Trends (Each Model Shows Its Own Metrics)")
for model_name, summary in available.items():
    st.markdown(f"### {model_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Final Accuracy (%)", f"{summary['final_accuracy']:.2f}")
    c2.metric("Average Latency (ms)", f"{summary['avg_latency']:.2f}")
    c3.metric("Average Energy (J)", f"{summary['avg_energy']:.4f}")

    rounds = summary["rounds"]
    if rounds:
        df = pd.DataFrame(
            {
                "round": rounds,
                "accuracy": summary["accuracy_series"],
                "latency_ms": summary["latency_series"],
                "energy_j": summary["energy_series"],
            }
        )
        st.line_chart(df.set_index("round")[["accuracy", "latency_ms", "energy_j"]])
    else:
        st.info("No round-level series in this metrics file.")

st.divider()
st.subheader("Side-by-Side Comparison: All Available Models")

models = list(available.keys())
common_rounds = min((len(available[m]["rounds"]) for m in models), default=0)
if common_rounds > 0:
    fair_view = {m: _truncate_summary(available[m], common_rounds) for m in models}
    st.caption(f"Fair view uses the first {common_rounds} rounds for every model.")
else:
    fair_view = available

accuracy = [fair_view[m]["final_accuracy"] for m in models]
latency = [fair_view[m]["avg_latency"] for m in models]
energy = [fair_view[m]["avg_energy"] for m in models]
colors = ["#5b7fa6", "#4a9e7f", "#3266ad", "#b56576", "#e76f51"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, vals, title, unit in zip(
    axes,
    [accuracy, latency, energy],
    ["Accuracy", "Avg Latency", "Avg Energy"],
    ["%", "ms", "J"],
):
    bars = ax.bar(models, vals, color=colors[: len(models)], width=0.6)
    top = max(vals) * 1.2 if max(vals) > 0 else 1.0
    ax.set_ylim(0, top)
    ax.set_title(f"{title} ({unit})", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=10)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top * 0.02,
            f"{val:.2f}{unit}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
plt.tight_layout()
st.pyplot(fig)

st.subheader("Fair Comparison (Normalized to 0-1, Higher is Better)")
acc_arr = np.array(accuracy, dtype=float)
lat_arr = np.array(latency, dtype=float)
eng_arr = np.array(energy, dtype=float)

acc_n = acc_arr / max(acc_arr.max(), 1e-9)
lat_n = 1.0 - (lat_arr / max(lat_arr.max(), 1e-9))
eng_n = 1.0 - (eng_arr / max(eng_arr.max(), 1e-9))

x = np.arange(len(models))
width = 0.25
fig2, ax2 = plt.subplots(figsize=(10, 5))
b1 = ax2.bar(x - width, acc_n, width=width, label="Accuracy", color="#2F6CAD")
b2 = ax2.bar(x, lat_n, width=width, label="Latency (inverted)", color="#56A67A")
b3 = ax2.bar(x + width, eng_n, width=width, label="Energy (inverted)", color="#F2C230")
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Normalized Score")
ax2.grid(axis="y", linestyle="--", alpha=0.25)
ax2.legend()

for bars in (b1, b2, b3):
    for bar in bars:
        v = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

plt.tight_layout()
st.pyplot(fig2)

st.subheader("Balanced Overall Score (Fair Rounds)")
balanced_scores = (acc_n + lat_n + eng_n) / 3.0
rank_df = pd.DataFrame(
    {"Model": models, "Balanced Score": balanced_scores}
).sort_values("Balanced Score", ascending=False, ignore_index=True)
rank_df["Rank"] = rank_df.index + 1
st.dataframe(rank_df[["Rank", "Model", "Balanced Score"]], use_container_width=True)

st.info(
    "This dashboard now compares models over a shared round budget (minimum common rounds). "
    "For strict fairness, re-run all models with the same dataset split, rounds, and metric formulas."
)
