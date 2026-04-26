import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from fedavg_model import run_fedavg

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedAvg vs SYMBIOTIC-TWIN Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 FedAvg vs SYMBIOTIC-TWIN — Performance Dashboard")
st.markdown("Centralized and FedAvg both read the same canonical preprocessed CSV used for comparison.")

# ── Sidebar controls ───────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulation Settings")
n_rounds  = st.sidebar.slider("Federated Rounds",  5, 50, 20)
n_clients = st.sidebar.slider("Number of Edge Nodes", 2, 8, 4)
seed      = st.sidebar.number_input("Random Seed", 0, 999, 42)
st.sidebar.markdown("### Overall Score Weights")
w_acc = st.sidebar.slider("Accuracy weight", 0.0, 1.0, 0.4, 0.05)
w_lat = st.sidebar.slider("Latency weight", 0.0, 1.0, 0.3, 0.05)
w_eng = st.sidebar.slider("Energy weight", 0.0, 1.0, 0.3, 0.05)

if st.sidebar.button("▶ Run Simulation", use_container_width=True):
    st.session_state.result = run_fedavg(n_rounds=n_rounds, n_clients=n_clients, seed=seed)

if "result" not in st.session_state:
    with st.spinner("Running FedAvg simulation..."):
        st.session_state.result = run_fedavg(n_rounds=n_rounds, n_clients=n_clients, seed=seed)

res = st.session_state.result
cen = res["centralized"]
sym = res["symbiotic"]
fed_acc = res["fedavg_accuracy"]
fed_lat = res["fedavg_latency"]
fed_eng = res["fedavg_energy"]
metrics_payload = res.get("metrics_payload", {})


def _normalized_overall_scores(models, accuracy_vals, latency_vals, energy_vals, w_acc, w_lat, w_eng):
    total_w = w_acc + w_lat + w_eng
    if total_w <= 0:
        w_acc, w_lat, w_eng = 1/3, 1/3, 1/3
    else:
        w_acc, w_lat, w_eng = w_acc / total_w, w_lat / total_w, w_eng / total_w

    acc_arr = np.array(accuracy_vals, dtype=float)
    lat_arr = np.array(latency_vals, dtype=float)
    eng_arr = np.array(energy_vals, dtype=float)

    acc_n = acc_arr / max(acc_arr.max(), 1e-9)
    lat_n = 1 - (lat_arr / max(lat_arr.max(), 1e-9))
    eng_n = 1 - (eng_arr / max(eng_arr.max(), 1e-9))

    scores = (w_acc * acc_n) + (w_lat * lat_n) + (w_eng * eng_n)
    return pd.DataFrame({
        "Model": models,
        "Accuracy (norm)": np.round(acc_n, 4),
        "Latency (norm, inverted)": np.round(lat_n, 4),
        "Energy (norm, inverted)": np.round(eng_n, 4),
        "Overall Score": np.round(scores, 4),
    }).sort_values("Overall Score", ascending=False)

# ── Top metric cards ───────────────────────────────────────────────────────────
st.subheader("📊 Final Metric Comparison")
c1, c2, c3 = st.columns(3)

def delta_str(fed, base, higher_better=True):
    d = fed - base
    better = (d > 0) if higher_better else (d < 0)
    status = "better" if better else "worse"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.1f} vs centralized ({status})"

c1.metric("Accuracy (%)",
    f"{fed_acc:.1f}%",
    delta_str(fed_acc, cen["accuracy"], higher_better=True))
c2.metric("Avg Latency (ms)",
    f"{fed_lat:.1f} ms",
    delta_str(fed_lat, cen["latency_ms"], higher_better=False),
    delta_color="inverse")
c3.metric("Avg Energy (J)",
    f"{fed_eng:.2f} J",
    delta_str(fed_eng, cen["energy_j"], higher_better=False),
    delta_color="inverse")

st.divider()

# ── Bar chart comparison ───────────────────────────────────────────────────────
if sym:
    st.subheader("📈 Side-by-Side Comparison: All Three Models")
    methods  = ["Centralized", "FedAvg (Simple)", "SYMBIOTIC-TWIN"]
    accuracy = [cen["accuracy"], fed_acc, sym["accuracy"]]
    latency  = [cen["latency_ms"], fed_lat, sym["latency_ms"]]
    energy   = [cen["energy_j"], fed_eng, sym["energy_j"]]
    st.caption(f"SYMBIOTIC-TWIN values loaded from: {sym.get('source', 'main/logs/metrics.json')}")
else:
    st.subheader("📈 Side-by-Side Comparison: Baselines (Symbiotic not available)")
    methods  = ["Centralized", "FedAvg (Simple)"]
    accuracy = [cen["accuracy"], fed_acc]
    latency  = [cen["latency_ms"], fed_lat]
    energy   = [cen["energy_j"], fed_eng]
    st.warning(
        "SYMBIOTIC-TWIN metrics not found. Run the main SYMBIOTIC-TWIN training first "
        "to include it in the comparison."
    )
colors   = ["#5b7fa6", "#4a9e7f", "#3266ad"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor("white")

for ax, vals, title, unit, higher_better in zip(
    axes,
    [accuracy, latency, energy],
    ["Accuracy", "Avg Latency", "Avg Energy"],
    ["%", "ms", "J"],
    [True, False, False]
):
    bars = ax.bar(methods, vals, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=9, color="#333")
    ax.set_title(f"{title} ({unit})", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, max(vals) * 1.2)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fafafa")

plt.tight_layout()
st.pyplot(fig)

st.divider()

score_df = _normalized_overall_scores(methods, accuracy, latency, energy, w_acc, w_lat, w_eng)
winner = score_df.iloc[0]["Model"]
winner_score = score_df.iloc[0]["Overall Score"]
st.subheader("🏆 Overall Balanced Ranking (Normalized)")
st.caption(
    "Overall score uses normalized metrics with higher-is-better logic "
    "(latency/energy are inverted). Adjust weights in sidebar."
)
st.success(f"Best overall model: **{winner}** (score: {winner_score:.4f})")
st.dataframe(score_df, use_container_width=True, hide_index=True)

st.divider()

# ── Convergence line chart ─────────────────────────────────────────────────────
st.subheader("📉 FedAvg Convergence Over Rounds")

df = pd.DataFrame(res["round_metrics"])

fig2, axes2 = plt.subplots(1, 3, figsize=(14, 3.5))
fig2.patch.set_facecolor("white")

plots = [
    ("accuracy",   "Accuracy (%)",   "#4a9e7f"),
    ("latency_ms", "Latency (ms)",   "#e07b3f"),
    ("energy_j",   "Energy (J)",     "#8b5cf6"),
]

for ax, (col, label, color) in zip(axes2, plots):
    ax.plot(df["round"], df[col], color=color, linewidth=2, marker="o", markersize=3)
    ax.fill_between(df["round"], df[col], alpha=0.1, color=color)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Round", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#fafafa")
    ax.set_xlim(1, len(df))

plt.tight_layout()
st.pyplot(fig2)

st.divider()

# ── Full comparison table ──────────────────────────────────────────────────────
if metrics_payload:
    st.subheader("Extra Comparison Metrics")

    rounds = metrics_payload.get("rounds", [])

    edge_acc_df = pd.DataFrame(metrics_payload.get("edge_acc", {}), index=rounds)
    if not edge_acc_df.empty:
        st.markdown("Client Accuracy Over Rounds")
        st.line_chart(edge_acc_df)

    agg_weights_df = pd.DataFrame(metrics_payload.get("agg_weights", []), index=rounds)
    if not agg_weights_df.empty:
        st.markdown("Aggregation Weights")
        st.line_chart(agg_weights_df)

    reputation_df = pd.DataFrame(metrics_payload.get("reputation", {}), index=rounds)
    if not reputation_df.empty:
        st.markdown("Reputation Scores")
        st.line_chart(reputation_df)

    st.divider()

st.subheader("📋 Summary Table")

table_data = {
    "Model":        ["Centralized", "FedAvg (Simple)"],
    "Accuracy (%)": [cen["accuracy"], round(fed_acc, 2)],
    "Latency (ms)": [cen["latency_ms"], round(fed_lat, 1)],
    "Energy (J)":   [cen["energy_j"], round(fed_eng, 2)],
    "Privacy":      ["❌ No", "✅ Yes (FL)"],
    "Adaptive":     ["❌ No", "❌ No"],
}
if sym:
    table_data["Model"].append("SYMBIOTIC-TWIN")
    table_data["Accuracy (%)"].append(sym["accuracy"])
    table_data["Latency (ms)"].append(sym["latency_ms"])
    table_data["Energy (J)"].append(sym["energy_j"])
    table_data["Privacy"].append("✅ Yes (FL+CDT)")
    table_data["Adaptive"].append("✅ Yes (Cognitive)")
st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ── Round detail table ─────────────────────────────────────────────────────────
with st.expander("🔍 Round-by-Round FedAvg Details"):
    st.dataframe(df.rename(columns={
        "round": "Round", "accuracy": "Accuracy (%)",
        "latency_ms": "Latency (ms)", "energy_j": "Energy (J)"
    }), use_container_width=True, hide_index=True)

st.caption(f"Canonical shared preprocessed CSV: {res['baseline_data_path']}")
