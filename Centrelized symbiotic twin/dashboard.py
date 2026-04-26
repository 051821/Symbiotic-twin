"""
Simple Centralized Training Dashboard
Shows:
  - Accuracy
  - Latency
  - Energy
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Centralized Training Dashboard", layout="wide")

st.title("Centralized Learning Dashboard")

metrics_path = Path("Centrelized symbiotic twin\metrics.csv")
metrics_json_path = Path("C:\Symbiotic-twin\Centrelized symbiotic twin\metrics.json")
if not metrics_path.exists():
    st.error("metrics.csv not found. Run training first.")
    st.stop()

df = pd.read_csv(metrics_path)
if df.empty:
    st.warning("No metrics available.")
    st.stop()

latest = df.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{latest['accuracy'] * 100:.2f}%")
col2.metric("Latency (ms)", f"{latest['latency']:.2f}")
col3.metric("Energy (J)", f"{latest['energy']:.6f}")

st.divider()

st.subheader("Accuracy Over Epochs")
fig1, ax1 = plt.subplots()
ax1.plot(df["round"], df["accuracy"], marker="o")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.grid(True)
st.pyplot(fig1)

st.subheader("Latency Over Epochs")
fig2, ax2 = plt.subplots()
ax2.plot(df["round"], df["latency"], marker="o")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Latency (ms)")
ax2.grid(True)
st.pyplot(fig2)

st.subheader("Energy Consumption Over Epochs")
fig3, ax3 = plt.subplots()
ax3.plot(df["round"], df["energy"], marker="o")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Energy (J)")
ax3.grid(True)
st.pyplot(fig3)

st.success("Dashboard Loaded Successfully")
if metrics_json_path.exists():
    metrics_payload = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    rounds = metrics_payload.get("rounds", [])

    st.divider()
    st.subheader("Baseline Comparison Metrics")

    agg_weights_df = pd.DataFrame(metrics_payload.get("agg_weights", []), index=rounds)
    if not agg_weights_df.empty:
        st.markdown("Aggregation Weights")
        st.line_chart(agg_weights_df)

    reputation_df = pd.DataFrame(metrics_payload.get("reputation", {}), index=rounds)
    if not reputation_df.empty:
        st.markdown("Reputation Scores")
        st.line_chart(reputation_df)

