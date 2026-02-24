"""
Optimized SYMBIOTIC-TWIN Streamlit Dashboard
Production-ready version
"""

import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

METRICS_PATH = Path("logs/metrics.json")

EDGE_COLORS = {
    "edge1": "#4F8EF7",
    "edge2": "#F76B6B",
    "edge3": "#4CD964",
}

EDGE_LABELS = {
    "edge1": "Edge 1",
    "edge2": "Edge 2",
    "edge3": "Edge 3",
}

st.set_page_config(
    page_title="SYMBIOTIC-TWIN Dashboard",
    layout="wide",
    page_icon="🤖"
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.title("SYMBIOTIC-TWIN")

auto_refresh = st.sidebar.toggle("Auto refresh", True)
refresh_sec = st.sidebar.slider("Refresh interval", 2, 30, 3)

if auto_refresh:
    st.sidebar.caption(f"Refreshing every {refresh_sec}s")

# ─────────────────────────────────────────────
# DATA LOADER (CACHED)
# ─────────────────────────────────────────────

@st.cache_data(ttl=2)
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except:
        return {}

data = load_metrics()

# ─────────────────────────────────────────────
# NO DATA CASE
# ─────────────────────────────────────────────

if not data or not data.get("rounds"):
    st.info("Waiting for federated training...")
    st.stop()

# ─────────────────────────────────────────────
# EXTRACT DATA
# ─────────────────────────────────────────────

rounds = data["rounds"]
global_acc = data["global_acc"]
edge_acc = data["edge_acc"]
latency = data["latency"]
energy = data["energy"]
agg_weights = data["agg_weights"]
reputation = data["reputation"]

edge_ids = list(edge_acc.keys())

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("SYMBIOTIC-TWIN Federated Dashboard")

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────

col = st.columns(len(edge_ids) + 1)

col[0].metric(
    "Global Accuracy",
    f"{global_acc[-1]:.2f}%"
)

for i, eid in enumerate(edge_ids):
    col[i+1].metric(
        EDGE_LABELS[eid],
        f"{edge_acc[eid][-1]:.2f}%"
    )

# ─────────────────────────────────────────────
# CHART ROW 1
# ─────────────────────────────────────────────

left, right = st.columns(2)

# PIE CHART
with left:

    latest_weights = agg_weights[-1]

    fig_pie = go.Figure(data=[
        go.Pie(
            labels=list(latest_weights.keys()),
            values=list(latest_weights.values()),
            hole=0.4
        )
    ])

    fig_pie.update_layout(title="Aggregation Contribution")

    st.plotly_chart(
        fig_pie,
        width="stretch",
        key="pie"
    )

# ACCURACY CHART
with right:

    fig_acc = go.Figure()

    fig_acc.add_trace(go.Scatter(
        x=rounds,
        y=global_acc,
        name="Global"
    ))

    for eid in edge_ids:

        fig_acc.add_trace(go.Scatter(
            x=rounds[:len(edge_acc[eid])],
            y=edge_acc[eid],
            name=eid
        ))

    fig_acc.update_layout(title="Accuracy Over Rounds")

    st.plotly_chart(
        fig_acc,
        width="stretch",
        key="acc"
    )

# ─────────────────────────────────────────────
# LATENCY + ENERGY
# ─────────────────────────────────────────────

left, right = st.columns(2)

# LATENCY
with left:

    fig_lat = go.Figure()

    for eid in edge_ids:

        fig_lat.add_trace(go.Scatter(
            x=rounds[:len(latency[eid])],
            y=latency[eid],
            name=eid
        ))

    fig_lat.update_layout(title="Latency")

    st.plotly_chart(
        fig_lat,
        width="stretch",
        key="lat"
    )

# ENERGY
with right:

    fig_eng = go.Figure()

    for eid in edge_ids:

        fig_eng.add_trace(go.Bar(
            x=rounds[:len(energy[eid])],
            y=energy[eid],
            name=eid
        ))

    fig_eng.update_layout(title="Energy")

    st.plotly_chart(
        fig_eng,
        width="stretch",
        key="eng"
    )

# ─────────────────────────────────────────────
# REPUTATION
# ─────────────────────────────────────────────

fig_rep = go.Figure()

for eid in edge_ids:

    fig_rep.add_trace(go.Scatter(
        x=rounds[:len(reputation[eid])],
        y=reputation[eid],
        name=eid
    ))

fig_rep.update_layout(title="Reputation")

st.plotly_chart(
    fig_rep,
    width="stretch",
    key="rep"
)

# ─────────────────────────────────────────────
# TABLE
# ─────────────────────────────────────────────

table = []

for eid in edge_ids:

    table.append({
        "Edge": eid,
        "Accuracy": edge_acc[eid][-1],
        "Latency": latency[eid][-1],
        "Energy": energy[eid][-1],
        "Reputation": reputation[eid][-1]
    })

st.dataframe(pd.DataFrame(table), width="stretch")

# ─────────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────────

if auto_refresh:
    st.rerun()