"""
dashboard/app.py
Streamlit real-time dashboard for the SYMBIOTIC-TWIN federated learning system.

Displays:
  - Edge node names, sample counts, and aggregation contributions
  - Model accuracy per edge and global
  - Inference latency per edge
  - Energy consumption per edge
  - Reputation / trust scores
  - Federated round progress
"""

import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "SYMBIOTIC-TWIN Dashboard",
    page_icon   = "🤖",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

METRICS_PATH = Path("logs/metrics.json")
REFRESH_INTERVAL = 3   # seconds

# ── Colour palette per edge ──────────────────────────────────────────────────
EDGE_COLORS = {
    "edge1": "#4F8EF7",
    "edge2": "#F76B6B",
    "edge3": "#4CD964",
}

EDGE_LABELS = {
    "edge1": "Edge 1 (b8:27:eb:bf:9d:51)",
    "edge2": "Edge 2 (00:0f:00:70:91:0a)",
    "edge3": "Edge 3 (1c:bf:ce:15:ec:4d)",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def latest_weights(data: dict) -> dict:
    """Return the most recent aggregation weight dict."""
    weights = data.get("agg_weights", [])
    return weights[-1] if weights else {}


def edge_display(eid: str) -> str:
    return EDGE_LABELS.get(eid, eid)


def color(eid: str) -> str:
    return EDGE_COLORS.get(eid, "#AAAAAA")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/4F8EF7/artificial-intelligence.png", width=80)
    st.title("SYMBIOTIC-TWIN")
    st.caption("Federated Multi-Agent Digital Twin Framework")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_sec  = st.slider("Refresh interval (s)", 2, 30, REFRESH_INTERVAL)
    st.divider()
    st.markdown("**Legend**")
    for eid, label in EDGE_LABELS.items():
        st.markdown(
            f"<span style='color:{color(eid)}; font-size:14px'>● {label}</span>",
            unsafe_allow_html=True,
        )


# ── Main layout ──────────────────────────────────────────────────────────────

st.title("🤖 SYMBIOTIC-TWIN · Federated Learning Dashboard")
st.caption("Real-time monitoring of federated edge nodes, model performance, and aggregation.")
st.divider()

placeholder = st.empty()

while True:
    data = load_metrics()

    with placeholder.container():

        # ── No data yet ──────────────────────────────────────────────────────
        if not data or not data.get("rounds"):
            st.info("⏳ Waiting for federated training to begin...")
            time.sleep(refresh_sec)
            continue

        rounds      = data.get("rounds", [])
        global_acc  = data.get("global_acc", [])
        edge_acc    = data.get("edge_acc", {})
        latency     = data.get("latency", {})
        energy      = data.get("energy", {})
        agg_weights = data.get("agg_weights", [])
        reputation  = data.get("reputation", {})
        edge_ids    = list(edge_acc.keys())

        current_round  = rounds[-1] if rounds else 0
        current_g_acc  = global_acc[-1] if global_acc else 0
        latest_weights_dict = agg_weights[-1] if agg_weights else {}

        # ── KPI cards ────────────────────────────────────────────────────────
        st.subheader("📊 Live Overview")
        cols = st.columns(len(edge_ids) + 2)

        cols[0].metric(
            label = "🌐 Global Accuracy",
            value = f"{current_g_acc:.2f}%",
            delta = f"{current_g_acc - global_acc[-2]:.2f}%" if len(global_acc) > 1 else None,
        )
        cols[1].metric(
            label = "🔄 Current Round",
            value = f"{current_round}",
        )
        for i, eid in enumerate(edge_ids):
            acc_list = edge_acc.get(eid, [])
            acc_now  = acc_list[-1] if acc_list else 0
            acc_prev = acc_list[-2] if len(acc_list) > 1 else None
            cols[i + 2].metric(
                label = f"{edge_display(eid)}",
                value = f"{acc_now:.2f}%",
                delta = f"{acc_now - acc_prev:.2f}%" if acc_prev is not None else None,
            )

        st.divider()

        # ── Aggregation weights ──────────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.subheader("⚖️ Aggregation Contributions")

            if latest_weights_dict:
                weight_df = pd.DataFrame([
                    {
                        "Edge":            edge_display(eid),
                        "Weight":          round(w, 4),
                        "Contribution %":  round(w * 100, 2),
                    }
                    for eid, w in latest_weights_dict.items()
                ])
                st.dataframe(weight_df, use_container_width=True, hide_index=True)

                fig_pie = go.Figure(go.Pie(
                    labels  = [edge_display(eid) for eid in latest_weights_dict],
                    values  = list(latest_weights_dict.values()),
                    marker  = dict(colors=[color(eid) for eid in latest_weights_dict]),
                    hole    = 0.4,
                    textinfo= "percent+label",
                ))
                fig_pie.update_layout(
                    title       = "Edge Contribution Share",
                    showlegend  = False,
                    height      = 320,
                    margin      = dict(t=40, b=10, l=10, r=10),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with right:
            st.subheader("📈 Accuracy Over Rounds")

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x    = rounds,
                y    = global_acc,
                name = "Global",
                line = dict(color="#FFD700", width=3, dash="dot"),
                mode = "lines+markers",
            ))
            for eid in edge_ids:
                acc_list = edge_acc.get(eid, [])
                fig_acc.add_trace(go.Scatter(
                    x    = rounds[:len(acc_list)],
                    y    = acc_list,
                    name = edge_display(eid),
                    line = dict(color=color(eid), width=2),
                    mode = "lines+markers",
                ))
            fig_acc.update_layout(
                xaxis_title = "Round",
                yaxis_title = "Accuracy (%)",
                yaxis       = dict(range=[0, 105]),
                height      = 380,
                legend      = dict(orientation="h", y=-0.2),
                margin      = dict(t=10, b=10),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        st.divider()

        # ── Aggregation weights over rounds ──────────────────────────────────
        st.subheader("📉 Aggregation Weights Over Rounds")
        if len(agg_weights) > 1:
            fig_w = go.Figure()
            for eid in edge_ids:
                w_series = [rnd.get(eid, 0) for rnd in agg_weights]
                fig_w.add_trace(go.Bar(
                    name = edge_display(eid),
                    x    = rounds[:len(w_series)],
                    y    = w_series,
                    marker_color = color(eid),
                ))
            fig_w.update_layout(
                barmode     = "stack",
                xaxis_title = "Round",
                yaxis_title = "Weight",
                height      = 300,
                legend      = dict(orientation="h", y=-0.25),
                margin      = dict(t=10, b=10),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        st.divider()

        # ── Latency and Energy ───────────────────────────────────────────────
        col_lat, col_eng = st.columns(2)

        with col_lat:
            st.subheader("⏱ Inference Latency (ms)")
            fig_lat = go.Figure()
            for eid in edge_ids:
                lat_list = latency.get(eid, [])
                fig_lat.add_trace(go.Scatter(
                    x    = rounds[:len(lat_list)],
                    y    = lat_list,
                    name = edge_display(eid),
                    line = dict(color=color(eid), width=2),
                    mode = "lines+markers",
                ))
            fig_lat.update_layout(
                xaxis_title = "Round",
                yaxis_title = "Latency (ms)",
                height      = 300,
                legend      = dict(orientation="h", y=-0.3),
                margin      = dict(t=10, b=10),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_eng:
            st.subheader("⚡ Energy Consumption (J)")
            fig_eng = go.Figure()
            for eid in edge_ids:
                eng_list = energy.get(eid, [])
                fig_eng.add_trace(go.Bar(
                    name         = edge_display(eid),
                    x            = rounds[:len(eng_list)],
                    y            = eng_list,
                    marker_color = color(eid),
                ))
            fig_eng.update_layout(
                xaxis_title = "Round",
                yaxis_title = "Energy (J)",
                barmode     = "group",
                height      = 300,
                legend      = dict(orientation="h", y=-0.3),
                margin      = dict(t=10, b=10),
            )
            st.plotly_chart(fig_eng, use_container_width=True)

        st.divider()

        # ── Reputation scores ─────────────────────────────────────────────────
        if reputation:
            st.subheader("🔒 Edge Reputation / Trust Scores")
            fig_rep = go.Figure()
            for eid in edge_ids:
                rep_list = reputation.get(eid, [])
                fig_rep.add_trace(go.Scatter(
                    x    = rounds[:len(rep_list)],
                    y    = rep_list,
                    name = edge_display(eid),
                    line = dict(color=color(eid), width=2),
                    mode = "lines+markers",
                    fill = "tozeroy",
                    fillcolor = color(eid).replace(")", ", 0.1)").replace("rgb", "rgba") if "rgb" in color(eid) else None,
                ))
            fig_rep.update_layout(
                xaxis_title = "Round",
                yaxis_title = "Reputation Score",
                yaxis       = dict(range=[0, 1.1]),
                height      = 300,
                legend      = dict(orientation="h", y=-0.3),
                margin      = dict(t=10, b=10),
            )
            st.plotly_chart(fig_rep, use_container_width=True)
            st.divider()

        # ── Node health table ─────────────────────────────────────────────────
        st.subheader("🖥 Node Health Status")
        health_rows = []
        for eid in edge_ids:
            acc_list = edge_acc.get(eid, [])
            lat_list = latency.get(eid, [])
            eng_list = energy.get(eid, [])
            rep_list = reputation.get(eid, [])
            health_rows.append({
                "Node":            edge_display(eid),
                "Rounds":          len(acc_list),
                "Last Accuracy %": f"{acc_list[-1]:.2f}" if acc_list else "—",
                "Last Latency ms": f"{lat_list[-1]:.1f}" if lat_list else "—",
                "Last Energy J":   f"{eng_list[-1]:.4f}" if eng_list else "—",
                "Reputation":      f"{rep_list[-1]:.4f}" if rep_list else "—",
                "Status":          "🟢 Active" if len(acc_list) == current_round else "🔴 Stalled",
            })
        st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

        st.caption(f"Last updated: round {current_round} | Auto-refresh: {'ON' if auto_refresh else 'OFF'}")

    if not auto_refresh:
        st.stop()

    time.sleep(refresh_sec)
    placeholder.empty()
