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
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SYMBIOTIC-TWIN · Dashboard",
    page_icon="◇",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRICS_PATH = Path("logs/metrics.json")
REFRESH_INTERVAL = 3  # seconds

# ── Modern colour palette (distinct, accessible) ───────────────────────────────
EDGE_COLORS = {
    "edge1": "#0EA5E9",   # sky
    "edge2": "#F59E0B",   # amber
    "edge3": "#8B5CF6",   # violet
}

EDGE_LABELS = {
    "edge1": "Edge 1",
    "edge2": "Edge 2",
    "edge3": "Edge 3",
}

# Base chart style only (no xaxis/yaxis/legend/margin – set per chart to avoid duplicate kwargs)
CHART_BASE = dict(
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,0.8)",
    hovermode="x unified",
)

# Axes/legend used by most XY charts (bar/line)
AXIS_GRID = dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
LEGEND_BELOW = dict(
    orientation="h",
    yanchor="top",
    y=-0.18,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(255,255,255,0.8)",
)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def latest_weights(data: dict) -> dict:
    weights = data.get("agg_weights", [])
    return weights[-1] if weights else {}


def edge_display(eid: str) -> str:
    return EDGE_LABELS.get(eid, eid)


def color(eid: str) -> str:
    return EDGE_COLORS.get(eid, "#64748B")


def with_alpha(color_str: str, alpha: float) -> str:
    """
    Convert a hex color like '#8B5CF6' to an rgba() string with the given alpha.
    If the format is not hex, return the original string.
    """
    if isinstance(color_str, str) and color_str.startswith("#") and len(color_str) == 7:
        try:
            r = int(color_str[1:3], 16)
            g = int(color_str[3:5], 16)
            b = int(color_str[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        except ValueError:
            return color_str
    return color_str


# ── Custom styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header block */
    .stApp [data-testid="stHeader"] { background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); }
    /* Tighter spacing for metrics */
    div[data-testid="stMetric"] { padding: 0.6rem 0.5rem; }
    div[data-testid="stMetric"] label { font-size: 0.85rem; opacity: 0.9; }
    div[data-testid="stMetric"] value { font-size: 1.5rem; font-weight: 700; }
    /* Section spacing */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    /* Sidebar branding */
    [data-testid="stSidebar"] .stMarkdown { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ◇ SYMBIOTIC-TWIN")
    st.caption("Federated Multi-Agent Digital Twin")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh", value=True, help="Refresh metrics automatically")
    refresh_sec = st.slider("Refresh every (s)", 2, 30, REFRESH_INTERVAL)
    st.divider()
    st.markdown("**Nodes**")
    for eid, label in EDGE_LABELS.items():
        st.markdown(
            f"<span style='color:{color(eid)}; font-weight:600;'>● {label}</span>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("Dashboard reads from `logs/metrics.json`")

# ── Main layout ──────────────────────────────────────────────────────────────
st.title("Federated Learning Dashboard")
st.caption("Real-time model accuracy, aggregation weights, latency, and energy across edge nodes.")
st.divider()

placeholder = st.empty()

while True:
    data = load_metrics()

    with placeholder.container():
        if not data or not data.get("rounds"):
            st.markdown("---")
            st.markdown(
                "<p style='text-align:center; color:#64748b; font-size:1.1rem;'>"
                "◇ Waiting for federated training to start…<br>"
                "<small>Ensure the server and edge containers are running and completing at least one round.</small></p>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            time.sleep(refresh_sec)
            continue

        rounds = data.get("rounds", [])
        global_acc = data.get("global_acc", [])
        edge_acc = data.get("edge_acc", {})
        latency = data.get("latency", {})
        energy = data.get("energy", {})
        agg_weights = data.get("agg_weights", [])
        reputation = data.get("reputation", {})
        edge_ids = list(edge_acc.keys())

        current_round = rounds[-1] if rounds else 0
        ts = int(time.time() * 1000)
        current_g_acc = global_acc[-1] if global_acc else 0
        latest_weights_dict = agg_weights[-1] if agg_weights else {}

        # ── KPI row + round progress ──────────────────────────────────────────
        st.subheader("Live overview")
        cols = st.columns(len(edge_ids) + 2)

        delta_global = None
        if len(global_acc) > 1:
            d = current_g_acc - global_acc[-2]
            delta_global = f"{d:+.2f}%" if d != 0 else "0.00%"

        cols[0].metric("Global accuracy", f"{current_g_acc:.2f}%", delta_global)
        cols[1].metric("Round", str(current_round), None)

        for i, eid in enumerate(edge_ids):
            acc_list = edge_acc.get(eid, [])
            acc_now = acc_list[-1] if acc_list else 0
            acc_prev = acc_list[-2] if len(acc_list) > 1 else None
            delta = f"{acc_now - acc_prev:+.2f}%" if acc_prev is not None else None
            cols[i + 2].metric(edge_display(eid), f"{acc_now:.2f}%", delta)

        # Round progress (parse num_rounds from config without PyYAML)
        num_rounds = 10
        try:
            config_path = Path("config/config.yaml")
            if config_path.exists():
                text = config_path.read_text()
                for line in text.splitlines():
                    if "num_rounds" in line and ":" in line:
                        parts = line.split(":")[-1].strip()
                        if parts.isdigit():
                            num_rounds = int(parts)
                        break
        except Exception:
            pass
        pct = min(1.0, current_round / num_rounds) if num_rounds else 0
        st.progress(pct)
        st.caption(f"Round {current_round} of {num_rounds}")

        st.divider()

        # ── Aggregation + Accuracy (main two panels) ───────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Aggregation contributions")
            if latest_weights_dict:
                weight_df = pd.DataFrame([
                    {
                        "Node": edge_display(eid),
                        "Weight": round(w, 4),
                        "Share (%)": round(w * 100, 2),
                    }
                    for eid, w in latest_weights_dict.items()
                ])
                st.dataframe(weight_df, hide_index=True)

                fig_pie = go.Figure(go.Pie(
                    labels=[edge_display(eid) for eid in latest_weights_dict],
                    values=list(latest_weights_dict.values()),
                    marker=dict(colors=[color(eid) for eid in latest_weights_dict]),
                    hole=0.52,
                    textinfo="percent+label",
                    textposition="outside",
                    pull=[0.02] * len(latest_weights_dict),
                ))
                fig_pie.update_layout(
                    **CHART_BASE,
                    title=dict(text="Contribution share", font=dict(size=16)),
                    showlegend=False,
                    height=320,
                    margin=dict(t=56, b=32, l=32, r=32),
                    uniformtext=dict(minsize=11, mode="hide"),
                )
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{current_round}_{ts}")

        with right:
            st.subheader("Accuracy over rounds")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=rounds,
                y=global_acc,
                name="Global",
                line=dict(color="#0f172a", width=2.5, dash="dot"),
                mode="lines+markers",
                marker=dict(size=6),
            ))
            for eid in edge_ids:
                acc_list = edge_acc.get(eid, [])
                fig_acc.add_trace(go.Scatter(
                    x=rounds[: len(acc_list)],
                    y=acc_list,
                    name=edge_display(eid),
                    line=dict(color=color(eid), width=2),
                    mode="lines+markers",
                    marker=dict(size=5),
                ))
            fig_acc.update_layout(
                **CHART_BASE,
                title=dict(text="Accuracy by round", font=dict(size=16)),
                margin=dict(t=48, b=48, l=52, r=24),
                xaxis=dict(title="Round", **AXIS_GRID),
                yaxis=dict(title="Accuracy (%)", range=[0, 105], **AXIS_GRID),
                legend=LEGEND_BELOW,
                height=380,
            )
            st.plotly_chart(fig_acc, use_container_width=True, key=f"acc_{current_round}_{ts}")

        st.divider()

        # ── Weights over rounds ───────────────────────────────────────────────
        st.subheader("Aggregation weights over rounds")
        if len(agg_weights) > 1:
            fig_w = go.Figure()
            for eid in edge_ids:
                w_series = [rnd.get(eid, 0) for rnd in agg_weights]
                fig_w.add_trace(go.Bar(
                    name=edge_display(eid),
                    x=rounds[: len(w_series)],
                    y=w_series,
                    marker_color=color(eid),
                    marker_line=dict(width=0),
                ))
            fig_w.update_layout(
                **CHART_BASE,
                barmode="stack",
                margin=dict(t=48, b=48, l=52, r=24),
                xaxis=dict(title="Round", **AXIS_GRID),
                yaxis=dict(title="Weight", **AXIS_GRID),
                legend=LEGEND_BELOW,
                height=300,
            )
            st.plotly_chart(fig_w, use_container_width=True, key=f"weights_{current_round}_{ts}")
        else:
            st.caption("Weights will appear after multiple rounds.")

        st.divider()

        # ── Latency & Energy ─────────────────────────────────────────────────
        col_lat, col_eng = st.columns(2)
        with col_lat:
            st.subheader("Inference latency (ms)")
            fig_lat = go.Figure()
            for eid in edge_ids:
                lat_list = latency.get(eid, [])
                fig_lat.add_trace(go.Scatter(
                    x=rounds[: len(lat_list)],
                    y=lat_list,
                    name=edge_display(eid),
                    line=dict(color=color(eid), width=2),
                    mode="lines+markers",
                    marker=dict(size=5),
                ))
            fig_lat.update_layout(
                **CHART_BASE,
                margin=dict(t=48, b=48, l=52, r=24),
                xaxis=dict(title="Round", **AXIS_GRID),
                yaxis=dict(title="Latency (ms)", **AXIS_GRID),
                legend=LEGEND_BELOW,
                height=300,
            )
            st.plotly_chart(fig_lat, use_container_width=True, key=f"lat_{current_round}_{ts}")

        with col_eng:
            st.subheader("Energy (J)")
            fig_eng = go.Figure()
            for eid in edge_ids:
                eng_list = energy.get(eid, [])
                fig_eng.add_trace(go.Bar(
                    name=edge_display(eid),
                    x=rounds[: len(eng_list)],
                    y=eng_list,
                    marker_color=color(eid),
                    marker_line=dict(width=0),
                ))
            fig_eng.update_layout(
                **CHART_BASE,
                barmode="group",
                margin=dict(t=48, b=48, l=52, r=24),
                xaxis=dict(title="Round", **AXIS_GRID),
                yaxis=dict(title="Energy (J)", **AXIS_GRID),
                legend=LEGEND_BELOW,
                height=300,
            )
            st.plotly_chart(fig_eng, use_container_width=True, key=f"eng_{current_round}_{ts}")

        st.divider()

        # ── Reputation (expandable) ───────────────────────────────────────────
        if reputation:
            with st.expander("Edge reputation / trust scores", expanded=False):
                fig_rep = go.Figure()
                for eid in edge_ids:
                    rep_list = reputation.get(eid, [])
                    c = color(eid)
                    fig_rep.add_trace(go.Scatter(
                        x=rounds[: len(rep_list)],
                        y=rep_list,
                        name=edge_display(eid),
                        line=dict(color=c, width=2),
                        mode="lines+markers",
                        fill="tozeroy",
                        fillcolor=with_alpha(c, 0.13),
                    ))
                fig_rep.update_layout(
                    **CHART_BASE,
                    margin=dict(t=48, b=48, l=52, r=24),
                    xaxis=dict(title="Round", **AXIS_GRID),
                    yaxis=dict(title="Reputation", range=[0, 1.1], **AXIS_GRID),
                    legend=LEGEND_BELOW,
                    height=280,
                )
                st.plotly_chart(fig_rep, use_container_width=True, key=f"rep_{current_round}_{ts}")

        # ── Node health table ─────────────────────────────────────────────────
        st.subheader("Node status")
        health_rows = []
        for eid in edge_ids:
            acc_list = edge_acc.get(eid, [])
            lat_list = latency.get(eid, [])
            eng_list = energy.get(eid, [])
            rep_list = reputation.get(eid, [])
            health_rows.append({
                "Node": edge_display(eid),
                "Rounds": len(acc_list),
                "Accuracy %": f"{acc_list[-1]:.2f}" if acc_list else "—",
                "Latency ms": f"{lat_list[-1]:.1f}" if lat_list else "—",
                "Energy J": f"{eng_list[-1]:.4f}" if eng_list else "—",
                "Rep.": f"{rep_list[-1]:.3f}" if rep_list else "—",
                "Status": "Active" if (acc_list and len(acc_list) >= current_round) else "Stalled",
            })
        st.dataframe(pd.DataFrame(health_rows), hide_index=True)

        st.caption(f"Round {current_round} · Auto-refresh {'on' if auto_refresh else 'off'}")

    if not auto_refresh:
        st.stop()

    time.sleep(refresh_sec)
    placeholder.empty()
