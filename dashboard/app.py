"""
dashboard/app.py  — SYMBIOTIC-TWIN v2.0
Complete functional Streamlit dashboard with:
  Tab 1 — Live Classifier  (sensor sliders → real-time 3-class prediction)
  Tab 2 — Federated Metrics (accuracy, latency, energy, reputation, weights)
  Tab 3 — Agent Panel       (4 agents: Analyst, Anomaly, Predictor, Security)
  Tab 4 — Security Monitor  (event log, norm history, HMAC/poison status)
  Tab 5 — Window Visualizer (temporal sliding window explained visually)
"""

import json
import time
import random
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SYMBIOTIC-TWIN",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRICS_PATH = Path("logs/metrics.json")
SERVER_URL   = "http://server:8000"

EDGE_COLORS = {"edge1": "#0EA5E9", "edge2": "#F59E0B", "edge3": "#8B5CF6"}
EDGE_LABELS = {
    "edge1": "Edge 1 · b8:27:eb",
    "edge2": "Edge 2 · 00:0f:00",
    "edge3": "Edge 3 · 1c:bf:ce",
}
CLASS_NAMES  = ["Normal", "Warning", "Critical"]
CLASS_COLORS = {"Normal": "#22C55E", "Warning": "#F59E0B", "Critical": "#EF4444"}
CLASS_ICONS  = {"Normal": "✅",       "Warning": "⚠️",       "Critical": "🚨"}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background: #060910; }

.kpi {
    background: linear-gradient(135deg,#0d1117 0%,#161b22 100%);
    border: 1px solid #21262d; border-radius: 14px;
    padding: 18px 22px; text-align: center; position: relative; overflow: hidden;
}
.kpi::after {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg,#0EA5E9,#8B5CF6);
}
.kpi-val  { font-size:2.2rem; font-weight:800; color:#f0f6fc; line-height:1.1; }
.kpi-lbl  { font-size:0.7rem; color:#8b949e; margin-top:4px; letter-spacing:.1em; text-transform:uppercase; }
.kpi-d-up { color:#22C55E; font-size:.8rem; font-weight:600; margin-top:4px; }
.kpi-d-dn { color:#EF4444; font-size:.8rem; font-weight:600; margin-top:4px; }

.classify-box {
    border-radius:18px; padding:28px 36px; text-align:center;
    font-weight:800; font-size:2rem; border:2px solid; margin:10px 0;
    letter-spacing:.02em;
}
.cls-normal   { background:#052e16aa; color:#22C55E; border-color:#22C55E; }
.cls-warning  { background:#1c1007aa; color:#F59E0B; border-color:#F59E0B; }
.cls-critical { background:#1c0505aa; color:#EF4444; border-color:#EF4444;
                animation:pulse 1.2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.65} }

.prob-row { display:flex; align-items:center; gap:10px; margin:6px 0; }
.prob-lbl { width:70px; font-size:.8rem; color:#8b949e; }
.prob-bar { flex:1; height:9px; background:#21262d; border-radius:5px; overflow:hidden; }
.prob-fill{ height:100%; border-radius:5px; transition:width .4s ease; }
.prob-pct { width:42px; text-align:right; font-size:.8rem; font-weight:600; font-family:'JetBrains Mono'; }

.sec-row  { font-family:'JetBrains Mono',monospace; font-size:.75rem;
            padding:5px 10px; border-left:3px solid #21262d; color:#8b949e; margin:2px 0;
            background:#0d1117; border-radius:0 6px 6px 0; }
.sec-fail { border-left-color:#EF4444!important; color:#f87171; }
.sec-warn { border-left-color:#F59E0B!important; color:#fbbf24; }
.sec-ok   { border-left-color:#22C55E!important; color:#86efac; }
.sec-info { border-left-color:#0EA5E9!important; color:#7dd3fc; }

.agent-hdr{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.badge    { font-size:.7rem; font-weight:700; padding:2px 8px; border-radius:99px; }
.bdg-done { background:#052e16; color:#22C55E; }
.bdg-run  { background:#1c1007; color:#F59E0B; }
.bdg-err  { background:#1c0505; color:#EF4444; }

.alert-crit{ background:#1c0505; color:#f87171; border:1px solid #7f1d1d;
             border-radius:8px; padding:7px 14px; margin:4px 0; font-size:.82rem; }
.alert-warn{ background:#1c1007; color:#fbbf24; border:1px solid #78350f;
             border-radius:8px; padding:7px 14px; margin:4px 0; font-size:.82rem; }
.alert-info{ background:#0c1a2e; color:#7dd3fc; border:1px solid #0c4a6e;
             border-radius:8px; padding:7px 14px; margin:4px 0; font-size:.82rem; }

.win-box   { background:#0d1117; border:1px solid #21262d; border-radius:12px; padding:16px; }
.win-title { font-weight:700; color:#f0f6fc; margin-bottom:6px; }
.win-sub   { font-size:.8rem; color:#8b949e; }

.section-hdr { font-size:.75rem; font-weight:700; color:#8b949e; letter-spacing:.1em;
               text-transform:uppercase; border-bottom:1px solid #21262d;
               padding-bottom:8px; margin:18px 0 12px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8b949e", family="Syne"),
    margin=dict(t=20, b=30, l=40, r=10),
    legend=dict(orientation="h", y=-0.28, font=dict(size=11)),
)

_AXIS_STYLE = dict(gridcolor="#21262d", zerolinecolor="#30363d", color="#8b949e")


def _layout(**kwargs):
    """Merge PLOTLY_BASE with per-chart axis overrides without key conflicts."""
    # Merge axis style into any caller-supplied xaxis/yaxis dicts
    out = dict(**PLOTLY_BASE)
    xa = dict(**_AXIS_STYLE)
    ya = dict(**_AXIS_STYLE)
    if "xaxis" in kwargs:
        xa.update(kwargs.pop("xaxis"))
    if "yaxis" in kwargs:
        ya.update(kwargs.pop("yaxis"))
    out["xaxis"] = xa
    out["yaxis"] = ya
    out.update(kwargs)
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=2)
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def server_up() -> bool:
    try:
        return requests.get(f"{SERVER_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def api_get(path: str) -> dict:
    try:
        r = requests.get(f"{SERVER_URL}{path}", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def api_post(path: str, data: dict) -> dict:
    try:
        r = requests.post(f"{SERVER_URL}{path}", json=data, timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def classify_local(f: dict) -> dict:
    """Rule-based fallback classifier matching preprocess.py label logic exactly."""
    s, c, t, l = f["smoke"], f["co"], f["temp"], f["lpg"]
    if s > 0.10 or c > 0.005:
        pred, label = 2, "Critical"
        probs = [0.03, 0.07, 0.90]
        reasons = []
        if s > 0.10: reasons.append(f"Smoke {s:.3f} > threshold 0.10")
        if c > 0.005: reasons.append(f"CO {c:.4f} > threshold 0.005")
    elif t > 90.0 or l > 0.007:
        pred, label = 1, "Warning"
        probs = [0.15, 0.74, 0.11]
        reasons = []
        if t > 90.0: reasons.append(f"Temp {t:.1f}°F > threshold 90°F")
        if l > 0.007: reasons.append(f"LPG {l:.4f} > threshold 0.007")
    else:
        pred, label = 0, "Normal"
        sm = 1 - min(s / 0.10, 1.0)
        cm = 1 - min(c / 0.005, 1.0)
        conf = 0.55 + 0.40 * min(sm, cm)
        probs = [conf, (1-conf)*0.55, (1-conf)*0.45]
        reasons = ["All sensor readings within safe thresholds"]

    noise = [random.uniform(-0.015, 0.015) for _ in range(3)]
    probs = [max(0.01, min(0.99, p + n)) for p, n in zip(probs, noise)]
    s_p   = sum(probs)
    probs = [p / s_p for p in probs]

    return {
        "prediction":    pred,
        "label":         label,
        "confidence":    round(max(probs) * 100, 1),
        "probabilities": {CLASS_NAMES[i]: round(p * 100, 1) for i, p in enumerate(probs)},
        "model_version": "local-rules",
        "reasons":       reasons,
    }


def mock_metrics(n_rounds: int = 8) -> dict:
    data = {"rounds":[], "global_acc":[], "edge_acc":{"edge1":[],"edge2":[],"edge3":[]},
            "latency":{"edge1":[],"edge2":[],"edge3":[]},
            "energy":{"edge1":[],"edge2":[],"edge3":[]},
            "agg_weights":[], "reputation":{"edge1":[],"edge2":[],"edge3":[]}}
    total_n = 149960 + 89452 + 84734
    for r in range(1, n_rounds + 1):
        base = min(93, 58 + r * 2.2)
        accs = {"edge1": base + random.uniform(0,3), "edge2": base-random.uniform(0,2), "edge3": base+random.uniform(-1,2)}
        wacc = (accs["edge1"]*149960 + accs["edge2"]*89452 + accs["edge3"]*84734) / total_n
        data["rounds"].append(r)
        data["global_acc"].append(round(wacc, 2))
        for eid, v in accs.items():
            data["edge_acc"][eid].append(round(v, 2))
            prev_lat = (data["latency"][eid] or [14])[-1]
            data["latency"][eid].append(round(max(5, prev_lat + random.uniform(-.8,.4)), 2))
            prev_eng = (data["energy"][eid] or [.003])[-1]
            data["energy"][eid].append(round(max(.001, prev_eng + random.uniform(-.0002,.0003)), 5))
        w1,w2 = 149960/total_n, 89452/total_n
        data["agg_weights"].append({"edge1":round(w1,4),"edge2":round(w2,4),"edge3":round(84734/total_n,4)})
        for eid in ["edge1","edge2","edge3"]:
            prev = (data["reputation"][eid] or [.8])[-1]
            data["reputation"][eid].append(round(min(1.0,max(.1,prev+random.uniform(-.01,.02))),3))
    return data


def prob_bars_html(probs: dict) -> str:
    html = ""
    for cls, pct in probs.items():
        color = CLASS_COLORS[cls]
        html += (
            f'<div class="prob-row">'
            f'<div class="prob-lbl">{CLASS_ICONS[cls]} {cls}</div>'
            f'<div class="prob-bar"><div class="prob-fill" style="width:{pct}%;background:{color}"></div></div>'
            f'<div class="prob-pct" style="color:{color}">{pct:.1f}%</div>'
            f'</div>'
        )
    return html


def kpi_html(val, lbl, delta=None, color=None):
    col = color or "#f0f6fc"
    d_html = ""
    if delta is not None:
        dcls = "kpi-d-up" if delta >= 0 else "kpi-d-dn"
        sign = "+" if delta >= 0 else ""
        d_html = f'<div class="{dcls}">{sign}{delta:.2f}%</div>'
    return (
        f'<div class="kpi">'
        f'<div class="kpi-val" style="color:{col}">{val}</div>'
        f'<div class="kpi-lbl">{lbl}</div>{d_html}'
        f'</div>'
    )


def alert_html(alerts: list) -> str:
    if not alerts:
        return '<div class="alert-info">✅ No alerts this round — system nominal</div>'
    html = ""
    for a in alerts:
        cls = "alert-crit" if any(w in a for w in ["FIRE","CRITICAL","POISON","HMAC","🚨","🔐","🔑"]) \
              else "alert-warn" if any(w in a for w in ["⚠️","GAS","TEMP","📉","ℹ️"]) \
              else "alert-info"
        html += f'<div class="{cls}">{a}</div>'
    return html


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 SYMBIOTIC-TWIN")
    st.caption("Secure Multi-Agent Federated Digital Twin v2.0")
    st.divider()

    live_mode    = st.toggle("🔴 Live Mode (Server)", value=False)
    auto_refresh = st.toggle("🔄 Auto-refresh", value=True)
    refresh_sec  = st.slider("Interval (s)", 2, 15, 3)

    st.divider()
    st.markdown("**Edge Nodes**")
    for eid, lbl in EDGE_LABELS.items():
        c = EDGE_COLORS[eid]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{c}'></div>"
            f"<span style='font-size:12px;color:#8b949e'>{lbl}</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    if live_mode:
        alive = server_up()
        st.markdown(
            f"**Server:** {'🟢 Connected' if alive else '🔴 Unreachable'}",
            unsafe_allow_html=True,
        )
    else:
        st.info("Demo mode — using mock data")


# ── Title ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='display:flex;align-items:center;gap:14px;margin-bottom:6px'>"
    "<div style='font-size:2.4rem'>🤖</div>"
    "<div><div style='font-size:1.7rem;font-weight:800;color:#f0f6fc'>SYMBIOTIC-TWIN</div>"
    "<div style='font-size:.85rem;color:#6e7681'>Secure Multi-Agent Federated Learning · IoT Digital Twin Framework</div>"
    "</div></div>",
    unsafe_allow_html=True,
)
st.divider()

tabs = st.tabs(["🔬 Live Classifier", "📊 Federated Metrics", "🧠 Agent Panel", "🔐 Security", "📅 Window Visualizer"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown("### 🔬 Real-Time IoT Sensor Classification")
    st.caption(
        "The model was trained to classify 3 risk levels based on IoT sensor readings. "
        "Adjust the sliders to see how each sensor affects the prediction."
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="section-hdr">📡 Sensor Input</div>', unsafe_allow_html=True)

        # Presets
        p1, p2, p3, p4 = st.columns(4)
        preset = None
        if p1.button("🟢 Normal",   use_container_width=True): preset = "normal"
        if p2.button("⚠️ Warning",  use_container_width=True): preset = "warning"
        if p3.button("🚨 Critical", use_container_width=True): preset = "critical"
        if p4.button("🎲 Random",   use_container_width=True): preset = "random"

        presets = {
            "normal":   dict(co=0.002,  humidity=55.0, light=1.0, lpg=0.003, motion=0.0, smoke=0.02,  temp=72.0),
            "warning":  dict(co=0.003,  humidity=42.0, light=1.0, lpg=0.009, motion=1.0, smoke=0.06,  temp=95.0),
            "critical": dict(co=0.009,  humidity=28.0, light=0.0, lpg=0.013, motion=1.0, smoke=0.18,  temp=108.0),
            "random":   dict(co=random.uniform(0,.012), humidity=random.uniform(20,90),
                             light=float(random.randint(0,1)), lpg=random.uniform(0,.015),
                             motion=float(random.randint(0,1)), smoke=random.uniform(0,.22),
                             temp=random.uniform(55,125)),
        }

        dv = presets.get(preset, {})
        defaults = dict(co=0.002, humidity=55.0, light=1.0, lpg=0.003, motion=0.0, smoke=0.02, temp=72.0)
        for k, default in defaults.items():
            if k not in st.session_state or preset:
                st.session_state[k] = dv.get(k, default)

        co       = st.slider("💨 CO Concentration",   0.0, 0.015, st.session_state["co"],       0.0001, format="%.4f")
        humidity = st.slider("💧 Humidity (%)",         0.0, 100.0, st.session_state["humidity"], 0.5)
        light    = st.slider("💡 Light (0=off 1=on)",   0.0, 1.0,   st.session_state["light"],   1.0)
        lpg      = st.slider("⛽ LPG Concentration",   0.0, 0.020,  st.session_state["lpg"],     0.0001, format="%.4f")
        motion   = st.slider("🚶 Motion (0=no 1=yes)",  0.0, 1.0,   st.session_state["motion"],  1.0)
        smoke    = st.slider("🔥 Smoke Level",          0.0, 0.30,   st.session_state["smoke"],   0.001,  format="%.3f")
        temp     = st.slider("🌡️ Temperature (°F)",    50.0, 130.0,  st.session_state["temp"],    0.5)

        features = dict(co=co, humidity=humidity, light=light, lpg=lpg,
                        motion=motion, smoke=smoke, temp=temp)

        # Threshold indicators
        st.markdown('<div class="section-hdr">🎚️ Sensor vs Threshold</div>', unsafe_allow_html=True)
        thresholds = [
            ("smoke", smoke, 0.10, "Smoke"),
            ("co",    co,    0.005, "CO"),
            ("temp",  temp,  90.0, "Temp"),
            ("lpg",   lpg,   0.007, "LPG"),
        ]
        for _, val, thresh, name in thresholds:
            pct = min(val / thresh, 1.5) * 100
            bar_color = "#EF4444" if val > thresh else ("#F59E0B" if val > thresh * 0.7 else "#22C55E")
            pct_disp = min(pct, 100)
            st.markdown(
                f'<div style="margin:5px 0">'
                f'<div style="display:flex;justify-content:space-between;font-size:.78rem;color:#8b949e;margin-bottom:2px">'
                f'<span>{name}</span>'
                f'<span style="color:{bar_color};font-weight:600">{val:.4f} / {thresh}</span>'
                f'</div>'
                f'<div style="height:7px;background:#21262d;border-radius:4px;overflow:hidden">'
                f'<div style="height:100%;width:{pct_disp:.1f}%;background:{bar_color};border-radius:4px;transition:width .3s"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    with col_out:
        st.markdown('<div class="section-hdr">🎯 Classification Result</div>', unsafe_allow_html=True)

        result = None
        if live_mode and server_up():
            result = api_post("/classify", features)
        if not result:
            result = classify_local(features)

        label  = result["label"]
        conf   = result["confidence"]
        probs  = result["probabilities"]
        reasons = result["reasons"]

        st.markdown(
            f'<div class="classify-box cls-{label.lower()}">'
            f'{CLASS_ICONS[label]} {label.upper()}'
            f'<div style="font-size:1rem;font-weight:500;margin-top:6px">Confidence: {conf:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown(prob_bars_html(probs), unsafe_allow_html=True)

        st.markdown('<div class="section-hdr" style="margin-top:18px">💡 Why this prediction?</div>', unsafe_allow_html=True)
        for r in reasons:
            icon = "🔴" if label == "Critical" else ("🟡" if label == "Warning" else "🟢")
            st.markdown(f"{icon} {r}")

        with st.expander("📋 Classification Rules (3 classes)"):
            st.markdown("""
| Priority | Condition | Class |
|----------|-----------|-------|
| 1st checked | `smoke > 0.10` **OR** `co > 0.005` | 🚨 **Critical** |
| 2nd checked | `temp > 90°F` **OR** `lpg > 0.007` | ⚠️ **Warning** |
| Default | None of the above | ✅ **Normal** |

> **Why 3 classes matter:** Normal ≈ 85% of data, Warning ≈ 10%, Critical ≈ 5%.
> Class-weighted loss ensures the model learns all three, not just Normal.
""")

        mv = result.get("model_version", "?")
        st.caption(f"Model: {'Federated IoTClassifier' if mv != 'local-rules' else 'Local rule-based fallback'} · v{mv}")

    st.divider()

    # Batch simulator
    st.markdown("### 🧪 Batch Scenario Simulator")
    n_batch = st.slider("Number of random readings to simulate", 20, 200, 50, 10)
    if st.button(f"▶️ Run Simulation ({n_batch} readings)"):
        rows = []
        for _ in range(n_batch):
            f = dict(co=random.uniform(0,.012), humidity=random.uniform(20,90),
                     light=float(random.randint(0,1)), lpg=random.uniform(0,.015),
                     motion=float(random.randint(0,1)), smoke=random.uniform(0,.22),
                     temp=random.uniform(55,125))
            r = classify_local(f)
            rows.append({**f, "label": r["label"], "confidence": r["confidence"]})
        df = pd.DataFrame(rows)

        ca, cb, cc = st.columns(3)
        counts = df["label"].value_counts()
        with ca:
            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(),
                marker=dict(colors=[CLASS_COLORS.get(l,"#aaa") for l in counts.index]),
                hole=0.45, textinfo="percent+label",
            ))
            fig.update_layout(**_layout(height=260, title="Class Distribution", showlegend=False))
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            fig2 = px.scatter(df, x="temp", y="smoke", color="label",
                              color_discrete_map=CLASS_COLORS, height=260,
                              title="Smoke vs Temperature")
            fig2.update_layout(**_layout())
            st.plotly_chart(fig2, use_container_width=True)

        with cc:
            fig3 = px.scatter(df, x="co", y="lpg", color="label",
                              color_discrete_map=CLASS_COLORS, height=260,
                              title="CO vs LPG")
            fig3.update_layout(**_layout())
            st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(df[["co","smoke","temp","lpg","humidity","label","confidence"]].head(20),
                     use_container_width=True, height=220, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEDERATED METRICS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    data = load_metrics()
    if not data and not live_mode:
        data = mock_metrics()

    if not data or not data.get("rounds"):
        st.info("⏳ Waiting for federated training to begin…")
    else:
        rounds     = data["rounds"]
        global_acc = data["global_acc"]
        edge_acc   = data.get("edge_acc", {})
        latency    = data.get("latency", {})
        energy     = data.get("energy", {})
        agg_weights= data.get("agg_weights", [])
        reputation = data.get("reputation", {})
        edge_ids   = list(edge_acc.keys())

        curr_r   = rounds[-1]
        curr_g   = global_acc[-1]
        prev_g   = global_acc[-2] if len(global_acc) > 1 else None

        # KPIs
        st.markdown('<div class="section-hdr">📊 Live KPIs</div>', unsafe_allow_html=True)
        kpi_cols = st.columns(len(edge_ids) + 2)
        kpi_cols[0].markdown(kpi_html(f"{curr_g:.1f}%", "🌐 Global Accuracy",
                                      delta=(curr_g - prev_g) if prev_g else None), unsafe_allow_html=True)
        kpi_cols[1].markdown(kpi_html(str(curr_r), "🔄 Round"), unsafe_allow_html=True)
        for i, eid in enumerate(edge_ids):
            al   = edge_acc.get(eid, [])
            anow = al[-1] if al else 0
            aprev= al[-2] if len(al) > 1 else None
            kpi_cols[i+2].markdown(
                kpi_html(f"{anow:.1f}%", EDGE_LABELS.get(eid, eid),
                         delta=(anow - aprev) if aprev else None,
                         color=EDGE_COLORS.get(eid)),
                unsafe_allow_html=True,
            )

        st.markdown("")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown('<div class="section-hdr">📈 Accuracy Over Rounds</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rounds, y=global_acc, name="Global (weighted)",
                                     line=dict(color="#FFD700", width=3, dash="dot"), mode="lines+markers"))
            for eid in edge_ids:
                al = edge_acc.get(eid, [])
                fig.add_trace(go.Scatter(x=rounds[:len(al)], y=al, name=EDGE_LABELS.get(eid,eid),
                                         line=dict(color=EDGE_COLORS.get(eid,"#aaa"), width=2), mode="lines+markers"))
            fig.update_layout(**_layout(yaxis=dict(range=[0,105],title="Accuracy (%)"), height=320,
                               xaxis=dict(title="Round")))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-hdr">⚖️ Aggregation Weights</div>', unsafe_allow_html=True)
            latest_w = agg_weights[-1] if agg_weights else {}
            if latest_w:
                fig2 = go.Figure(go.Pie(
                    labels=[EDGE_LABELS.get(e,e) for e in latest_w],
                    values=list(latest_w.values()),
                    marker=dict(colors=[EDGE_COLORS.get(e,"#aaa") for e in latest_w]),
                    hole=0.5, textinfo="percent",
                ))
                fig2.update_layout(**_layout(height=260, showlegend=True))
                st.plotly_chart(fig2, use_container_width=True)
                wdf = pd.DataFrame([{"Edge": EDGE_LABELS.get(e,e), "Weight": f"{w:.4f}",
                                     "Share": f"{w*100:.1f}%"} for e,w in latest_w.items()])
                st.dataframe(wdf, hide_index=True, use_container_width=True)

        if len(agg_weights) > 1:
            st.markdown('<div class="section-hdr">📉 Weight Evolution Over Rounds</div>', unsafe_allow_html=True)
            fig3 = go.Figure()
            for eid in edge_ids:
                ws = [rnd.get(eid,0) for rnd in agg_weights]
                fig3.add_trace(go.Bar(name=EDGE_LABELS.get(eid,eid), x=rounds[:len(ws)], y=ws,
                                      marker_color=EDGE_COLORS.get(eid,"#aaa")))
            fig3.update_layout(**_layout(barmode="stack", height=260, xaxis=dict(title="Round"),
                                yaxis=dict(title="Weight")))
            st.plotly_chart(fig3, use_container_width=True)

        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.markdown('<div class="section-hdr">⏱ Latency (ms)</div>', unsafe_allow_html=True)
            fig4 = go.Figure()
            for eid in edge_ids:
                ll = latency.get(eid,[])
                fig4.add_trace(go.Scatter(x=rounds[:len(ll)], y=ll, name=EDGE_LABELS.get(eid,eid),
                                          line=dict(color=EDGE_COLORS.get(eid,"#aaa"), width=2), mode="lines+markers"))
            fig4.update_layout(**_layout(height=260, xaxis=dict(title="Round"), yaxis=dict(title="ms")))
            st.plotly_chart(fig4, use_container_width=True)

        with c4:
            st.markdown('<div class="section-hdr">⚡ Energy (J)</div>', unsafe_allow_html=True)
            fig5 = go.Figure()
            for eid in edge_ids:
                el = energy.get(eid,[])
                fig5.add_trace(go.Bar(name=EDGE_LABELS.get(eid,eid), x=rounds[:len(el)], y=el,
                                      marker_color=EDGE_COLORS.get(eid,"#aaa")))
            fig5.update_layout(**_layout(barmode="group", height=260, xaxis=dict(title="Round"),
                                yaxis=dict(title="Joules")))
            st.plotly_chart(fig5, use_container_width=True)

        if reputation:
            st.markdown('<div class="section-hdr">🔒 Edge Reputation Scores</div>', unsafe_allow_html=True)
            fig6 = go.Figure()
            for eid in edge_ids:
                rl = reputation.get(eid,[])
                fig6.add_trace(go.Scatter(x=rounds[:len(rl)], y=rl, name=EDGE_LABELS.get(eid,eid),
                                          line=dict(color=EDGE_COLORS.get(eid,"#aaa"), width=2),
                                          mode="lines+markers", fill="tozeroy"))
            fig6.update_layout(**_layout(yaxis=dict(range=[0,1.1],title="Score"), height=240,
                                xaxis=dict(title="Round")))
            st.plotly_chart(fig6, use_container_width=True)

        st.markdown('<div class="section-hdr">🖥 Node Health Status</div>', unsafe_allow_html=True)
        rows = []
        for eid in edge_ids:
            al = edge_acc.get(eid,[])
            ll = latency.get(eid,[])
            el = energy.get(eid,[])
            rl = reputation.get(eid,[])
            rows.append({
                "Node":         EDGE_LABELS.get(eid,eid),
                "Rounds":       len(al),
                "Accuracy %":   f"{al[-1]:.2f}" if al else "—",
                "Latency ms":   f"{ll[-1]:.1f}" if ll else "—",
                "Energy J":     f"{el[-1]:.5f}" if el else "—",
                "Reputation":   f"{rl[-1]:.4f}" if rl else "—",
                "Status":       "🟢 Active" if len(al)==curr_r else "🔴 Stalled",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Last updated: round {curr_r} · {datetime.now().strftime('%H:%M:%S')}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AGENT PANEL
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown("### 🧠 Multi-Agent Analysis Panel")
    st.caption("Four specialized agents run concurrently every federated round and share their findings here.")

    AGENT_META = {
        "analyst":   {"icon":"📊","name":"Analyst Agent",   "desc":"Accuracy trends, convergence, edge divergence"},
        "anomaly":   {"icon":"🔍","name":"Anomaly Agent",   "desc":"Sensor anomaly detection across IoT fleet"},
        "predictor": {"icon":"🔮","name":"Predictor Agent", "desc":"Next-round accuracy forecast (linear extrapolation)"},
        "security":  {"icon":"🛡️","name":"Security Agent",  "desc":"Poisoning detection, HMAC, trust scoring"},
    }

    if live_mode:
        agent_data = api_get("/agents")
    else:
        agent_data = {
            "round": 8,
            "results": {
                "analyst":   {"status":"done","findings":{"trend":"improving","delta_3rounds":3.1,
                               "edge_divergence":{"edge1":1.1,"edge2":3.2,"edge3":0.8},"rounds_analysed":8},"alerts":[]},
                "anomaly":   {"status":"done","findings":{"anomalies_detected":4,"breakdown":
                               {"fire_risk":2,"gas_leak":1,"temp_spike":1},"batch_size":100},
                               "alerts":["🔥 FIRE RISK: 2 readings exceed thresholds","⛽ GAS LEAK: 1 reading"]},
                "predictor": {"status":"done","findings":{"predicted_global_acc":84.2,
                               "edge_forecasts":{"edge1":86.1,"edge2":81.0,"edge3":83.5},
                               "predicted_best_edge":"edge1","predicted_worst_edge":"edge2"},"alerts":[]},
                "security":  {"status":"done","findings":{"suspicious_edges":[],"norm_deviations":
                               {"edge1":0.02,"edge2":0.05,"edge3":0.01},"total_alerts":0,"round_secure":True},"alerts":[]},
            },
            "alerts": ["🔥 FIRE RISK: 2 readings exceed thresholds","⛽ GAS LEAK: 1 reading"],
        }

    if not agent_data.get("results"):
        st.info("⏳ Waiting for first federated round…")
    else:
        rn       = agent_data.get("round", 0)
        results  = agent_data.get("results", {})
        all_alts = agent_data.get("alerts", [])

        st.markdown(f"**Round {rn} Analysis** — {len(results)} agents active")
        st.markdown('<div class="section-hdr">🚨 All Alerts This Round</div>', unsafe_allow_html=True)
        st.markdown(alert_html(all_alts), unsafe_allow_html=True)
        st.markdown("")

        ca, cb = st.columns(2, gap="large")
        for idx, (key, meta) in enumerate(AGENT_META.items()):
            col = ca if idx % 2 == 0 else cb
            r   = results.get(key, {})
            status   = r.get("status","idle")
            findings = r.get("findings", {})
            alerts   = r.get("alerts", [])
            bdg_cls  = "bdg-done" if status=="done" else ("bdg-run" if status=="running" else "bdg-err")

            with col:
                with st.expander(f"{meta['icon']} {meta['name']}", expanded=True):
                    st.markdown(
                        f'<div class="agent-hdr">'
                        f'<span style="font-size:.78rem;color:#6e7681">{meta["desc"]}</span>'
                        f'<span class="badge {bdg_cls}">{status.upper()}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if key == "analyst" and findings:
                        t  = findings.get("trend","?")
                        tc = "#22C55E" if t=="improving" else ("#EF4444" if t=="regressing" else "#F59E0B")
                        st.markdown(f"**Trend:** <span style='color:{tc};font-weight:700'>{t.upper()}</span>", unsafe_allow_html=True)
                        if "delta_3rounds" in findings:
                            st.markdown(f"**Δ 3-round:** `{findings['delta_3rounds']:+.2f}%`")
                        for eid, div in findings.get("edge_divergence",{}).items():
                            c = EDGE_COLORS.get(eid,"#aaa")
                            w = "⚠️" if div > 10 else ""
                            st.markdown(f"<span style='color:{c}'>● {eid}</span> divergence: `{div:.2f}%` {w}", unsafe_allow_html=True)

                    elif key == "anomaly" and findings:
                        total = findings.get("anomalies_detected",0)
                        st.markdown(f"**Anomalies:** `{total}` / `{findings.get('batch_size','?')}` readings")
                        for k, v in findings.get("breakdown",{}).items():
                            ico = "🔥" if "fire" in k else ("⛽" if "gas" in k else "🌡️")
                            color = "#EF4444" if v > 0 else "#22C55E"
                            st.markdown(f"{ico} {k.replace('_',' ').title()}: <span style='color:{color};font-weight:600'>{v}</span>", unsafe_allow_html=True)

                    elif key == "predictor" and findings:
                        pg = findings.get("predicted_global_acc")
                        if pg: st.markdown(f"**Next-round global:** `{pg:.1f}%`")
                        for eid, p in findings.get("edge_forecasts",{}).items():
                            c = EDGE_COLORS.get(eid,"#aaa")
                            st.markdown(f"<span style='color:{c}'>● {eid}</span>: `{p:.1f}%`", unsafe_allow_html=True)
                        if findings.get("predicted_best_edge"):
                            st.markdown(f"🏆 Best: **{findings['predicted_best_edge']}** | Worst: **{findings['predicted_worst_edge']}**")

                    elif key == "security" and findings:
                        sec = findings.get("round_secure", True)
                        st.markdown(f"**Round secure:** {'✅ Yes' if sec else '🚨 No'}")
                        for eid, dev in findings.get("norm_deviations",{}).items():
                            c = "#EF4444" if dev > 0.3 else EDGE_COLORS.get(eid,"#aaa")
                            flag = " 🚩" if dev > 0.3 else ""
                            st.markdown(f"<span style='color:{c}'>● {eid}</span> norm deviation: `{dev:.4f}`{flag}", unsafe_allow_html=True)
                        susp = findings.get("suspicious_edges",[])
                        if susp: st.markdown(f"⚠️ **Excluded from aggregation:** {', '.join(susp)}")

                    if alerts:
                        for a in alerts:
                            st.markdown(alert_html([a]), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SECURITY MONITOR
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown("### 🔐 Security Monitor")
    st.caption("HMAC payload signing, JWT edge authentication, Z-score poisoning detection, and rate limiting.")

    if live_mode:
        sec_data = api_get("/security")
    else:
        now = time.time()
        sec_data = {
            "events": [
                {"edge_id":"edge1","type":"HMAC_OK",    "detail":"Signature valid ✓",             "timestamp":now-300},
                {"edge_id":"edge2","type":"HMAC_OK",    "detail":"Signature valid ✓",             "timestamp":now-290},
                {"edge_id":"edge3","type":"HMAC_OK",    "detail":"Signature valid ✓",             "timestamp":now-285},
                {"edge_id":"system","type":"AGGREGATION","detail":"Round 7 aggregation complete", "timestamp":now-280},
                {"edge_id":"system","type":"AGENT_ALERT","detail":"🔥 FIRE RISK: 2 readings",     "timestamp":now-279},
                {"edge_id":"edge1","type":"HMAC_OK",    "detail":"Signature valid ✓",             "timestamp":now-180},
                {"edge_id":"edge2","type":"HMAC_OK",    "detail":"Signature valid ✓",             "timestamp":now-175},
                {"edge_id":"edge3","type":"HMAC_OK",    "detail":"Signature valid ✓",             "timestamp":now-170},
                {"edge_id":"system","type":"AGGREGATION","detail":"Round 8 aggregation complete", "timestamp":now-165},
            ],
            "norm_history": {
                "edge1": [1.21, 1.23, 1.25, 1.24, 1.26, 1.27, 1.25, 1.26],
                "edge2": [1.18, 1.19, 1.20, 1.21, 1.20, 1.19, 1.21, 1.20],
                "edge3": [1.30, 1.29, 1.31, 1.28, 1.30, 1.31, 1.29, 1.30],
            },
        }

    c_log, c_norms = st.columns([3, 2], gap="large")

    with c_log:
        st.markdown('<div class="section-hdr">📋 Security Event Log</div>', unsafe_allow_html=True)
        events = sec_data.get("events", [])
        if not events:
            st.markdown('<div class="sec-row sec-info">No events recorded yet</div>', unsafe_allow_html=True)
        else:
            for ev in reversed(events[-25:]):
                ts     = datetime.fromtimestamp(ev.get("timestamp",0)).strftime("%H:%M:%S")
                etype  = ev.get("type","")
                detail = ev.get("detail","")
                eid    = ev.get("edge_id","")
                css    = "sec-fail" if "FAIL" in etype or "POISON" in etype \
                         else "sec-warn" if "AGENT_ALERT" in etype or "RATE" in etype \
                         else "sec-ok"  if "OK" in etype \
                         else "sec-info"
                st.markdown(
                    f'<div class="sec-row {css}">'
                    f'<span style="opacity:.5">[{ts}]</span> '
                    f'<span style="color:#e6edf3">{eid}</span> · '
                    f'<span style="font-weight:600">{etype}</span> · {detail}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with c_norms:
        st.markdown('<div class="section-hdr">📊 Weight Norm History</div>', unsafe_allow_html=True)
        st.caption("Sudden spikes = possible model poisoning attack")
        nh = sec_data.get("norm_history", {})
        if nh:
            fig_n = go.Figure()
            for eid, norms in nh.items():
                fig_n.add_trace(go.Scatter(
                    x=list(range(len(norms))), y=norms,
                    name=EDGE_LABELS.get(eid,eid),
                    line=dict(color=EDGE_COLORS.get(eid,"#aaa"), width=2),
                    mode="lines+markers",
                ))
            fig_n.update_layout(**_layout(height=260, xaxis=dict(title="Round"),
                                 yaxis=dict(title="L2 Norm")))
            st.plotly_chart(fig_n, use_container_width=True)

        st.markdown('<div class="section-hdr">🛡️ Security Features Active</div>', unsafe_allow_html=True)
        hmac_fails   = [e for e in events if "FAIL" in e.get("type","")]
        poison_evts  = [e for e in events if "POISON" in e.get("type","")]
        rate_evts    = [e for e in events if "RATE" in e.get("type","")]

        sc1, sc2, sc3 = st.columns(3)
        sc1.markdown(kpi_html(
            "✗" if hmac_fails else "✓",
            "HMAC Status",
            color="#EF4444" if hmac_fails else "#22C55E",
        ), unsafe_allow_html=True)
        sc2.markdown(kpi_html(
            str(len(poison_evts)),
            "Poison Events",
            color="#EF4444" if poison_evts else "#22C55E",
        ), unsafe_allow_html=True)
        sc3.markdown(kpi_html(
            str(len(rate_evts)),
            "Rate Limit Hits",
            color="#F59E0B" if rate_evts else "#22C55E",
        ), unsafe_allow_html=True)

        with st.expander("🔒 Security Layer Details"):
            st.markdown("""
| Feature | Status |
|---------|--------|
| HMAC-SHA256 payload signing | 🟢 Active |
| JWT edge authentication tokens | 🟢 Active |
| Replay attack protection (60s window) | 🟢 Active |
| Z-score poisoning detector | 🟢 Active |
| Rate limiter (5 req / 60s per edge) | 🟢 Active |
| Reputation-based exclusion | 🟢 Active |
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — WINDOW VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown("### 📅 Temporal Sliding Window Visualizer")
    st.caption(
        "Each federated round trains on a **different time slice** of the IoT device's data. "
        "This simulates real-world streaming where new sensor data continuously arrives at the edge."
    )

    w_strat   = st.selectbox("Window Strategy", ["sliding", "expanding", "full"], index=0)
    w_frac    = st.slider("Window Fraction (% of data per round)", 10, 60, 30, 5)
    w_step    = st.slider("Window Step (% advance per round)", 5, 30, 10, 5)
    n_rounds  = st.slider("Number of rounds to visualize", 3, 15, 10)
    n_samples = 1000   # total data points (normalized)

    fig_win = go.Figure()

    # Gray background = full dataset
    fig_win.add_trace(go.Bar(
        x=[f"Round {r}" for r in range(1, n_rounds+1)],
        y=[n_samples] * n_rounds,
        marker_color="#21262d", name="Full Dataset", showlegend=True,
    ))

    for r in range(n_rounds):
        if w_strat == "full":
            s_idx, e_idx = 0, n_samples
        elif w_strat == "expanding":
            s_idx  = 0
            e_frac = min(1.0, (w_frac/100) + r * (w_step/100))
            e_idx  = int(n_samples * e_frac)
        else:  # sliding
            s_frac = (r * w_step / 100) % 1.0
            e_frac = min(1.0, s_frac + w_frac / 100)
            s_idx  = int(n_samples * s_frac)
            e_idx  = int(n_samples * e_frac)

        window_size = e_idx - s_idx
        color_intensity = 0.5 + 0.5 * (r / max(n_rounds-1, 1))
        blue_val = int(100 + 155 * color_intensity)

        fig_win.add_trace(go.Bar(
            x=[f"Round {r+1}"],
            y=[window_size],
            base=[s_idx],
            marker_color=f"rgba(14,165,233,{0.4 + 0.5*(r/max(n_rounds-1,1)):.2f})",
            name=f"R{r+1} window",
            showlegend=False,
            text=f"{window_size} samples",
            textposition="inside",
            textfont=dict(size=9, color="white"),
        ))

    fig_win.update_layout(**_layout(
        barmode="overlay",
        height=380,
        title=f"Training Windows — {w_strat.title()} Strategy",
        xaxis=dict(title="Federated Round"),
        yaxis=dict(title="Data Index (time →)", range=[0, n_samples]),
    ))
    st.plotly_chart(fig_win, use_container_width=True)

    # Explanation cards
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown(
            '<div class="win-box">'
            '<div class="win-title">🔵 Sliding Window</div>'
            '<div class="win-sub">Fixed-size window moves forward each round.<br>'
            'Simulates: real-time IoT data stream arriving continuously.<br>'
            '<b>Best for:</b> detecting temporal pattern shifts in sensor data.</div>'
            '</div>', unsafe_allow_html=True,
        )
    with e2:
        st.markdown(
            '<div class="win-box">'
            '<div class="win-title">🟡 Expanding Window</div>'
            '<div class="win-sub">Window grows from the beginning each round.<br>'
            'Simulates: cumulative learning as more history becomes available.<br>'
            '<b>Best for:</b> stable environments with growing datasets.</div>'
            '</div>', unsafe_allow_html=True,
        )
    with e3:
        st.markdown(
            '<div class="win-box">'
            '<div class="win-title">⚪ Full Dataset</div>'
            '<div class="win-sub">Uses all training data every round (original behaviour).<br>'
            'Simulates: batch federated learning with static data.<br>'
            '<b>Best for:</b> baseline comparison.</div>'
            '</div>', unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### 📐 Why Temporal Splitting Matters for This Dataset")
    st.markdown("""
The IoT telemetry dataset has **405,184 timestamped readings** sorted chronologically.
A key principle of time-series machine learning is that you should **never randomly shuffle** the data,
because that leaks future information into training (data leakage).

The temporal sliding window respects this by:
- Training always uses **past data only** (no future leakage)
- Each round the model sees a **fresh slice**, simulating real sensor streaming  
- The held-out **test set is always the final 20%** (most recent data = hardest generalization)
- Class weights are **recomputed per window** since Normal/Warning/Critical ratios shift over time
""")


# ── Auto-refresh ──────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
