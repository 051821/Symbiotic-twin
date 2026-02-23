"""
scripts/mock_feeder.py
Simple simulator that appends rounds to `logs/metrics.json` to drive the dashboard.
Run alongside the dashboard to see live updates without running the full stack.
"""
import json
import random
import time
from pathlib import Path

METRICS = Path("logs/metrics.json")

def load():
    if not METRICS.exists():
        return {
            "rounds": [],
            "global_acc": [],
            "edge_acc": {"edge1": [], "edge2": [], "edge3": []},
            "latency": {"edge1": [], "edge2": [], "edge3": []},
            "energy": {"edge1": [], "edge2": [], "edge3": []},
            "agg_weights": [],
            "reputation": {"edge1": [], "edge2": [], "edge3": []},
        }
    with open(METRICS, "r") as f:
        return json.load(f)

def save(data):
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS, "w") as f:
        json.dump(data, f, indent=2)

def step(data):
    # determine next round
    r = (data.get("rounds") or [0])[-1] + 1

    # base accuracies slowly improve
    base = 60 + r * 1.8
    e1 = base + random.uniform(0, 3)
    e2 = base - random.uniform(0, 2)
    e3 = base + random.uniform(-1, 2)
    g = (e1 + e2 + e3) / 3

    data.setdefault("rounds", []).append(r)
    data.setdefault("global_acc", []).append(round(g, 2))

    for eid, val in [("edge1", e1), ("edge2", e2), ("edge3", e3)]:
        data.setdefault("edge_acc", {}).setdefault(eid, []).append(round(val, 2))

    # latency and energy random walk
    for eid in ["edge1", "edge2", "edge3"]:
        prev_lat = (data.get("latency", {}).get(eid) or [random.uniform(10, 20)])[-1]
        lat = max(5.0, prev_lat + random.uniform(-0.8, 0.5))
        data.setdefault("latency", {}).setdefault(eid, []).append(round(lat, 2))

        prev_eng = (data.get("energy", {}).get(eid) or [0.0025])[-1]
        eng = max(0.0005, prev_eng + random.uniform(-0.0002, 0.0003))
        data.setdefault("energy", {}).setdefault(eid, []).append(round(eng, 6))

    # aggregation weights favor edge1 slightly
    w1 = min(0.6, 0.45 + r * 0.005 + random.uniform(-0.02, 0.02))
    w2 = max(0.2, 0.28 + random.uniform(-0.02, 0.02))
    w3 = max(0.15, 1.0 - w1 - w2)
    weights = {"edge1": round(w1, 4), "edge2": round(w2, 4), "edge3": round(w3, 4)}
    data.setdefault("agg_weights", []).append(weights)

    # reputation nudges upwards
    for eid in ["edge1", "edge2", "edge3"]:
        prev = (data.get("reputation", {}).get(eid) or [0.8])[-1]
        rep = min(1.0, max(0.0, prev + random.uniform(-0.01, 0.02)))
        data.setdefault("reputation", {}).setdefault(eid, []).append(round(rep, 3))

    return data

def main(interval=3):
    print("Mock feeder starting. Writing to logs/metrics.json every", interval, "s")
    data = load()
    try:
        while True:
            data = step(data)
            save(data)
            print(f"Wrote round {data['rounds'][-1]}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped")

if __name__ == '__main__':
    main()
