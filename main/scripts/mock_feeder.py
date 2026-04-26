"""
scripts/mock_feeder.py
Simulates federated rounds writing to logs/metrics.json.
Run alongside the dashboard for a demo without the full Docker stack.
"""
import json, random, time
from pathlib import Path

METRICS = Path("logs/metrics.json")
TOTAL_N = 149960 + 89452 + 84734


def load():
    if not METRICS.exists():
        return {"rounds":[], "global_acc":[], "edge_acc":{"edge1":[],"edge2":[],"edge3":[]},
                "latency":{"edge1":[],"edge2":[],"edge3":[]}, "energy":{"edge1":[],"edge2":[],"edge3":[]},
                "agg_weights":[], "reputation":{"edge1":[],"edge2":[],"edge3":[]}}
    with open(METRICS) as f:
        return json.load(f)


def save(data):
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS, "w") as f:
        json.dump(data, f, indent=2)


def step(data):
    r = (data["rounds"] or [0])[-1] + 1
    # Simulate sliding window effect: accuracy improves as model sees more data
    base = min(94, 56 + r * 2.1)
    accs = {"edge1": base + random.uniform(0,3.5),
            "edge2": base - random.uniform(0,2.5),
            "edge3": base + random.uniform(-1.5,2.5)}
    # Weighted global accuracy
    w_acc = (accs["edge1"]*149960 + accs["edge2"]*89452 + accs["edge3"]*84734) / TOTAL_N

    data["rounds"].append(r)
    data["global_acc"].append(round(w_acc, 2))

    for eid, v in accs.items():
        data["edge_acc"][eid].append(round(v, 2))
        prev_lat = (data["latency"][eid] or [15])[-1]
        data["latency"][eid].append(round(max(5, prev_lat + random.uniform(-.8,.4)), 2))
        prev_eng = (data["energy"][eid] or [.003])[-1]
        data["energy"][eid].append(round(max(.001, prev_eng + random.uniform(-.0002,.0003)), 5))

    w1, w2 = 149960/TOTAL_N, 89452/TOTAL_N
    data["agg_weights"].append({"edge1":round(w1,4),"edge2":round(w2,4),"edge3":round(84734/TOTAL_N,4)})

    for eid in ["edge1","edge2","edge3"]:
        prev = (data["reputation"][eid] or [.8])[-1]
        data["reputation"][eid].append(round(min(1.0, max(.1, prev + random.uniform(-.01,.02))),3))
    return data


def main(interval=3):
    print(f"Mock feeder starting → logs/metrics.json (every {interval}s)")
    data = load()
    try:
        while True:
            data = step(data)
            save(data)
            print(f"Round {data['rounds'][-1]} | global_acc={data['global_acc'][-1]:.2f}%")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped")


if __name__ == "__main__":
    main()
