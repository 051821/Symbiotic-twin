from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def build_metrics_payload(
    rounds: List[int],
    global_acc: List[float],
    edge_acc: Dict[str, List[float]],
    latency: Dict[str, List[float]],
    energy: Dict[str, List[float]],
    agg_weights: List[Dict[str, float]],
    reputation: Dict[str, List[float]],
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "rounds": rounds,
        "global_acc": global_acc,
        "edge_acc": edge_acc,
        "latency": latency,
        "energy": energy,
        "agg_weights": agg_weights,
        "reputation": reputation,
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def save_metrics_payload(path: Path, payload: Dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
