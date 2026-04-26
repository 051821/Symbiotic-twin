"""
metrics/tracker.py
Centralized metrics store. Records per-round performance for all edges
and the global model. Persists to JSON for the Streamlit dashboard.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from config.loader import get_config
from config.logging_config import setup_logger
from shared.utils import ensure_dir

logger = setup_logger("tracker")
_lock  = threading.Lock()

class MetricsTracker:
    """Thread-safe tracker for federated learning metrics."""

    def __init__(self, metrics_path: str = None):
        cfg = get_config()
        self.metrics_path = Path(metrics_path or cfg["dashboard"]["metrics_path"])
        ensure_dir(self.metrics_path.parent)

        self.rounds:      List[int]              = []
        self.global_acc:  List[float]            = []
        self.edge_acc:    Dict[str, List[float]] = defaultdict(list)
        self.latency:     Dict[str, List[float]] = defaultdict(list)
        self.energy:      Dict[str, List[float]] = defaultdict(list)
        self.agg_weights: List[Dict[str, float]] = []
        self.reputation:  Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    def record_round(
        self,
        round_num:   int,
        global_acc:  float,
        edge_metrics: Dict[str, Dict[str, float]],
        agg_weights:  Dict[str, float],
        reputations:  Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record all metrics for one federated round.

        Args:
            round_num    : Current round index (1-based).
            global_acc   : Accuracy of global model after aggregation.
            edge_metrics : {edge_id: {accuracy, latency_ms, energy_j}}
            agg_weights  : {edge_id: weight}
            reputations  : {edge_id: score}  (optional)
        """
        with _lock:
            self.rounds.append(round_num)
            self.global_acc.append(round(global_acc, 4))
            self.agg_weights.append({k: round(v, 4) for k, v in agg_weights.items()})

            for edge_id, m in edge_metrics.items():
                self.edge_acc[edge_id].append(round(m.get("accuracy", 0.0), 4))
                self.latency [edge_id].append(round(m.get("latency_ms", 0.0), 4))
                self.energy  [edge_id].append(round(m.get("energy_j", 0.0), 6))

            if reputations:
                for edge_id, score in reputations.items():
                    self.reputation[edge_id].append(round(score, 4))

        logger.info(
            f"Round {round_num} recorded | global_acc={global_acc:.2f}% | "
            f"edges={list(edge_metrics.keys())}"
        )
        self.save()

    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist all metrics to JSON for the dashboard."""
        with _lock:
            payload = {
                "rounds":      self.rounds,
                "global_acc":  self.global_acc,
                "edge_acc":    dict(self.edge_acc),
                "latency":     dict(self.latency),
                "energy":      dict(self.energy),
                "agg_weights": self.agg_weights,
                "reputation":  dict(self.reputation),
            }
        with open(self.metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        if not self.metrics_path.exists():
            return {}
        with open(self.metrics_path, "r") as f:
            return json.load(f)

    def summary(self) -> str:
        """Return a brief text summary of the latest round."""
        if not self.rounds:
            return "No rounds recorded yet."
        r   = self.rounds[-1]
        acc = self.global_acc[-1]
        w   = self.agg_weights[-1] if self.agg_weights else {}
        return (
            f"Round {r} | Global Accuracy: {acc:.2f}% | "
            f"Weights: {w}"
        )


# Module-level singleton
_tracker: Optional[MetricsTracker] = None


def get_tracker() -> MetricsTracker:
    global _tracker
    if _tracker is None:
        _tracker = MetricsTracker()
    return _tracker
