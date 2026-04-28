"""
edge/cognitive_layer.py
Adaptive intelligence layer — tunes learning rate and training behaviour
based on accuracy history and energy budget.
"""

from typing import List
from config.logging_config import setup_logger

logger = setup_logger("cognitive")


class CognitiveLayer:
    def __init__(self, edge_id: str, initial_lr: float = 0.001, energy_budget_j: float = 10000.0):
        self.edge_id       = edge_id
        self.lr            = initial_lr
        self.energy_budget = energy_budget_j
        self.energy_spent  = 0.0
        self.acc_history:  List[float] = []
        self.lr_history:   List[float] = []

    def adapt(self, current_accuracy: float, energy_used_j: float) -> float:
        self.acc_history.append(current_accuracy)
        self.energy_spent += energy_used_j

        # Use a longer horizon and gentler decay to avoid over-reacting to noisy rounds.
        if len(self.acc_history) >= 5:
            recent = self.acc_history[-5:]
            delta  = max(recent) - min(recent)
            if delta < 0.2:
                self.lr *= 0.9
                logger.info(f"[{self.edge_id}] Plateau detected → LR={self.lr:.6f}")

        if len(self.acc_history) >= 2:
            if self.acc_history[-1] > self.acc_history[-2] + 0.5:
                self.lr = min(self.lr * 1.05, 0.005)
                logger.info(f"[{self.edge_id}] Improving → LR={self.lr:.6f}")

        if self.energy_spent > self.energy_budget:
            self.lr *= 0.8
            logger.warning(f"[{self.edge_id}] Energy exceeded → LR={self.lr:.6f}")

        self.lr = max(1e-4, min(self.lr, 0.01))
        self.lr_history.append(self.lr)
        return self.lr

    def training_plan(self) -> dict:
        """
        Return adaptive per-round training plan.
        The plan balances three goals:
          1) Keep enough data/steps for high accuracy.
          2) Reduce compute once accuracy becomes stable.
          3) Recover quickly when accuracy drops.
        """
        rounds_seen = len(self.acc_history)
        sample_ratio = 1.0
        extra_epochs = 0

        # Warm-up rounds: prioritize convergence quality.
        if rounds_seen < 2:
            return {"sample_ratio": 1.0, "extra_epochs": 0}

        recent = self.acc_history[-3:]
        recent_best = max(recent)
        recent_last = recent[-1]
        drift = recent_best - recent_last

        # If the latest round dropped materially, train a bit harder next round.
        if drift > 1.5:
            sample_ratio = 1.0
            extra_epochs = 1
        # Stable and strong accuracy -> trim compute budget.
        elif recent_last >= 86.0 and drift < 0.6:
            sample_ratio = 0.72
        elif recent_last >= 81.0 and drift < 0.9:
            sample_ratio = 0.82

        # If energy is growing too quickly, apply an additional soft cap.
        budget_ratio = self.energy_spent / max(self.energy_budget, 1e-9)
        if budget_ratio > 0.75:
            sample_ratio = min(sample_ratio, 0.68)
        elif budget_ratio > 0.5:
            sample_ratio = min(sample_ratio, 0.78)

        sample_ratio = max(0.7, min(sample_ratio, 1.0))
        return {"sample_ratio": sample_ratio, "extra_epochs": extra_epochs}

    def should_skip_round(self) -> bool:
        # Skipping too early can create unstable/zero metrics for edges.
        return self.energy_spent > self.energy_budget * 3

    def status(self) -> dict:
        return {
            "edge_id":       self.edge_id,
            "current_lr":    self.lr,
            "energy_spent":  round(self.energy_spent, 4),
            "energy_budget": self.energy_budget,
            "rounds":        len(self.acc_history),
            "last_accuracy": self.acc_history[-1] if self.acc_history else None,
        }
