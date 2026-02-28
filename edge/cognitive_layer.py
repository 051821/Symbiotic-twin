"""
edge/cognitive_layer.py
Adaptive intelligence layer — tunes learning rate and training behaviour
based on accuracy history and energy budget.
"""

from typing import List
from config.logging_config import setup_logger

logger = setup_logger("cognitive")


class CognitiveLayer:
    def __init__(self, edge_id: str, initial_lr: float = 0.001, energy_budget_j: float = 50.0):
        self.edge_id       = edge_id
        self.lr            = initial_lr
        self.energy_budget = energy_budget_j
        self.energy_spent  = 0.0
        self.acc_history:  List[float] = []
        self.lr_history:   List[float] = []

    def adapt(self, current_accuracy: float, energy_used_j: float) -> float:
        self.acc_history.append(current_accuracy)
        self.energy_spent += energy_used_j

        if len(self.acc_history) >= 3:
            recent = self.acc_history[-3:]
            delta  = max(recent) - min(recent)
            if delta < 0.5:
                self.lr *= 0.7
                logger.info(f"[{self.edge_id}] Plateau detected → LR={self.lr:.6f}")

        if len(self.acc_history) >= 2:
            if self.acc_history[-1] > self.acc_history[-2] + 1.0:
                self.lr = min(self.lr * 1.1, 0.01)
                logger.info(f"[{self.edge_id}] Improving → LR={self.lr:.6f}")

        if self.energy_spent > self.energy_budget:
            self.lr *= 0.5
            logger.warning(f"[{self.edge_id}] Energy exceeded → LR={self.lr:.6f}")

        self.lr = max(1e-6, min(self.lr, 0.05))
        self.lr_history.append(self.lr)
        return self.lr

    def should_skip_round(self) -> bool:
        return self.energy_spent > self.energy_budget * 2

    def status(self) -> dict:
        return {
            "edge_id":       self.edge_id,
            "current_lr":    self.lr,
            "energy_spent":  round(self.energy_spent, 4),
            "energy_budget": self.energy_budget,
            "rounds":        len(self.acc_history),
            "last_accuracy": self.acc_history[-1] if self.acc_history else None,
        }
