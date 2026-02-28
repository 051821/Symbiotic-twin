"""
edge/trainer.py
Local model training and evaluation on edge nodes.

Improvements:
  - class_weights passed to CrossEntropyLoss to fix Normal/Warning/Critical imbalance
  - Evaluates on test set after every epoch so we can track real generalisation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional

from config.loader import get_config
from config.logging_config import setup_logger
from metrics.accuracy import compute_batch_accuracy
from metrics.latency import LatencyTimer
from metrics.energy import EnergyMonitor

logger = setup_logger("trainer")


class LocalTrainer:
    """Trains a model on a local edge dataset for one federated round."""

    def __init__(
        self,
        model: nn.Module,
        edge_id: str,
        class_weights: Optional[torch.Tensor] = None,
    ):
        cfg = get_config()
        self.edge_id = edge_id
        self.model   = model
        self.epochs  = cfg["system"]["epochs_per_round"]
        self.lr      = cfg["system"]["learning_rate"]
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Use class-weighted loss to handle Normal >> Warning > Critical imbalance
        if class_weights is not None:
            cw = class_weights.to(self.device)
            logger.info(f"[{edge_id}] Using class weights: {cw.tolist()}")
        else:
            cw = None

        self.criterion = nn.CrossEntropyLoss(weight=cw)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def update_class_weights(self, class_weights: torch.Tensor) -> None:
        """Update loss weights when the window shifts to a new data slice."""
        cw = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=cw)
        logger.info(f"[{self.edge_id}] Class weights updated: {cw.tolist()}")

    def train(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Train for configured epochs. Returns (accuracy_%, latency_ms, energy_j).
        Each epoch processes one full pass over the current temporal window.
        """
        self.model.train()
        total_correct, total_samples = 0, 0

        with EnergyMonitor(self.model) as energy:
            with LatencyTimer() as timer:
                for epoch in range(self.epochs):
                    epoch_correct, epoch_total = 0, 0

                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        self.optimizer.zero_grad()
                        outputs = self.model(X_batch)
                        loss    = self.criterion(outputs, y_batch)
                        loss.backward()
                        self.optimizer.step()

                        c, t = compute_batch_accuracy(outputs, y_batch)
                        epoch_correct += c
                        epoch_total   += t

                    epoch_acc = epoch_correct / epoch_total * 100 if epoch_total else 0
                    logger.info(
                        f"[{self.edge_id}] Epoch {epoch+1}/{self.epochs} "
                        f"| Accuracy: {epoch_acc:.2f}%"
                    )
                    total_correct = epoch_correct
                    total_samples = epoch_total

        accuracy   = total_correct / total_samples * 100 if total_samples else 0
        latency_ms = timer.elapsed_ms
        energy_j   = energy.energy_j

        logger.info(
            f"[{self.edge_id}] Train done "
            f"| Acc={accuracy:.2f}% | Latency={latency_ms:.1f}ms | Energy={energy_j:.4f}J"
        )
        return accuracy, latency_ms, energy_j

    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate model on the fixed held-out test set. Returns accuracy %."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                c, t    = compute_batch_accuracy(outputs, y_batch)
                correct += c
                total   += t
        acc = correct / total * 100 if total else 0
        logger.info(f"[{self.edge_id}] Test Accuracy: {acc:.2f}%")
        return acc
