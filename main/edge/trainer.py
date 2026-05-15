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
torch.set_num_threads(2)


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
        model_cfg    = cfg.get("model", {})
        self.weight_decay    = model_cfg.get("weight_decay", 1e-4)
        self.label_smoothing = model_cfg.get("label_smoothing", 0.0)
        self.grad_clip_norm  = model_cfg.get("grad_clip_norm", 0.0)
        self.fedprox_mu      = float(model_cfg.get("fedprox_mu", 0.0) or 0.0)
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._global_reference = None

        self.model.to(self.device)

        # Use class-weighted loss to handle Normal >> Warning > Critical imbalance
        if class_weights is not None:
            cw = class_weights.to(self.device)
            logger.info(f"[{edge_id}] Using class weights: {cw.tolist()}")
        else:
            cw = None

        self.criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=self.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self._use_amp = self.device.type == "cuda"
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_amp)

    def set_global_reference(self, state_dict: dict) -> None:
        """
        Store a detached snapshot of global parameters for FedProx.
        Called once per round after syncing with the latest global model.
        """
        self._global_reference = {
            name: param.detach().to(self.device)
            for name, param in state_dict.items()
        }

    def update_class_weights(self, class_weights: torch.Tensor) -> None:
        """Update loss weights when the window shifts to a new data slice."""
        cw = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=self.label_smoothing)
        logger.info(f"[{self.edge_id}] Class weights updated: {cw.tolist()}")

    def train(self, train_loader: DataLoader, epochs_override: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Train for configured epochs. Returns (accuracy_%, latency_ms, energy_j).
        Each epoch processes one full pass over the current temporal window.
        """
        self.model.train()
        total_correct, total_samples = 0, 0

        epochs_to_run = max(1, int(epochs_override if epochs_override is not None else self.epochs))
        with EnergyMonitor(self.model) as energy:
            with LatencyTimer() as timer:
                for epoch in range(epochs_to_run):
                    epoch_correct, epoch_total = 0, 0

                    for X_batch, y_batch in train_loader:
                        X_batch = X_batch.to(self.device, non_blocking=True)
                        y_batch = y_batch.to(self.device, non_blocking=True)

                        self.optimizer.zero_grad(set_to_none=True)
                        if self._use_amp:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                outputs = self.model(X_batch)
                                loss = self.criterion(outputs, y_batch)
                                if self.fedprox_mu > 0.0 and self._global_reference is not None:
                                    prox_term = 0.0
                                    for name, param in self.model.named_parameters():
                                        g_ref = self._global_reference.get(name)
                                        if g_ref is not None:
                                            prox_term = prox_term + torch.sum((param - g_ref) ** 2)
                                    loss = loss + 0.5 * self.fedprox_mu * prox_term
                        else:
                            outputs = self.model(X_batch)
                            loss = self.criterion(outputs, y_batch)
                            if self.fedprox_mu > 0.0 and self._global_reference is not None:
                                prox_term = 0.0
                                for name, param in self.model.named_parameters():
                                    g_ref = self._global_reference.get(name)
                                    if g_ref is not None:
                                        prox_term = prox_term + torch.sum((param - g_ref) ** 2)
                                loss = loss + 0.5 * self.fedprox_mu * prox_term
                        if self._use_amp:
                            self._scaler.scale(loss).backward()
                            if self.grad_clip_norm and self.grad_clip_norm > 0:
                                self._scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            self._scaler.step(self.optimizer)
                            self._scaler.update()
                        else:
                            loss.backward()
                            if self.grad_clip_norm and self.grad_clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            self.optimizer.step()

                        c, t = compute_batch_accuracy(outputs, y_batch)
                        epoch_correct += c
                        epoch_total   += t

                    epoch_acc = epoch_correct / epoch_total * 100 if epoch_total else 0
                    logger.info(
                        f"[{self.edge_id}] Epoch {epoch+1}/{epochs_to_run} "
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

    def evaluate(self, test_loader: DataLoader, max_samples: Optional[int] = None) -> float:
        """Evaluate model on held-out test set (optionally capped). Returns accuracy %."""
        self.model.eval()
        correct, total = 0, 0
        max_samples = int(max_samples or 0)
        with torch.inference_mode():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                outputs = self.model(X_batch)
                c, t    = compute_batch_accuracy(outputs, y_batch)
                correct += c
                total   += t
                if max_samples > 0 and total >= max_samples:
                    break
        acc = correct / total * 100 if total else 0
        logger.info(f"[{self.edge_id}] Test Accuracy: {acc:.2f}%")
        return acc
