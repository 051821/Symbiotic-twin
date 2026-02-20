"""
server/reputation.py
Tracks per-edge trust/reputation scores based on model contribution quality.
Score decays if an edge's update diverges significantly from the global model.
"""

import numpy as np
import torch
from typing import Dict
from collections import defaultdict

from config.loader import get_config
from config.logging_config import setup_logger
from shared.utils import flatten_weights

logger = setup_logger("server")


class ReputationManager:
    """Maintains and updates reputation scores for all edge nodes."""

    def __init__(self):
        cfg = get_config()
        self.min_score  = cfg["aggregation"].get("min_reputation", 0.1)
        self.scores: Dict[str, float] = {}
        self.history: Dict[str, list] = defaultdict(list)

    def initialize(self, edge_ids: list) -> None:
        """Set all edges to a neutral starting score of 1.0."""
        for eid in edge_ids:
            self.scores[eid] = 1.0
        logger.info(f"Reputation initialized for: {edge_ids}")

    def update(
        self,
        edge_id: str,
        local_weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        local_accuracy: float,
    ) -> float:
        """
        Update reputation score for one edge based on:
          1. Cosine similarity between local and global weight vectors.
          2. Local accuracy as a quality signal.

        Args:
            edge_id        : Edge identifier.
            local_weights  : Weights submitted by this edge.
            global_weights : Current global model weights.
            local_accuracy : Training accuracy reported by the edge (0–100).

        Returns:
            Updated reputation score.
        """
        # Cosine similarity between flattened weight vectors
        local_flat  = flatten_weights(local_weights).astype(np.float64)
        global_flat = flatten_weights(global_weights).astype(np.float64)

        norm_l = np.linalg.norm(local_flat)
        norm_g = np.linalg.norm(global_flat)

        if norm_l == 0 or norm_g == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(local_flat, global_flat) / (norm_l * norm_g))
            similarity = max(0.0, similarity)   # clip to [0, 1]

        acc_factor = local_accuracy / 100.0     # normalise to [0, 1]

        # Combined quality signal (equal weight)
        quality = 0.5 * similarity + 0.5 * acc_factor

        # Exponential moving average (α=0.3)
        alpha = 0.3
        prev  = self.scores.get(edge_id, 1.0)
        new_score = alpha * quality + (1 - alpha) * prev
        new_score = max(self.min_score, new_score)

        self.scores[edge_id] = new_score
        self.history[edge_id].append(round(new_score, 4))

        logger.info(
            f"Reputation [{edge_id}] similarity={similarity:.4f} "
            f"acc_factor={acc_factor:.4f} → score={new_score:.4f}"
        )
        return new_score

    def get_scores(self) -> Dict[str, float]:
        return dict(self.scores)

    def get_history(self, edge_id: str) -> list:
        return self.history.get(edge_id, [])
