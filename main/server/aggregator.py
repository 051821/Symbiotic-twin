"""
server/aggregator.py
Implements FedAvg and adaptive weighted aggregation.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

from config.loader import get_config
from config.logging_config import setup_logger
from shared.serialization import weights_to_numpy, numpy_to_weights

logger = setup_logger("server")


def compute_fedavg_weights(sample_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute FedAvg contribution weights proportional to sample counts.

        w_i = n_i / sum(n_k)

    Args:
        sample_counts : {edge_id: num_training_samples}

    Returns:
        {edge_id: weight}  where weights sum to 1.0
    """
    total = sum(sample_counts.values())
    if total == 0:
        raise ValueError("Total sample count is zero — cannot compute weights.")
    return {eid: count / total for eid, count in sample_counts.items()}


def compute_adaptive_weights(
    sample_counts: Dict[str, int],
    reputations: Dict[str, float],
) -> Dict[str, float]:
    """
    Adaptive weighting: combines sample proportion with reputation score.

        score_i = (n_i / N) * reputation_i
        w_i     = score_i / sum(score_k)

    Args:
        sample_counts : {edge_id: num_training_samples}
        reputations   : {edge_id: reputation_score [0..1]}

    Returns:
        {edge_id: weight}  where weights sum to 1.0
    """
    total = sum(sample_counts.values())
    raw_scores = {
        eid: (count / total) * reputations.get(eid, 1.0)
        for eid, count in sample_counts.items()
    }
    score_sum = sum(raw_scores.values())
    if score_sum == 0:
        return compute_fedavg_weights(sample_counts)
    return {eid: score / score_sum for eid, score in raw_scores.items()}


def aggregate(
    local_weights: Dict[str, Dict[str, torch.Tensor]],
    agg_weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """
    Weighted aggregation of model parameters.

        θ_global = Σ w_i · θ_i

    Args:
        local_weights : {edge_id: state_dict}
        agg_weights   : {edge_id: float weight}

    Returns:
        Aggregated state_dict as Dict[str, torch.Tensor]
    """
    if not local_weights:
        raise ValueError("No local weights provided for aggregation.")

    edge_ids = list(local_weights.keys())
    ref_keys = list(local_weights[edge_ids[0]].keys())

    aggregated = {}
    for key in ref_keys:
        weighted_sum = None
        for edge_id in edge_ids:
            tensor    = local_weights[edge_id][key].float()
            weight    = agg_weights.get(edge_id, 0.0)
            weighted  = tensor * weight
            weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted
        aggregated[key] = weighted_sum

    logger.info(
        f"Aggregated {len(edge_ids)} edge models | "
        f"weights={{{', '.join(f'{k}: {v:.3f}' for k, v in agg_weights.items())}}}"
    )
    return aggregated


def run_aggregation(
    local_weights: Dict[str, Dict[str, torch.Tensor]],
    sample_counts: Dict[str, int],
    reputations: Dict[str, float] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Full aggregation pipeline — chooses strategy from config.

    Returns:
        (aggregated_state_dict, agg_weights)
    """
    cfg      = get_config()
    strategy = cfg["aggregation"]["strategy"]
    adaptive = cfg["aggregation"]["adaptive_weighting"]

    reputations = reputations or {eid: 1.0 for eid in sample_counts}

    if adaptive and strategy == "fedavg":
        agg_weights = compute_adaptive_weights(sample_counts, reputations)
    else:
        agg_weights = compute_fedavg_weights(sample_counts)

    aggregated = aggregate(local_weights, agg_weights)
    return aggregated, agg_weights
