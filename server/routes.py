"""
server/routes.py
FastAPI route definitions for the federated server.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from config.logging_config import setup_logger
from server.model_manager import ModelManager
from server.aggregator import run_aggregation
from server.reputation import ReputationManager
from shared.serialization import deserialize_weights
from metrics.tracker import get_tracker

logger = setup_logger("server")
router = APIRouter()

# Shared state (injected from main.py via dependency or module-level singletons)
_model_manager: ModelManager       = None
_reputation:    ReputationManager  = None
_pending_updates: Dict[str, dict]  = {}   # edge_id → {weights, sample_count, accuracy}
_sample_counts:   Dict[str, int]   = {}
_expected_edges:  int              = 0


def init_router(model_manager: ModelManager, reputation: ReputationManager, num_edges: int):
    global _model_manager, _reputation, _expected_edges
    _model_manager  = model_manager
    _reputation     = reputation
    _expected_edges = num_edges


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class UpdatePayload(BaseModel):
    edge_id:      str
    weights:      Dict[str, Any]
    sample_count: int
    accuracy:     float
    latency_ms:   float = 0.0
    energy_j:     float = 0.0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "model_version": _model_manager.version}


@router.get("/global-model")
def get_global_model():
    """Edges call this to download the current global model."""
    return {
        "version": _model_manager.version,
        "weights": _model_manager.get_serialized_weights(),
    }


@router.post("/update")
def receive_update(payload: UpdatePayload):
    """Edges POST their local model weights after a training round."""
    global _pending_updates, _sample_counts

    if not _model_manager:
        raise HTTPException(status_code=503, detail="Server not initialised.")

    logger.info(
        f"Received update from {payload.edge_id} | "
        f"samples={payload.sample_count} | acc={payload.accuracy:.2f}%"
    )

    _pending_updates[payload.edge_id] = {
        "weights":      deserialize_weights(payload.weights),
        "sample_count": payload.sample_count,
        "accuracy":     payload.accuracy,
        "latency_ms":   payload.latency_ms,
        "energy_j":     payload.energy_j,
    }
    _sample_counts[payload.edge_id] = payload.sample_count

    # Aggregate once all expected edges have submitted
    if len(_pending_updates) >= _expected_edges:
        _trigger_aggregation()

    return {"status": "received", "pending": len(_pending_updates)}


@router.get("/weights")
def get_aggregation_weights():
    """Return latest aggregation weights for the dashboard."""
    tracker = get_tracker()
    if tracker.agg_weights:
        return {"weights": tracker.agg_weights[-1]}
    return {"weights": {}}


@router.get("/metrics")
def get_metrics():
    """Return full metrics history for the dashboard."""
    return get_tracker().load()


# ── Internal aggregation trigger ──────────────────────────────────────────────

def _trigger_aggregation():
    global _pending_updates

    logger.info(f"All {_expected_edges} edges submitted. Starting aggregation...")

    local_weights = {eid: u["weights"] for eid, u in _pending_updates.items()}
    reputations   = {}

    # Update reputation for each edge
    for eid, update in _pending_updates.items():
        score = _reputation.update(
            edge_id        = eid,
            local_weights  = update["weights"],
            global_weights = _model_manager.get_weights(),
            local_accuracy = update["accuracy"],
        )
        reputations[eid] = score

    # Aggregate
    aggregated, agg_weights = run_aggregation(
        local_weights = local_weights,
        sample_counts = _sample_counts,
        reputations   = reputations,
    )

    _model_manager.update_weights(aggregated)

    # Record metrics
    edge_metrics = {
        eid: {
            "accuracy":   u["accuracy"],
            "latency_ms": u["latency_ms"],
            "energy_j":   u["energy_j"],
        }
        for eid, u in _pending_updates.items()
    }

    tracker = get_tracker()
    tracker.record_round(
        round_num    = _model_manager.version,
        global_acc   = sum(u["accuracy"] for u in _pending_updates.values()) / len(_pending_updates),
        edge_metrics = edge_metrics,
        agg_weights  = agg_weights,
        reputations  = reputations,
    )

    logger.info(f"Aggregation complete. Global model v{_model_manager.version}")
    _pending_updates.clear()
