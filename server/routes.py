"""
server/routes.py
FastAPI routes with security, multi-agent orchestration, and live classifier.
"""

import time
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from config.logging_config import setup_logger
from server.model_manager import ModelManager
from server.aggregator import run_aggregation
from server.reputation import ReputationManager
from shared.serialization import deserialize_weights
from metrics.tracker import get_tracker
from agents.orchestrator import get_orchestrator
from security.security_layer import (
    get_detector, get_rate_limiter, verify_signature, initialize_security
)

logger = setup_logger("server")
router = APIRouter()

_model_manager:   ModelManager      = None
_reputation:      ReputationManager = None
_pending_updates: Dict[str, dict]   = {}
_sample_counts:   Dict[str, int]    = {}
_expected_edges:  int               = 0
_edge_tokens:     Dict[str, str]    = {}
_security_log:    List[Dict]        = []


def init_router(model_manager, reputation, num_edges, edge_tokens=None):
    global _model_manager, _reputation, _expected_edges, _edge_tokens
    _model_manager  = model_manager
    _reputation     = reputation
    _expected_edges = num_edges
    _edge_tokens    = edge_tokens or {}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class UpdatePayload(BaseModel):
    edge_id:      str
    weights:      Dict[str, Any]
    sample_count: int
    accuracy:     float
    latency_ms:   float = 0.0
    energy_j:     float = 0.0
    signature:    str   = ""
    timestamp:    int   = 0


class ClassifyPayload(BaseModel):
    co:       float
    humidity: float
    light:    float
    lpg:      float
    motion:   float
    smoke:    float
    temp:     float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": _model_manager.version if _model_manager else 0,
        "security":      "active",
        "agents":        list(get_orchestrator().agents.keys()),
    }


@router.get("/global-model")
def get_global_model():
    return {
        "version": _model_manager.version,
        "weights": _model_manager.get_serialized_weights(),
    }


@router.post("/update")
def receive_update(payload: UpdatePayload):
    global _pending_updates, _sample_counts
    if not _model_manager:
        raise HTTPException(status_code=503, detail="Server not initialised.")

    edge_id = payload.edge_id

    if not get_rate_limiter().allow(edge_id):
        _log_sec(edge_id, "RATE_LIMIT", "Too many updates")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    hmac_valid = True
    if payload.signature and payload.timestamp:
        hmac_valid = verify_signature(edge_id, payload.signature, payload.timestamp)
        if not hmac_valid:
            _log_sec(edge_id, "HMAC_FAIL", "Signature mismatch")

    logger.info(
        f"Update from {edge_id} | samples={payload.sample_count} | "
        f"acc={payload.accuracy:.2f}% | hmac={'✓' if hmac_valid else '✗'}"
    )

    _pending_updates[edge_id] = {
        "weights":      deserialize_weights(payload.weights),
        "sample_count": payload.sample_count,
        "accuracy":     payload.accuracy,
        "latency_ms":   payload.latency_ms,
        "energy_j":     payload.energy_j,
        "hmac_valid":   hmac_valid,
    }
    _sample_counts[edge_id] = payload.sample_count

    if len(_pending_updates) >= _expected_edges:
        _trigger_aggregation()

    return {"status": "received", "pending": len(_pending_updates), "hmac_valid": hmac_valid}


@router.post("/classify")
def classify_sensor(payload: ClassifyPayload):
    """Live classification — accepts raw (unscaled) sensor values."""
    if not _model_manager:
        raise HTTPException(status_code=503, detail="Server not initialised.")

    features = torch.tensor([[
        payload.co, payload.humidity, payload.light,
        payload.lpg, payload.motion, payload.smoke, payload.temp,
    ]], dtype=torch.float32)

    model = _model_manager.model
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()
        pred   = int(torch.argmax(logits, dim=1).item())

    label_map = {0: "Normal", 1: "Warning", 2: "Critical"}
    label     = label_map[pred]

    reasons = []
    if payload.smoke > 0.10 or payload.co > 0.005:
        reasons.append("High smoke/CO levels detected → Fire risk")
    if payload.lpg > 0.007:
        reasons.append("Elevated LPG concentration → Gas leak risk")
    if payload.temp > 90.0:
        reasons.append("Temperature exceeds 90°F → Environmental anomaly")
    if not reasons:
        reasons.append("All sensor readings within safe thresholds")

    return {
        "prediction":    pred,
        "label":         label,
        "confidence":    round(max(probs) * 100, 2),
        "probabilities": {label_map[i]: round(p * 100, 2) for i, p in enumerate(probs)},
        "model_version": _model_manager.version,
        "reasons":       reasons,
    }


@router.get("/agents")
def get_agent_results():
    orch = get_orchestrator()
    if not orch.results_log:
        return {"status": "no_rounds_yet", "results": {}}
    latest    = orch.results_log[-1]
    round_num = latest.get("round", 0)
    return {
        "round":   round_num,
        "results": orch.get_serializable_results(round_num),
        "alerts":  orch.get_all_alerts(round_num),
    }


@router.get("/security")
def get_security_status():
    detector = get_detector()
    return {
        "events":       _security_log[-20:],
        "norm_history": {eid: h[-5:] for eid, h in detector.norm_history.items()},
        "active_edges": list(_pending_updates.keys()),
    }


@router.get("/weights")
def get_aggregation_weights():
    tracker = get_tracker()
    return {"weights": tracker.agg_weights[-1] if tracker.agg_weights else {}}


@router.get("/metrics")
def get_metrics():
    return get_tracker().load()


# ── Internal ──────────────────────────────────────────────────────────────────

def _log_sec(edge_id, etype, detail):
    _security_log.append({"edge_id": edge_id, "type": etype, "detail": detail, "timestamp": time.time()})
    logger.warning(f"[SECURITY] {edge_id} | {etype} | {detail}")


def _trigger_aggregation():
    global _pending_updates
    logger.info(f"All {_expected_edges} edges submitted — aggregating...")

    local_weights = {eid: u["weights"] for eid, u in _pending_updates.items()}

    # Poisoning detection
    detector = get_detector()
    suspicious, weight_norms = detector.detect(local_weights)
    hmac_failures = [eid for eid, u in _pending_updates.items() if not u.get("hmac_valid", True)]

    clean_weights = {eid: w for eid, w in local_weights.items() if eid not in suspicious}
    clean_counts  = {eid: v for eid, v in _sample_counts.items() if eid not in suspicious}

    if not clean_weights:
        logger.error("All edges flagged — aborting aggregation")
        _pending_updates.clear()
        return

    for eid in suspicious:
        _log_sec(eid, "POISONING_EXCLUDED", "norm z-score outlier")

    reputations = {}
    for eid, u in _pending_updates.items():
        if eid in suspicious:
            reputations[eid] = 0.05
            continue
        reputations[eid] = _reputation.update(
            edge_id=eid, local_weights=u["weights"],
            global_weights=_model_manager.get_weights(), local_accuracy=u["accuracy"],
        )

    aggregated, agg_weights = run_aggregation(
        local_weights=clean_weights,
        sample_counts=clean_counts,
        reputations={k: v for k, v in reputations.items() if k in clean_weights},
    )
    _model_manager.update_weights(aggregated)

    # Weighted global accuracy
    total_n      = sum(clean_counts.values()) or 1
    weighted_acc = sum(
        _pending_updates[eid]["accuracy"] * (clean_counts.get(eid, 0) / total_n)
        for eid in clean_weights
    )

    edge_metrics = {
        eid: {"accuracy": u["accuracy"], "latency_ms": u["latency_ms"], "energy_j": u["energy_j"]}
        for eid, u in _pending_updates.items()
    }

    tracker   = get_tracker()
    round_num = _model_manager.version
    tracker.record_round(
        round_num=round_num, global_acc=weighted_acc,
        edge_metrics=edge_metrics, agg_weights=agg_weights, reputations=reputations,
    )

    # Multi-agent analysis
    orch    = get_orchestrator()
    context = {
        "global_acc":    tracker.global_acc,
        "edge_acc":      dict(tracker.edge_acc),
        "weight_norms":  weight_norms,
        "reputations":   reputations,
        "hmac_failures": hmac_failures,
        "sensor_batch":  [],
    }
    agent_results = orch.run_round(context, round_num)
    sec_r = agent_results.get("security")
    if sec_r:
        for alert in sec_r.alerts:
            _security_log.append({"edge_id": "system", "type": "AGENT_ALERT", "detail": alert, "timestamp": time.time()})

    logger.info(f"Aggregation done. Global model v{round_num} | weighted_acc={weighted_acc:.2f}%")
    _pending_updates.clear()
