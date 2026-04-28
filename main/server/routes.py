"""
server/routes.py
FastAPI routes with security, multi-agent orchestration, and live classifier.
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agents.orchestrator import get_orchestrator
from config.loader import get_config
from config.logging_config import setup_logger
from metrics.tracker import get_tracker
from security.security_layer import get_detector, get_rate_limiter, verify_signature
from server.aggregator import run_aggregation
from server.model_manager import ModelManager
from server.reputation import ReputationManager
from shared.serialization import deserialize_weights

logger = setup_logger("server")
router = APIRouter()

_model_manager: ModelManager | None = None
_reputation: ReputationManager | None = None
_pending_updates: Dict[str, dict] = {}
_sample_counts: Dict[str, int] = {}
_expected_edges: int = 0
_edge_tokens: Dict[str, str] = {}
_security_log: List[Dict[str, Any]] = []
_scaler = None


class UpdatePayload(BaseModel):
    edge_id: str
    weights: Dict[str, Any]
    sample_count: int
    accuracy: float
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    latency_ms: float = 0.0
    energy_j: float = 0.0
    signature: str = ""
    timestamp: int = 0


class ClassifyPayload(BaseModel):
    co: float
    humidity: float
    light: float
    lpg: float
    motion: float
    smoke: float
    temp: float


def init_router(model_manager, reputation, num_edges, edge_tokens=None):
    global _model_manager, _reputation, _expected_edges, _edge_tokens
    _model_manager = model_manager
    _reputation = reputation
    _expected_edges = num_edges
    _edge_tokens = edge_tokens or {}


def _get_scaler():
    global _scaler
    if _scaler is None:
        cfg = get_config()
        scaler_path = Path(cfg["data"]["processed_path"]) / "scaler.pkl"
        if not scaler_path.exists():
            raise HTTPException(status_code=503, detail="Scaler not found. Run preprocessing first.")
        _scaler = joblib.load(scaler_path)
    return _scaler


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": _model_manager.version if _model_manager else 0,
        "security": "active",
        "agents": list(get_orchestrator().agents.keys()),
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

    effective_accuracy = payload.test_accuracy or payload.accuracy
    logger.info(
        f"Update from {edge_id} | samples={payload.sample_count} | "
        f"train={payload.train_accuracy:.2f}% | eval={effective_accuracy:.2f}% | "
        f"hmac={'ok' if hmac_valid else 'fail'}"
    )

    _pending_updates[edge_id] = {
        "weights": deserialize_weights(payload.weights),
        "sample_count": payload.sample_count,
        "accuracy": effective_accuracy,
        "train_accuracy": payload.train_accuracy,
        "test_accuracy": payload.test_accuracy,
        "latency_ms": payload.latency_ms,
        "energy_j": payload.energy_j,
        "hmac_valid": hmac_valid,
    }
    _sample_counts[edge_id] = payload.sample_count

    if len(_pending_updates) >= _expected_edges:
        _trigger_aggregation()

    return {"status": "received", "pending": len(_pending_updates), "hmac_valid": hmac_valid}


@router.post("/classify")
def classify_sensor(payload: ClassifyPayload):
    """Live classification using raw sensor values scaled with the training scaler."""
    if not _model_manager:
        raise HTTPException(status_code=503, detail="Server not initialised.")

    features = [[
        payload.co,
        payload.humidity,
        payload.light,
        payload.lpg,
        payload.motion,
        payload.smoke,
        payload.temp,
    ]]
    scaler = _get_scaler()
    scaled = torch.tensor(scaler.transform(features), dtype=torch.float32)

    model = _model_manager.model
    model.eval()
    with torch.no_grad():
        logits = model(scaled)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred = int(torch.argmax(logits, dim=1).item())

    label_map = {0: "Normal", 1: "Warning", 2: "Critical"}
    label = label_map[pred]

    smoke_or_co_alert = payload.smoke > 0.10 or payload.co > 0.005
    lpg_alert = payload.lpg > 0.007
    temp_alert = payload.temp > 90.0

    reasons = []
    if smoke_or_co_alert:
        reasons.append("Threshold rule triggered: high smoke/CO levels (fire risk).")
    if lpg_alert:
        reasons.append("Threshold rule triggered: elevated LPG concentration (gas leak risk).")
    if temp_alert:
        reasons.append("Threshold rule triggered: high temperature (>90F).")

    # If no hard threshold is crossed, explain that this is model-pattern driven.
    if not reasons:
        top2 = sorted(
            ((label_map[i], float(probs[i])) for i in range(len(probs))),
            key=lambda x: x[1],
            reverse=True,
        )[:2]
        primary_lbl, primary_p = top2[0]
        secondary_lbl, secondary_p = top2[1]
        reasons.append("No hard threshold rule triggered; prediction comes from learned multivariate pattern.")
        reasons.append(
            f"Model confidence: {primary_lbl} {primary_p * 100:.1f}% "
            f"(next: {secondary_lbl} {secondary_p * 100:.1f}%)."
        )

    return {
        "prediction": pred,
        "label": label,
        "confidence": round(max(probs) * 100, 2),
        "probabilities": {label_map[i]: round(p * 100, 2) for i, p in enumerate(probs)},
        "model_version": _model_manager.version,
        "reasons": reasons,
    }


@router.get("/agents")
def get_agent_results():
    orch = get_orchestrator()
    if not orch.results_log:
        return {"status": "no_rounds_yet", "results": {}}
    latest = orch.results_log[-1]
    round_num = latest.get("round", 0)
    return {
        "round": round_num,
        "results": orch.get_serializable_results(round_num),
        "alerts": orch.get_all_alerts(round_num),
    }


@router.get("/security")
def get_security_status():
    detector = get_detector()
    return {
        "events": _security_log[-20:],
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


def _log_sec(edge_id, etype, detail):
    _security_log.append({"edge_id": edge_id, "type": etype, "detail": detail, "timestamp": time.time()})
    logger.warning(f"[SECURITY] {edge_id} | {etype} | {detail}")


def _trigger_aggregation():
    global _pending_updates
    logger.info(f"All {_expected_edges} edges submitted - aggregating...")

    local_weights = {eid: u["weights"] for eid, u in _pending_updates.items()}

    detector = get_detector()
    suspicious, weight_norms = detector.detect(local_weights)
    hmac_failures = [eid for eid, u in _pending_updates.items() if not u.get("hmac_valid", True)]

    clean_weights = {eid: w for eid, w in local_weights.items() if eid not in suspicious}
    clean_counts = {eid: v for eid, v in _sample_counts.items() if eid not in suspicious}

    if not clean_weights:
        logger.error("All edges flagged - aborting aggregation")
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
            edge_id=eid,
            local_weights=u["weights"],
            global_weights=_model_manager.get_weights(),
            local_accuracy=u["accuracy"],
        )

    aggregated, agg_weights = run_aggregation(
        local_weights=clean_weights,
        sample_counts=clean_counts,
        reputations={k: v for k, v in reputations.items() if k in clean_weights},
    )
    _model_manager.update_weights(aggregated)

    total_n = sum(clean_counts.values()) or 1
    weighted_acc = sum(
        _pending_updates[eid]["accuracy"] * (clean_counts.get(eid, 0) / total_n)
        for eid in clean_weights
    )

    edge_metrics = {
        eid: {
            "accuracy": u["accuracy"],
            "latency_ms": u["latency_ms"],
            "energy_j": u["energy_j"],
        }
        for eid, u in _pending_updates.items()
    }

    tracker = get_tracker()
    round_num = _model_manager.version
    tracker.record_round(
        round_num=round_num,
        global_acc=weighted_acc,
        edge_metrics=edge_metrics,
        agg_weights=agg_weights,
        reputations=reputations,
    )

    orch = get_orchestrator()
    context = {
        "global_acc": tracker.global_acc,
        "edge_acc": dict(tracker.edge_acc),
        "weight_norms": weight_norms,
        "reputations": reputations,
        "hmac_failures": hmac_failures,
        "sensor_batch": [],
    }
    agent_results = orch.run_round(context, round_num)
    sec_r = agent_results.get("security")
    if sec_r:
        for alert in sec_r.alerts:
            _security_log.append(
                {
                    "edge_id": "system",
                    "type": "AGENT_ALERT",
                    "detail": alert,
                    "timestamp": time.time(),
                }
            )

    logger.info(f"Aggregation done. Global model v{round_num} | weighted_eval_acc={weighted_acc:.2f}%")
    _pending_updates.clear()
