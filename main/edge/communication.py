"""
edge/communication.py
HTTP communication layer between edge nodes and the federated server.
"""

import requests
import torch
import time
from typing import Dict, Optional, Tuple

from config.loader import get_config
from config.logging_config import setup_logger
from shared.serialization import serialize_weights, deserialize_weights

logger = setup_logger("communication")


def _server_url() -> str:
    cfg = get_config()
    return f"http://{cfg['server']['host']}:{cfg['server']['port']}"


def send_update(
    edge_id:      str,
    weights:      Dict[str, torch.Tensor],
    round_num:    int,
    sample_count: int,
    accuracy:     float,
    train_accuracy: float = 0.0,
    test_accuracy:  float = 0.0,
    latency_ms:   float = 0.0,
    energy_j:     float = 0.0,
    timeout:      int   = 30,
) -> bool:
    url     = f"{_server_url()}/update"
    cfg = get_config()
    comm_cfg = cfg.get("security", {})
    max_retries = int(comm_cfg.get("update_max_retries", 4))
    retry_backoff_s = float(comm_cfg.get("update_retry_backoff_seconds", 1.0))
    payload = {
        "edge_id":      edge_id,
        "round_num":    int(round_num),
        "weights":      serialize_weights(weights),
        "sample_count": sample_count,
        "accuracy":     accuracy,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "latency_ms":   latency_ms,
        "energy_j":     energy_j,
        "signature":    "",
        "timestamp":    0,
    }
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 429 and attempt < max_retries:
                wait_s = retry_backoff_s * (2 ** attempt)
                logger.warning(
                    f"[{edge_id}] Update rate-limited (429). "
                    f"Retrying in {wait_s:.1f}s (attempt {attempt + 1}/{max_retries})."
                )
                time.sleep(wait_s)
                continue
            response.raise_for_status()
            body = response.json()
            if body.get("status") == "ignored" and body.get("reason") == "pending_round_in_progress" and attempt < max_retries:
                wait_s = retry_backoff_s * (2 ** attempt)
                logger.info(
                    f"[{edge_id}] Server still finishing earlier round "
                    f"(pending_round={body.get('pending_round')}). "
                    f"Retrying update in {wait_s:.1f}s."
                )
                time.sleep(wait_s)
                continue
            logger.info(f"[{edge_id}] Update sent → {body}")
            return True
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                wait_s = retry_backoff_s * (2 ** attempt)
                logger.warning(
                    f"[{edge_id}] Update send failed ({e}). "
                    f"Retrying in {wait_s:.1f}s (attempt {attempt + 1}/{max_retries})."
                )
                time.sleep(wait_s)
                continue
            logger.error(f"[{edge_id}] Failed to send update after retries: {e}")
            return False
    return False


def fetch_global_model(timeout: int = 30) -> Optional[Tuple[Dict[str, torch.Tensor], int]]:
    url = f"{_server_url()}/global-model"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data    = response.json()
        weights = deserialize_weights(data["weights"])
        version = data.get("version", -1)
        logger.info(f"Global model fetched (v{version})")
        return weights, version
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch global model: {e}")
        return None


def check_server_health(timeout: int = 5) -> bool:
    try:
        r = requests.get(f"{_server_url()}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
