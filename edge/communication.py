"""
edge/communication.py
HTTP communication layer between edge nodes and the federated server.
"""

import requests
import torch
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
    sample_count: int,
    accuracy:     float,
    latency_ms:   float = 0.0,
    energy_j:     float = 0.0,
    timeout:      int   = 30,
) -> bool:
    """
    POST local model weights to the federated server.

    Returns:
        True if the server acknowledged successfully.
    """
    url     = f"{_server_url()}/update"
    payload = {
        "edge_id":      edge_id,
        "weights":      serialize_weights(weights),
        "sample_count": sample_count,
        "accuracy":     accuracy,
        "latency_ms":   latency_ms,
        "energy_j":     energy_j,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        logger.info(f"[{edge_id}] Update sent successfully → {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"[{edge_id}] Failed to send update: {e}")
        return False


def fetch_global_model(
    timeout: int = 30,
) -> Optional[Tuple[Dict[str, torch.Tensor], int]]:
    """
    GET the latest global model from the federated server.

    Returns:
        (state_dict, version) or None on failure.
    """
    url = f"{_server_url()}/global-model"

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data     = response.json()
        weights  = deserialize_weights(data["weights"])
        version  = data.get("version", -1)
        logger.info(f"Global model fetched (version {version})")
        return weights, version
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch global model: {e}")
        return None


def check_server_health(timeout: int = 5) -> bool:
    """Ping the server health endpoint. Returns True if reachable."""
    try:
        r = requests.get(f"{_server_url()}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False
