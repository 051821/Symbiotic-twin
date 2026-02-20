"""
server/utils.py
Server-side helper functions.
"""

from typing import Dict
from config.loader import get_config


def get_edge_sample_counts() -> Dict[str, int]:
    """
    Return a placeholder sample-count dict for all configured edges.
    In production these are reported by edges during registration.
    """
    cfg = get_config()
    # Default: equal counts until edges report actual values
    return {edge["id"]: 1 for edge in cfg["edges"]}


def edge_ids_from_config() -> list:
    """Return list of edge IDs from config."""
    return [e["id"] for e in get_config()["edges"]]


def validate_update_payload(payload: dict) -> bool:
    """Check that an edge update payload has required fields."""
    required = {"edge_id", "weights", "sample_count", "accuracy"}
    return required.issubset(payload.keys())
