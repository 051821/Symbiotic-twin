"""
config/loader.py
Loads and provides global access to YAML configuration.
"""

import os
import yaml
from pathlib import Path

_config = None
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str = None) -> dict:
    """Load YAML config from file. Returns cached version after first load."""
    global _config
    if _config is not None:
        return _config

    config_path = Path(path) if path else _CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        _config = yaml.safe_load(f)

    return _config


def get_config() -> dict:
    """Return loaded config, loading it if necessary."""
    return load_config()


def reload_config(path: str = None) -> dict:
    """Force reload config from disk."""
    global _config
    _config = None
    return load_config(path)
