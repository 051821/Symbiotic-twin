"""
config/logging_config.py
Centralized logging setup. Each node gets its own log file.
"""

import logging
from pathlib import Path
from config.loader import get_config


def setup_logger(name: str) -> logging.Logger:
    config   = get_config()
    log_cfg  = config.get("logging", {})
    log_level  = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_dir    = Path(log_cfg.get("log_dir", "logs/"))
    log_format = log_cfg.get("format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    date_fmt   = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(fmt=log_format, datefmt=date_fmt)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
