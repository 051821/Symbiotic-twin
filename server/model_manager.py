"""
server/model_manager.py
Manages the global model: initialisation, versioning, and state sync.
"""

import torch
from pathlib import Path
from typing import Dict, Optional

from config.loader import get_config
from config.logging_config import setup_logger
from shared.model import build_model, IoTClassifier
from shared.serialization import serialize_weights, deserialize_weights
from shared.utils import ensure_dir

logger = setup_logger("server")

_CHECKPOINTS_DIR = Path("logs/checkpoints/")


class ModelManager:
    """Holds the global model and handles versioning."""

    def __init__(self):
        ensure_dir(_CHECKPOINTS_DIR)
        self.model:   IoTClassifier       = build_model()
        self.version: int                 = 0
        logger.info("Global model initialised (version 0)")

    # ------------------------------------------------------------------
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Return current global model state_dict."""
        return self.model.state_dict()

    def get_serialized_weights(self) -> Dict:
        """Return JSON-serializable weights for HTTP transport."""
        return serialize_weights(self.model.state_dict())

    # ------------------------------------------------------------------
    def update_weights(self, aggregated: Dict[str, torch.Tensor]) -> None:
        """Load new aggregated weights into the global model and bump version."""
        self.model.load_state_dict(aggregated)
        self.version += 1
        logger.info(f"Global model updated → version {self.version}")
        self._save_checkpoint()

    def load_weights_from_dict(self, weights_dict: Dict) -> None:
        """Accept deserialized weights (e.g. from HTTP POST) and load them."""
        state_dict = deserialize_weights(weights_dict)
        self.model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    def _save_checkpoint(self) -> None:
        path = _CHECKPOINTS_DIR / f"global_v{self.version}.pt"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, version: Optional[int] = None) -> bool:
        """Load a specific checkpoint version (latest if None)."""
        if version is None:
            checkpoints = sorted(_CHECKPOINTS_DIR.glob("global_v*.pt"))
            if not checkpoints:
                logger.warning("No checkpoints found.")
                return False
            path = checkpoints[-1]
        else:
            path = _CHECKPOINTS_DIR / f"global_v{version}.pt"

        if not path.exists():
            logger.error(f"Checkpoint not found: {path}")
            return False

        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Checkpoint loaded: {path}")
        return True
