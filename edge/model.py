"""
edge/model.py
Re-exports the shared IoTClassifier so edge code has a local import path.
"""

from shared.model import IoTClassifier, build_model

__all__ = ["IoTClassifier", "build_model"]
