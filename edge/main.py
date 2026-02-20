"""
edge/main.py
Entry point for a single Digital Twin edge node.

Usage (called by Docker or directly):
    EDGE_ID=edge1 python -m edge.main
"""

import os
import time
import sys
import torch

from config.loader import get_config
from config.logging_config import setup_logger
from shared.model import build_model
from shared.utils import set_seed
from edge.data_loader import load_edge_data, get_sample_count
from edge.trainer import LocalTrainer
from edge.cognitive_layer import CognitiveLayer
from edge.communication import send_update, fetch_global_model, check_server_health


def run_edge(edge_id: str, device_id: str) -> None:
    cfg    = get_config()
    logger = setup_logger(edge_id)
    set_seed(cfg["system"]["seed"])

    num_rounds  = cfg["system"]["num_rounds"]
    server_wait = 5   # seconds to wait between server health retries

    logger.info("=" * 60)
    logger.info(f"Edge Node: {edge_id}  |  Device: {device_id}")
    logger.info(f"Federated rounds: {num_rounds}")
    logger.info("=" * 60)

    # Wait for server to be ready
    logger.info("Waiting for federated server...")
    for _ in range(20):
        if check_server_health():
            logger.info("Server is reachable.")
            break
        time.sleep(server_wait)
    else:
        logger.error("Server unreachable after retries. Exiting.")
        sys.exit(1)

    # Load data
    train_loader, test_loader = load_edge_data(device_id)
    sample_count              = get_sample_count(device_id)
    logger.info(f"Training samples: {sample_count}")

    # Build model and cognitive layer
    model     = build_model()
    trainer   = LocalTrainer(model, edge_id)
    cognitive = CognitiveLayer(edge_id, initial_lr=cfg["system"]["learning_rate"])

    for round_num in range(1, num_rounds + 1):
        logger.info(f"--- Round {round_num}/{num_rounds} ---")

        if cognitive.should_skip_round():
            logger.warning(f"[{edge_id}] Skipping round {round_num} — energy budget exceeded.")
            continue

        # Fetch latest global model
        result = fetch_global_model()
        if result is not None:
            global_weights, version = result
            model.load_state_dict(global_weights)
            logger.info(f"Loaded global model v{version}")

        # Local training
        accuracy, latency_ms, energy_j = trainer.train(train_loader)

        # Cognitive adaptation
        new_lr = cognitive.adapt(accuracy, energy_j)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = new_lr

        # Send update to server
        send_update(
            edge_id      = edge_id,
            weights      = model.state_dict(),
            sample_count = sample_count,
            accuracy     = accuracy,
            latency_ms   = latency_ms,
            energy_j     = energy_j,
        )

        logger.info(
            f"Round {round_num} done | Acc={accuracy:.2f}% | "
            f"LR={new_lr:.6f} | Energy={energy_j:.4f}J"
        )

    logger.info(f"[{edge_id}] All rounds complete.")


if __name__ == "__main__":
    cfg = get_config()

    # Read EDGE_ID from environment (set in docker-compose)
    edge_id_env = os.environ.get("EDGE_ID", "edge1")

    # Look up device MAC from config
    edge_cfg = next((e for e in cfg["edges"] if e["id"] == edge_id_env), None)
    if edge_cfg is None:
        print(f"Unknown EDGE_ID: {edge_id_env}")
        sys.exit(1)

    run_edge(edge_id=edge_cfg["id"], device_id=edge_cfg["device"])
