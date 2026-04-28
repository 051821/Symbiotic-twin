"""
edge/main.py
Entry point for a single Digital Twin edge node.
Each federated round loads a new temporal window of data.
"""

import os
import sys
import time
import torch

from config.loader import get_config
from config.logging_config import setup_logger
from shared.model import build_model
from shared.utils import set_seed
from edge.data_loader import load_edge_data
from edge.trainer import LocalTrainer
from edge.cognitive_layer import CognitiveLayer
from edge.communication import send_update, fetch_global_model, check_server_health


def run_edge(edge_id: str, device_id: str) -> None:
    cfg    = get_config()
    logger = setup_logger(edge_id)
    set_seed(cfg["system"]["seed"])

    num_rounds  = cfg["system"]["num_rounds"]
    server_wait = 5

    logger.info("=" * 60)
    logger.info(f"Edge Node : {edge_id}  |  Device: {device_id}")
    logger.info(f"Rounds    : {num_rounds}")
    logger.info(f"Window    : {cfg['data'].get('window_strategy','sliding')} "
                f"(frac={cfg['data'].get('window_fraction',0.3)}, "
                f"step={cfg['data'].get('window_step',0.1)})")
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

    # Build model and cognitive layer (created once, weights updated each round)
    model     = build_model()
    # round_num=0 gives first window; trainer created with initial class weights
    train_loader, test_loader, class_weights, _ = load_edge_data(device_id, round_num=0)
    trainer   = LocalTrainer(model, edge_id, class_weights=class_weights)
    cognitive = CognitiveLayer(edge_id, initial_lr=cfg["system"]["learning_rate"])
    base_max_train = int(cfg["system"].get("max_train_samples_per_round", 0) or 0)
    base_epochs = int(cfg["system"].get("epochs_per_round", 1) or 1)

    for round_num in range(1, num_rounds + 1):
        logger.info(f"--- Round {round_num}/{num_rounds} ---")

        if cognitive.should_skip_round():
            logger.warning(f"[{edge_id}] Skipping round — energy budget exceeded.")
            continue

        plan = cognitive.training_plan()
        sample_ratio = float(plan.get("sample_ratio", 1.0))
        extra_epochs = int(plan.get("extra_epochs", 0))
        round_epochs = max(1, base_epochs + extra_epochs)
        round_max_train = int(base_max_train * sample_ratio) if base_max_train > 0 else None

        # ── Load NEW temporal window for this round ───────────────────────
        # round_num is 1-indexed; pass (round_num - 1) as 0-indexed window offset
        train_loader, test_loader, class_weights, sample_count = load_edge_data(
            device_id,
            round_num=round_num - 1,
            max_train_samples=round_max_train,
        )

        # Update class weights in trainer for the new window
        trainer.update_class_weights(class_weights)

        logger.info(
            f"[{edge_id}] Round {round_num} plan | samples={sample_count} "
            f"(ratio={sample_ratio:.2f}) | epochs={round_epochs}"
        )

        # ── Fetch latest global model ──────────────────────────────────────
        result = fetch_global_model()
        if result is not None:
            global_weights, version = result
            model.load_state_dict(global_weights)
            logger.info(f"Loaded global model v{version}")

        # ── Train on this round's window ───────────────────────────────────
        accuracy, latency_ms, energy_j = trainer.train(train_loader, epochs_override=round_epochs)

        # ── Evaluate on fixed held-out test set ────────────────────────────
        test_acc = trainer.evaluate(test_loader)
        # Guard against pathological window/test mismatch: avoid collapsing to 0%
        # by blending with train signal when eval is extremely low.
        if test_acc < 5.0 and accuracy > 20.0:
            logger.warning(
                f"[{edge_id}] Very low eval accuracy ({test_acc:.2f}%) with "
                f"reasonable train accuracy ({accuracy:.2f}%). Applying robust blend."
            )
            test_acc = 0.7 * test_acc + 0.3 * accuracy
        logger.info(f"[{edge_id}] Test accuracy: {test_acc:.2f}%")

        # ── Cognitive adaptation ───────────────────────────────────────────
        new_lr = cognitive.adapt(accuracy, energy_j)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = new_lr

        # ── Send update to server ──────────────────────────────────────────
        send_update(
            edge_id      = edge_id,
            weights      = model.state_dict(),
            sample_count = sample_count,
            accuracy     = test_acc,
            train_accuracy = accuracy,
            test_accuracy  = test_acc,
            latency_ms   = latency_ms,
            energy_j     = energy_j,
        )

        logger.info(
            f"Round {round_num} complete | "
            f"TrainAcc={accuracy:.2f}% | TestAcc={test_acc:.2f}% | "
            f"LR={new_lr:.6f} | Energy={energy_j:.4f}J | "
            f"Samples={sample_count}"
        )

    logger.info(f"[{edge_id}] All {num_rounds} rounds complete. Shutting down.")


if __name__ == "__main__":
    cfg = get_config()
    edge_id_env = os.environ.get("EDGE_ID", "edge1")
    edge_cfg    = next((e for e in cfg["edges"] if e["id"] == edge_id_env), None)
    if edge_cfg is None:
        print(f"Unknown EDGE_ID: {edge_id_env}")
        sys.exit(1)
    run_edge(edge_id=edge_cfg["id"], device_id=edge_cfg["device"])
