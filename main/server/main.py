"""
server/main.py
FastAPI federated server entry point with security initialization.
"""

import uvicorn
from fastapi import FastAPI

from config.loader import get_config
from config.logging_config import setup_logger
from server.model_manager import ModelManager
from server.reputation import ReputationManager
from server.routes import router, init_router
from shared.utils import set_seed
from security.security_layer import initialize_security

logger = setup_logger("server")


def create_app() -> FastAPI:
    cfg = get_config()
    set_seed(cfg["system"]["seed"])

    model_manager = ModelManager()
    reputation    = ReputationManager()
    edge_ids      = [e["id"] for e in cfg["edges"]]
    num_edges     = cfg["system"]["num_edges"]

    reputation.initialize(edge_ids)
    edge_tokens = initialize_security(edge_ids)
    init_router(model_manager, reputation, num_edges, edge_tokens)

    app = FastAPI(
        title       = "SYMBIOTIC-TWIN Federated Server",
        description = "Secure Multi-Agent Federated Learning for IoT Digital Twins",
        version     = "2.0.0",
    )
    app.include_router(router)

    @app.on_event("startup")
    async def on_startup():
        strategy = cfg["data"].get("window_strategy", "sliding")
        logger.info("=" * 60)
        logger.info("SYMBIOTIC-TWIN Federated Server v2.0")
        logger.info(f"Edges          : {num_edges}")
        logger.info(f"Rounds         : {cfg['system']['num_rounds']}")
        logger.info(f"Aggregation    : {cfg['aggregation']['strategy']}")
        logger.info(f"Data window    : {strategy}")
        logger.info(f"Security       : HMAC + JWT + Poisoning Detector ACTIVE")
        logger.info(f"Agents         : Analyst, Anomaly, Predictor, Security")
        logger.info("=" * 60)

    return app


app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "server.main:app",
        host      = "0.0.0.0",
        port      = cfg["server"]["port"],
        reload    = False,
        log_level = cfg["logging"]["level"].lower(),
    )
