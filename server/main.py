"""
server/main.py
FastAPI federated server entry point.
"""

import uvicorn
from fastapi import FastAPI

from config.loader import get_config
from config.logging_config import setup_logger
from server.model_manager import ModelManager
from server.reputation import ReputationManager
from server.routes import router, init_router
from shared.utils import set_seed

logger = setup_logger("server")


def create_app() -> FastAPI:
    cfg = get_config()
    set_seed(cfg["system"]["seed"])

    model_manager = ModelManager()
    reputation    = ReputationManager()

    edge_ids  = [e["id"] for e in cfg["edges"]]
    num_edges = cfg["system"]["num_edges"]

    reputation.initialize(edge_ids)
    init_router(model_manager, reputation, num_edges)

    app = FastAPI(title="SYMBIOTIC-TWIN Federated Server")
    app.include_router(router)

    @app.on_event("startup")
    async def on_startup():
        logger.info("=" * 60)
        logger.info("SYMBIOTIC-TWIN Federated Server starting")
        logger.info(f"Edges expected : {num_edges}")
        logger.info(f"Rounds         : {cfg['system']['num_rounds']}")
        logger.info(f"Aggregation    : {cfg['aggregation']['strategy']}")
        logger.info("=" * 60)

    return app


app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",        # ✅ MUST be 0.0.0.0 inside Docker
        port=cfg["server"]["port"],
        reload=False,
        log_level=cfg["logging"]["level"].lower(),
    )