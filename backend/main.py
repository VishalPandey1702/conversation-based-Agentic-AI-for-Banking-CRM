"""
Backend entry-point.

Usage:
    python -m backend.main                    # boot FastAPI
    python -m backend.main --seed             # (re)seed the SQLite database
    python -m backend.main --seed --no-serve  # seed and exit
"""
from __future__ import annotations

import argparse

import uvicorn

from backend.database.seed_data import seed
from backend.services.logging_service import get_logger
from backend.utils.config import settings

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Banking CRM agentic backend")
    p.add_argument("--seed", action="store_true", help="(Re)seed the SQLite database before serving.")
    p.add_argument("--no-serve", action="store_true", help="Skip starting the API server.")
    p.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload for development.")
    p.add_argument("--host", default=settings.API_HOST)
    p.add_argument("--port", type=int, default=settings.API_PORT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed:
        logger.info("Seeding database...")
        counts = seed()
        logger.info("Seed complete: %s", counts)

    if args.no_serve:
        return

    logger.info("Starting FastAPI on %s:%s (reload=%s)", args.host, args.port, args.reload)
    uvicorn.run(
        "backend.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
