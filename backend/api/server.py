"""
FastAPI application factory.

Run via:
    uvicorn backend.api.server:app --reload
or:
    python -m backend.main
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.database.db import init_db
from backend.services.logging_service import get_logger
from backend.utils.constants import APP_NAME, APP_TAGLINE

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title=APP_NAME,
        description=APP_TAGLINE,
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.on_event("startup")
    def _on_startup() -> None:  # noqa: D401
        """Ensure the schema exists when the API boots."""
        init_db(drop_existing=False)
        logger.info("FastAPI startup complete - %s", APP_NAME)

    @app.get("/")
    def _index():
        return {
            "app": APP_NAME,
            "tagline": APP_TAGLINE,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
