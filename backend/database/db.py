"""
Database connection and session management.

Provides:
- SQLAlchemy engine bound to SQLite (configurable via DATABASE_URL)
- Declarative Base for ORM models
- SessionLocal factory for short-lived sessions
- get_db dependency for FastAPI endpoints
- init_db helper to (re)create the schema
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.utils.config import settings

logger = logging.getLogger(__name__)

# SQLite needs check_same_thread=False to allow access across threads
# (FastAPI / Streamlit / agent threads).
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for non-FastAPI callers (agents/tools/scripts)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(drop_existing: bool = False) -> None:
    """
    Create all tables. If drop_existing is True, all tables are dropped first.

    Args:
        drop_existing: When True, recreate the schema from scratch.
    """
    # Import models so they register on the Base metadata
    from backend.database import models  # noqa: F401

    if drop_existing:
        logger.warning("Dropping all tables before recreating schema.")
        Base.metadata.drop_all(bind=engine)

    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized at %s", settings.DATABASE_URL)
