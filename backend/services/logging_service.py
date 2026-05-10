"""
Lightweight logging configuration used by the whole backend.

Configures root logging once at import-time and exposes a helper to fetch
a named logger with consistent formatting.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

from backend.utils.config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-32s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging exactly once."""
    global _configured
    if _configured:
        return
    log_level = (level or settings.LOG_LEVEL or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    # Quiet some noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger; ensures basicConfig is called."""
    configure_logging()
    return logging.getLogger(name)


# Auto-configure on import to ensure consistent output everywhere.
configure_logging()
