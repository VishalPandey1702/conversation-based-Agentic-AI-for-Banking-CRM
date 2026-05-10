"""
Generic helpers shared across the codebase.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

logger = logging.getLogger(__name__)


def generate_run_id(prefix: str = "run") -> str:
    """Create a short, sortable run identifier."""
    return f"{prefix}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def now_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def safe_json_dumps(obj: Any, max_chars: int = 4000) -> str:
    """
    Dump anything to a JSON string for storage/logging.

    Falls back to repr() for objects that aren't JSON-serializable, and
    truncates to a configurable size to keep audit rows compact.
    """
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(obj)
    if len(s) > max_chars:
        s = s[:max_chars] + "...<truncated>"
    return s


@contextmanager
def timed_block(label: str = "block") -> Generator[dict, None, None]:
    """
    Context manager that measures elapsed time of a code block.

    Usage:
        with timed_block("scoring") as t:
            ...
        print(t["duration_ms"])
    """
    out: dict = {"duration_ms": 0.0}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["duration_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        logger.debug("[timed_block:%s] duration=%.2f ms", label, out["duration_ms"])


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a number to the given inclusive range."""
    return max(lo, min(hi, value))


def fmt_currency(value: float, symbol: str = "₹") -> str:
    """Format a number as currency with thousands separators."""
    try:
        return f"{symbol}{value:,.0f}"
    except (TypeError, ValueError):
        return f"{symbol}0"
