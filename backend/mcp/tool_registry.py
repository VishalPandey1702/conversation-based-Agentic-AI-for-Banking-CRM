"""
Tool registry for the MCP server.

A `ToolSpec` is a strongly-typed wrapper around a callable. The registry
is populated at import-time with every tool the system knows about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.services.logging_service import get_logger

logger = get_logger(__name__)


@dataclass
class ToolSpec:
    """Metadata + handler for an MCP tool."""

    name: str
    description: str
    handler: Callable[..., Any]
    input_schema: Dict[str, str] = field(default_factory=dict)
    output_schema: Dict[str, str] = field(default_factory=dict)
    category: str = "general"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "category": self.category,
        }


class ToolRegistry:
    """In-memory registry of MCP tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            logger.warning("Overriding registered tool '%s'", spec.name)
        self._tools[spec.name] = spec
        logger.debug("Registered tool '%s' (category=%s)", spec.name, spec.category)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list(self) -> List[Dict[str, Any]]:
        """Return all tool metadata (does not expose handlers)."""
        return [t.metadata() for t in self._tools.values()]

    def names(self) -> List[str]:
        return sorted(self._tools.keys())
