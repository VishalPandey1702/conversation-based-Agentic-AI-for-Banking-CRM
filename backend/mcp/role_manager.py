"""
Role manager - enforces RBAC for tool invocations.

The MCP server delegates permission checks to RoleManager.assert_can_invoke().
This is intentionally a small object so it is easy to mock in tests.
"""
from __future__ import annotations

from typing import Set

from backend.mcp.permissions import ROLE_PERMISSIONS, is_allowed
from backend.services.logging_service import get_logger

logger = get_logger(__name__)


class PermissionDeniedError(PermissionError):
    """Raised when a role attempts to invoke a tool it cannot use."""


class RoleManager:
    """In-process RBAC enforcer."""

    def __init__(self, role_permissions: dict[str, Set[str]] | None = None):
        self._permissions = role_permissions or ROLE_PERMISSIONS

    def known_roles(self) -> list[str]:
        return sorted(self._permissions.keys())

    def allowed_tools(self, role: str) -> Set[str]:
        return set(self._permissions.get(role, set()))

    def assert_can_invoke(self, role: str, tool_name: str) -> None:
        """Raise PermissionDeniedError if the role can't use this tool."""
        if not is_allowed(role, tool_name):
            allowed = ", ".join(sorted(self.allowed_tools(role))) or "<none>"
            msg = (
                f"Role '{role}' is not permitted to invoke tool '{tool_name}'. "
                f"Allowed tools: [{allowed}]."
            )
            logger.warning(msg)
            raise PermissionDeniedError(msg)
