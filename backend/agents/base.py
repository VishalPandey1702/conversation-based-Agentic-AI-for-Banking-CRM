"""
Common base class shared by every specialized agent.

Provides:
- a typed reference to the MCP server
- an `invoke_tool` helper that injects the agent's role and run_id
- structured logging helpers (in-memory + database audit)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from backend.mcp.server import MCPServer, mcp_server
from backend.services.logging_service import get_logger
from backend.tools.audit_tools import log_agent_event
from backend.utils.constants import LogStatus
from backend.utils.helpers import safe_json_dumps
from backend.workflows.state import LogEntry, WorkflowState


class BaseAgent:
    """Base class for all agents."""

    role: str = "base"
    agent_name: str = "BaseAgent"

    def __init__(self, server: Optional[MCPServer] = None):
        self.mcp = server or mcp_server
        self.logger = get_logger(f"agent.{self.agent_name}")

    # ------------------------------------------------------------------
    # MCP integration
    # ------------------------------------------------------------------
    def invoke_tool(
        self,
        *,
        tool_name: str,
        params: Dict[str, Any],
        run_id: str,
    ) -> Dict[str, Any]:
        """
        Invoke an MCP tool under this agent's role.

        Returns the full MCP envelope ({ ok, result | error, ... }).
        """
        self.logger.debug(
            "[%s] invoking %s with %s", self.agent_name, tool_name, list(params.keys())
        )
        return self.mcp.invoke(
            role=self.role,
            tool_name=tool_name,
            params=params,
            run_id=run_id,
            agent_name=self.agent_name,
        )

    # ------------------------------------------------------------------
    # State + logging helpers
    # ------------------------------------------------------------------
    def append_log(
        self,
        state: WorkflowState,
        *,
        step_name: str,
        status: str,
        reasoning: str = "",
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        write_audit: bool = True,
    ) -> LogEntry:
        """
        Append a log entry to the in-memory state and write a DB audit row.
        """
        entry: LogEntry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent_name": self.agent_name,
            "step_name": step_name,
            "status": status,
            "reasoning": reasoning,
            "duration_ms": float(duration_ms),
            "error": error,
        }
        state.setdefault("logs", []).append(entry)

        if write_audit:
            try:
                log_agent_event(
                    run_id=state.get("run_id", "ad-hoc"),
                    agent_name=self.agent_name,
                    step_name=step_name,
                    status=status,
                    reasoning=reasoning,
                    duration_ms=duration_ms,
                    error_message=error,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Audit append failed: %s", exc)

        return entry

    def record_error(self, state: WorkflowState, message: str) -> None:
        state.setdefault("errors", []).append(f"[{self.agent_name}] {message}")
        self.logger.error("[%s] %s", self.agent_name, message)

    @staticmethod
    def shorten(payload: Any, max_chars: int = 200) -> str:
        """Convenience for compact log messages."""
        return safe_json_dumps(payload, max_chars=max_chars)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    def run(self, state: WorkflowState) -> WorkflowState:
        """Override in subclasses. Must return the (mutated) state."""
        raise NotImplementedError
