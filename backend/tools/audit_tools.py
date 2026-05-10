"""
Audit / logging tool.

Persists every agent / tool execution to the `agent_logs` table. The MCP
server uses this internally to record tool invocations, and agents call it
directly to record reasoning steps.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import desc

from backend.database.db import session_scope
from backend.database.models import AgentLog
from backend.services.logging_service import get_logger
from backend.utils.constants import LogStatus
from backend.utils.helpers import safe_json_dumps

logger = get_logger(__name__)


def log_agent_event(
    *,
    run_id: str,
    agent_name: str,
    status: str = LogStatus.SUCCESS.value,
    tool_name: Optional[str] = None,
    step_name: Optional[str] = None,
    reasoning: Optional[str] = None,
    input_payload: Any = None,
    output_payload: Any = None,
    error_message: Optional[str] = None,
    duration_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Insert an audit row.

    All `*_payload` arguments are JSON-stringified and truncated to keep
    rows compact. Returns the inserted row id.
    """
    try:
        with session_scope() as s:
            row = AgentLog(
                run_id=run_id,
                agent_name=agent_name,
                tool_name=tool_name,
                step_name=step_name,
                status=status,
                reasoning=reasoning,
                input_payload=safe_json_dumps(input_payload) if input_payload is not None else None,
                output_payload=safe_json_dumps(output_payload) if output_payload is not None else None,
                error_message=error_message,
                duration_ms=duration_ms,
                timestamp=datetime.utcnow(),
            )
            s.add(row)
            s.flush()
            return {
                "log_id": row.id,
                "run_id": run_id,
                "status": status,
                "timestamp": row.timestamp.isoformat() + "Z",
            }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Audit logging failed: %s", exc)
        return {"log_id": None, "status": "AUDIT_FAILED", "error": str(exc)}


def fetch_logs(
    *,
    run_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Fetch recent audit rows (newest-first)."""
    with session_scope() as s:
        q = s.query(AgentLog)
        if run_id:
            q = q.filter(AgentLog.run_id == run_id)
        if agent_name:
            q = q.filter(AgentLog.agent_name == agent_name)
        q = q.order_by(desc(AgentLog.timestamp)).limit(limit)
        rows = q.all()
        return [
            {
                "id": r.id,
                "run_id": r.run_id,
                "agent_name": r.agent_name,
                "tool_name": r.tool_name,
                "step_name": r.step_name,
                "status": r.status,
                "reasoning": r.reasoning,
                "input_payload": r.input_payload,
                "output_payload": r.output_payload,
                "error_message": r.error_message,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp.isoformat() + "Z" if r.timestamp else None,
            }
            for r in rows
        ]
