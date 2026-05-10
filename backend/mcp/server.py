"""
In-process MCP Server.

Responsibilities:
- expose a registry of tools (with metadata)
- enforce role-based permissions on every invocation
- wrap every call with timing + audit logging
- return structured responses (success / error envelopes)

Although this is *in-process* (rather than a remote MCP), the architecture
mirrors the MCP protocol: tools have explicit schemas, roles are validated
independently of the agent that initiates the call, and audit is centralized.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from backend.mcp.role_manager import RoleManager, PermissionDeniedError
from backend.mcp.tool_registry import ToolRegistry, ToolSpec
from backend.services.logging_service import get_logger
from backend.tools import (
    audit_tools,
    customer_tools,
    recommendation_tools,
    scoring_tools,
    transaction_tools,
    whatsapp_tools,
)
from backend.utils.constants import LogStatus, ToolName

logger = get_logger(__name__)


# =====================================================
# Response envelopes
# =====================================================
def _ok(tool_name: str, result: Any, duration_ms: float) -> Dict[str, Any]:
    return {
        "ok": True,
        "tool": tool_name,
        "result": result,
        "duration_ms": duration_ms,
    }


def _err(tool_name: str, error: str, duration_ms: float, code: str = "ERROR") -> Dict[str, Any]:
    return {
        "ok": False,
        "tool": tool_name,
        "error": error,
        "code": code,
        "duration_ms": duration_ms,
    }


# =====================================================
# MCP Server
# =====================================================
class MCPServer:
    """In-process MCP-style server with RBAC + audit."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        role_manager: Optional[RoleManager] = None,
    ):
        self.registry = registry or ToolRegistry()
        self.role_manager = role_manager or RoleManager()
        self._register_default_tools()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def list_tools(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all tools, or only the ones available to a given role."""
        all_tools = self.registry.list()
        if role is None:
            return all_tools
        allowed = self.role_manager.allowed_tools(role)
        return [t for t in all_tools if t["name"] in allowed]

    def invoke(
        self,
        *,
        role: str,
        tool_name: str,
        params: Optional[Dict[str, Any]] = None,
        run_id: str = "ad-hoc",
        agent_name: Optional[str] = None,
        write_audit: bool = True,
    ) -> Dict[str, Any]:
        """
        Invoke a registered tool under a given role.

        Workflow:
            1. Permission check (raises PermissionDeniedError on failure).
            2. Tool lookup.
            3. Timed handler call.
            4. Audit log row written through the audit tool.
            5. Structured response envelope returned.
        """
        params = params or {}
        agent_name = agent_name or role
        t0 = time.perf_counter()

        # 1. Permission check
        try:
            self.role_manager.assert_can_invoke(role, tool_name)
        except PermissionDeniedError as exc:
            duration = round((time.perf_counter() - t0) * 1000.0, 2)
            if write_audit and tool_name != ToolName.LOG_AGENT_EVENT.value:
                audit_tools.log_agent_event(
                    run_id=run_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    status=LogStatus.FAILED.value,
                    error_message=str(exc),
                    duration_ms=duration,
                )
            return _err(tool_name, str(exc), duration, code="PERMISSION_DENIED")

        # 2. Lookup
        spec = self.registry.get(tool_name)
        if spec is None:
            duration = round((time.perf_counter() - t0) * 1000.0, 2)
            return _err(tool_name, f"tool '{tool_name}' is not registered", duration, "NOT_FOUND")

        # 3. Execute (timed)
        try:
            result = spec.handler(**params)
        except TypeError as exc:
            duration = round((time.perf_counter() - t0) * 1000.0, 2)
            error_msg = f"Bad parameters for '{tool_name}': {exc}"
            logger.error(error_msg)
            if write_audit and tool_name != ToolName.LOG_AGENT_EVENT.value:
                audit_tools.log_agent_event(
                    run_id=run_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    status=LogStatus.FAILED.value,
                    input_payload=params,
                    error_message=error_msg,
                    duration_ms=duration,
                )
            return _err(tool_name, error_msg, duration, "BAD_PARAMS")
        except Exception as exc:  # noqa: BLE001
            duration = round((time.perf_counter() - t0) * 1000.0, 2)
            logger.exception("Tool '%s' raised: %s", tool_name, exc)
            if write_audit and tool_name != ToolName.LOG_AGENT_EVENT.value:
                audit_tools.log_agent_event(
                    run_id=run_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    status=LogStatus.FAILED.value,
                    input_payload=params,
                    error_message=str(exc),
                    duration_ms=duration,
                )
            return _err(tool_name, str(exc), duration, "TOOL_ERROR")

        duration = round((time.perf_counter() - t0) * 1000.0, 2)

        # 4. Audit success (skip recursion when the tool is the audit logger)
        if write_audit and tool_name != ToolName.LOG_AGENT_EVENT.value:
            audit_tools.log_agent_event(
                run_id=run_id,
                agent_name=agent_name,
                tool_name=tool_name,
                status=LogStatus.SUCCESS.value,
                input_payload=params,
                output_payload=result,
                duration_ms=duration,
            )

        return _ok(tool_name, result, duration)

    # -----------------------------------------------------
    # Internals
    # -----------------------------------------------------
    def _register_default_tools(self) -> None:
        """Register the canonical set of tools shipped with the system."""
        # Customer tools
        self.registry.register(
            ToolSpec(
                name=ToolName.FETCH_CUSTOMER_BY_ID.value,
                description="Fetch a single customer summary by id.",
                handler=customer_tools.fetch_customer_by_id,
                input_schema={"customer_id": "int"},
                output_schema={"customer": "CustomerSummary | null"},
                category="customer_db",
            )
        )
        self.registry.register(
            ToolSpec(
                name=ToolName.FETCH_HIGH_INCOME_CUSTOMERS.value,
                description="Fetch customers above the income/credit thresholds.",
                handler=customer_tools.fetch_high_income_customers,
                input_schema={
                    "min_annual_income": "float",
                    "min_credit_score": "int",
                    "only_segments": "list[str] | null",
                    "limit": "int",
                },
                output_schema={"customers": "list[CustomerSummary]"},
                category="customer_db",
            )
        )
        self.registry.register(
            ToolSpec(
                name=ToolName.FETCH_RECENT_LOAN_INQUIRIES.value,
                description="List recent loan inquiries within an N-day window.",
                handler=customer_tools.fetch_recent_loan_inquiries,
                input_schema={"days": "int", "loan_type": "str | null", "customer_ids": "list[int] | null"},
                output_schema={"inquiries": "list[LoanInquiry]"},
                category="customer_db",
            )
        )
        self.registry.register(
            ToolSpec(
                name=ToolName.FETCH_CUSTOMER_PROFILE.value,
                description="360° view: customer + recent inquiries + interactions + 90-day txn count.",
                handler=customer_tools.fetch_customer_profile,
                input_schema={"customer_id": "int"},
                output_schema={"profile": "CustomerProfile | null"},
                category="customer_db",
            )
        )

        # Transaction analytics
        self.registry.register(
            ToolSpec(
                name=ToolName.ANALYZE_MONTHLY_SPENDING.value,
                description="Aggregate monthly spend + category breakdown.",
                handler=transaction_tools.analyze_monthly_spending,
                input_schema={"customer_id": "int", "months": "int"},
                output_schema={"summary": "MonthlySpendSummary"},
                category="analytics",
            )
        )
        self.registry.register(
            ToolSpec(
                name=ToolName.ANALYZE_BALANCE_PATTERNS.value,
                description="Balance volatility / trend analysis.",
                handler=transaction_tools.analyze_balance_patterns,
                input_schema={"customer_id": "int", "days": "int"},
                output_schema={"summary": "BalancePatternsSummary"},
                category="analytics",
            )
        )
        self.registry.register(
            ToolSpec(
                name=ToolName.ANALYZE_TRANSACTION_FREQUENCY.value,
                description="Per-month transaction count + consistency score.",
                handler=transaction_tools.analyze_transaction_frequency,
                input_schema={"customer_id": "int", "days": "int"},
                output_schema={"summary": "TxnFrequencySummary"},
                category="analytics",
            )
        )

        # Scoring
        self.registry.register(
            ToolSpec(
                name=ToolName.COMPUTE_CONVERSION_SCORE.value,
                description="Deterministic, explainable conversion score.",
                handler=scoring_tools.compute_conversion_score,
                input_schema={
                    "customer": "CustomerSummary",
                    "txn_consistency": "float",
                    "has_recent_inquiry": "bool",
                    "inquiry_recency_days": "int | null",
                },
                output_schema={"score": "ConversionScore"},
                category="scoring",
            )
        )

        # Recommendation
        self.registry.register(
            ToolSpec(
                name=ToolName.RECOMMEND_PRODUCT.value,
                description="Choose the best-fit product for a customer.",
                handler=recommendation_tools.recommend_product,
                input_schema={
                    "customer": "CustomerSummary",
                    "conversion_probability": "float",
                    "preferred_audience": "str | null",
                },
                output_schema={"recommendation": "ProductRecommendation"},
                category="recommendation",
            )
        )

        # WhatsApp
        self.registry.register(
            ToolSpec(
                name=ToolName.SEND_WHATSAPP_MESSAGE.value,
                description="Simulate sending a WhatsApp message and persist a campaign record.",
                handler=whatsapp_tools.send_whatsapp_message,
                input_schema={
                    "customer_id": "int",
                    "phone": "str",
                    "message": "str",
                    "campaign_run_id": "str | null",
                },
                output_schema={"result": "CampaignReceipt"},
                category="outreach",
            )
        )

        # Audit
        self.registry.register(
            ToolSpec(
                name=ToolName.LOG_AGENT_EVENT.value,
                description="Append an audit row capturing reasoning, IO, and timing.",
                handler=audit_tools.log_agent_event,
                input_schema={
                    "run_id": "str",
                    "agent_name": "str",
                    "tool_name": "str | null",
                    "step_name": "str | null",
                    "status": "str",
                    "reasoning": "str | null",
                    "input_payload": "any",
                    "output_payload": "any",
                    "error_message": "str | null",
                    "duration_ms": "float",
                },
                output_schema={"audit": "AuditReceipt"},
                category="audit",
            )
        )


# A single, shared MCP server instance for the whole process
mcp_server = MCPServer()
