"""
Role-Based Access Control (RBAC) configuration for the MCP server.

Each agent role is granted access only to the tools it needs to perform
its single responsibility. The supervisor cannot directly invoke worker
tools - it orchestrates by calling agents (which then call their own
tools through the MCP).
"""
from __future__ import annotations

from typing import Dict, Set

from backend.utils.constants import AgentRole, ToolName


ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    # Supervisor only logs and reads logs - it never reaches into data tools.
    AgentRole.SUPERVISOR.value: {
        ToolName.LOG_AGENT_EVENT.value,
    },
    # Discovery agent: read customer + inquiry data
    AgentRole.DISCOVERY.value: {
        ToolName.FETCH_HIGH_INCOME_CUSTOMERS.value,
        ToolName.FETCH_RECENT_LOAN_INQUIRIES.value,
        ToolName.FETCH_CUSTOMER_BY_ID.value,
        ToolName.FETCH_CUSTOMER_PROFILE.value,
        ToolName.LOG_AGENT_EVENT.value,
    },
    # Scoring agent: needs profile + transaction analytics + scoring math
    AgentRole.SCORING.value: {
        ToolName.FETCH_CUSTOMER_PROFILE.value,
        ToolName.ANALYZE_MONTHLY_SPENDING.value,
        ToolName.ANALYZE_BALANCE_PATTERNS.value,
        ToolName.ANALYZE_TRANSACTION_FREQUENCY.value,
        ToolName.COMPUTE_CONVERSION_SCORE.value,
        ToolName.LOG_AGENT_EVENT.value,
    },
    # Recommendation agent: catalog selection only
    AgentRole.RECOMMENDATION.value: {
        ToolName.RECOMMEND_PRODUCT.value,
        ToolName.LOG_AGENT_EVENT.value,
    },
    # Outreach agent: needs customer profile to personalize
    AgentRole.OUTREACH.value: {
        ToolName.FETCH_CUSTOMER_PROFILE.value,
        ToolName.LOG_AGENT_EVENT.value,
    },
    # Campaign agent: just sends WhatsApp messages
    AgentRole.CAMPAIGN.value: {
        ToolName.SEND_WHATSAPP_MESSAGE.value,
        ToolName.LOG_AGENT_EVENT.value,
    },
}


def get_allowed_tools(role: str) -> Set[str]:
    """Return the set of tool names this role may invoke."""
    return ROLE_PERMISSIONS.get(role, set())


def is_allowed(role: str, tool_name: str) -> bool:
    """True iff the given role may invoke the given tool."""
    return tool_name in get_allowed_tools(role)
