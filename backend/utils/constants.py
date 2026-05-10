"""
Centralized constants used throughout the system.

Keeping these in one place makes role/permission/scoring changes easy
to audit and avoids magic strings scattered across agents and tools.
"""
from __future__ import annotations

from enum import Enum


# =====================================================
# Agent role identifiers (used by the MCP role manager)
# =====================================================
class AgentRole(str, Enum):
    SUPERVISOR = "supervisor_agent"
    DISCOVERY = "discovery_agent"
    SCORING = "scoring_agent"
    RECOMMENDATION = "recommendation_agent"
    OUTREACH = "outreach_agent"
    CAMPAIGN = "campaign_agent"


# =====================================================
# Tool identifiers (registered in the MCP server)
# =====================================================
class ToolName(str, Enum):
    # Customer DB tools
    FETCH_CUSTOMER_BY_ID = "fetch_customer_by_id"
    FETCH_HIGH_INCOME_CUSTOMERS = "fetch_high_income_customers"
    FETCH_RECENT_LOAN_INQUIRIES = "fetch_recent_loan_inquiries"
    FETCH_CUSTOMER_PROFILE = "fetch_customer_profile"

    # Transaction analysis tools
    ANALYZE_MONTHLY_SPENDING = "analyze_monthly_spending"
    ANALYZE_BALANCE_PATTERNS = "analyze_balance_patterns"
    ANALYZE_TRANSACTION_FREQUENCY = "analyze_transaction_frequency"

    # Scoring tool
    COMPUTE_CONVERSION_SCORE = "compute_conversion_score"

    # Recommendation tool
    RECOMMEND_PRODUCT = "recommend_product"

    # WhatsApp tool
    SEND_WHATSAPP_MESSAGE = "send_whatsapp_message"

    # Audit tool
    LOG_AGENT_EVENT = "log_agent_event"


# =====================================================
# Workflow step names (used by the LangGraph workflow)
# =====================================================
class WorkflowStep(str, Enum):
    DISCOVERY = "customer_discovery"
    SCORING = "customer_scoring"
    RECOMMENDATION = "product_recommendation"
    OUTREACH = "outreach_generation"
    CAMPAIGN = "campaign_execution"


# =====================================================
# Customer segments
# =====================================================
class CustomerSegment(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# =====================================================
# Product catalog (used by the recommendation tool)
# =====================================================
PRODUCT_CATALOG = [
    {
        "product_code": "PL_PRIME",
        "product_name": "Prime Personal Loan",
        "min_income": 1_500_000.0,
        "min_credit_score": 750,
        "max_amount": 2_500_000.0,
        "interest_rate": 9.75,
        "tenure_months": 60,
        "audience": "HIGH",
    },
    {
        "product_code": "PL_STANDARD",
        "product_name": "Standard Personal Loan",
        "min_income": 600_000.0,
        "min_credit_score": 680,
        "max_amount": 1_000_000.0,
        "interest_rate": 11.5,
        "tenure_months": 48,
        "audience": "MEDIUM",
    },
    {
        "product_code": "PL_LITE",
        "product_name": "Lite Personal Loan",
        "min_income": 300_000.0,
        "min_credit_score": 620,
        "max_amount": 400_000.0,
        "interest_rate": 13.5,
        "tenure_months": 36,
        "audience": "LOW",
    },
    {
        "product_code": "CC_PREMIUM",
        "product_name": "Premium Credit Card",
        "min_income": 1_200_000.0,
        "min_credit_score": 720,
        "max_amount": 500_000.0,
        "interest_rate": 0.0,
        "tenure_months": 12,
        "audience": "HIGH",
    },
]


# =====================================================
# Run / status enums
# =====================================================
class LogStatus(str, Enum):
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class CampaignStatus(str, Enum):
    SENT = "SENT"
    FAILED = "FAILED"
    PENDING = "PENDING"


# =====================================================
# UI / branding
# =====================================================
APP_NAME = "Agentic Banking CRM"
APP_TAGLINE = "Multi-Agent AI Workflow Platform for Relationship Managers"
