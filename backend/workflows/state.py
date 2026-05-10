"""
Centralized workflow state.

The state is a TypedDict (LangGraph-friendly) that flows through every
node. Each agent reads what it needs and *adds* its results, never silently
mutating other agents' fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class LogEntry(TypedDict, total=False):
    """In-memory log entry maintained alongside the database audit."""
    timestamp: str
    agent_name: str
    step_name: str
    status: str
    reasoning: str
    duration_ms: float
    error: Optional[str]


class DiscoveredCustomer(TypedDict, total=False):
    """Lightweight view of a customer surfaced by discovery."""
    id: int
    customer_code: str
    full_name: str
    email: str
    phone: str
    annual_income: float
    monthly_salary: float
    account_balance: float
    credit_score: int
    customer_segment: str
    has_existing_loan: bool
    has_recent_inquiry: bool
    most_recent_inquiry: Optional[Dict[str, Any]]
    discovery_reason: str


class ScoredCustomer(TypedDict, total=False):
    """Output of the scoring agent."""
    customer: DiscoveredCustomer
    score: float
    conversion_probability: float
    features: Dict[str, Any]
    rationale: str
    txn_consistency: float
    above_threshold: bool


class RecommendedCustomer(TypedDict, total=False):
    """Output of the recommendation agent."""
    customer: DiscoveredCustomer
    score: float
    conversion_probability: float
    rationale_score: str
    recommendation: Dict[str, Any]


class OutreachMessage(TypedDict, total=False):
    """Output of the outreach agent."""
    customer_id: int
    customer_code: str
    full_name: str
    phone: str
    message: str
    product_code: Optional[str]
    product_name: Optional[str]
    rationale: str


class CampaignResult(TypedDict, total=False):
    """Output of the campaign execution agent."""
    customer_id: int
    customer_code: str
    full_name: str
    phone: str
    status: str
    sent_at: str
    campaign_id: Optional[int]
    message: str


class WorkflowState(TypedDict, total=False):
    """The full state object that flows through the LangGraph."""
    # Inputs
    run_id: str
    user_query: str
    rm_name: Optional[str]

    # Knobs
    top_n_customers: int
    min_conversion_threshold: float
    loan_type: str

    # Per-agent outputs (always present, may be empty lists)
    discovered_customers: List[DiscoveredCustomer]
    scored_customers: List[ScoredCustomer]
    recommendations: List[RecommendedCustomer]
    generated_messages: List[OutreachMessage]
    campaign_results: List[CampaignResult]

    # Bookkeeping
    logs: List[LogEntry]
    errors: List[str]
    completed_steps: List[str]
    current_step: Optional[str]
    summary: Optional[str]


def empty_state(
    *,
    run_id: str,
    user_query: str,
    rm_name: Optional[str] = None,
    top_n_customers: int = 10,
    min_conversion_threshold: float = 0.55,
    loan_type: str = "PERSONAL",
) -> WorkflowState:
    """Construct a fresh, empty state with sensible defaults."""
    return WorkflowState(
        run_id=run_id,
        user_query=user_query,
        rm_name=rm_name,
        top_n_customers=top_n_customers,
        min_conversion_threshold=min_conversion_threshold,
        loan_type=loan_type,
        discovered_customers=[],
        scored_customers=[],
        recommendations=[],
        generated_messages=[],
        campaign_results=[],
        logs=[],
        errors=[],
        completed_steps=[],
        current_step=None,
        summary=None,
    )
