"""
LangGraph workflow definition.

Builds a deterministic StateGraph:

    START -> supervisor_begin
          -> customer_discovery
          -> customer_scoring
          -> product_recommendation
          -> outreach_generation
          -> campaign_execution
          -> supervisor_end -> END

Each worker is a single-responsibility agent. The supervisor brackets the
workflow but never randomly routes between siblings - the order is fixed
to give us determinism, easy auditability, and predictable rerun semantics.
"""
from __future__ import annotations

from typing import Callable

from langgraph.graph import StateGraph, START, END

from backend.agents.campaign_agent import CampaignAgent
from backend.agents.customer_discovery_agent import CustomerDiscoveryAgent
from backend.agents.outreach_agent import OutreachAgent
from backend.agents.recommendation_agent import RecommendationAgent
from backend.agents.scoring_agent import ScoringAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.services.logging_service import get_logger
from backend.utils.constants import LogStatus, WorkflowStep
from backend.workflows.state import WorkflowState

logger = get_logger(__name__)


def _wrap_node(name: str, fn: Callable[[WorkflowState], WorkflowState]) -> Callable[[WorkflowState], WorkflowState]:
    """Return a node function that catches exceptions and records them on state."""

    def _node(state: WorkflowState) -> WorkflowState:
        state["current_step"] = name
        try:
            return fn(state)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Node '%s' failed: %s", name, exc)
            state.setdefault("errors", []).append(f"[{name}] {exc}")
            state.setdefault("logs", []).append(
                {
                    "agent_name": name,
                    "step_name": name,
                    "status": LogStatus.FAILED.value,
                    "reasoning": f"Unhandled exception in node '{name}'.",
                    "error": str(exc),
                    "duration_ms": 0.0,
                    "timestamp": "",
                }
            )
            return state

    return _node


def build_workflow_graph():
    """Compile the LangGraph workflow."""
    supervisor = SupervisorAgent()
    discovery = CustomerDiscoveryAgent()
    scoring = ScoringAgent()
    recommendation = RecommendationAgent()
    outreach = OutreachAgent()
    campaign = CampaignAgent()

    graph = StateGraph(WorkflowState)

    graph.add_node("supervisor_begin", _wrap_node("supervisor_begin", supervisor.begin))
    graph.add_node(WorkflowStep.DISCOVERY.value, _wrap_node(WorkflowStep.DISCOVERY.value, discovery.run))
    graph.add_node(WorkflowStep.SCORING.value, _wrap_node(WorkflowStep.SCORING.value, scoring.run))
    graph.add_node(WorkflowStep.RECOMMENDATION.value, _wrap_node(WorkflowStep.RECOMMENDATION.value, recommendation.run))
    graph.add_node(WorkflowStep.OUTREACH.value, _wrap_node(WorkflowStep.OUTREACH.value, outreach.run))
    graph.add_node(WorkflowStep.CAMPAIGN.value, _wrap_node(WorkflowStep.CAMPAIGN.value, campaign.run))
    graph.add_node("supervisor_end", _wrap_node("supervisor_end", supervisor.end))

    # Linear, deterministic edges - the supervisor controls the order.
    graph.add_edge(START, "supervisor_begin")
    graph.add_edge("supervisor_begin", WorkflowStep.DISCOVERY.value)
    graph.add_edge(WorkflowStep.DISCOVERY.value, WorkflowStep.SCORING.value)
    graph.add_edge(WorkflowStep.SCORING.value, WorkflowStep.RECOMMENDATION.value)
    graph.add_edge(WorkflowStep.RECOMMENDATION.value, WorkflowStep.OUTREACH.value)
    graph.add_edge(WorkflowStep.OUTREACH.value, WorkflowStep.CAMPAIGN.value)
    graph.add_edge(WorkflowStep.CAMPAIGN.value, "supervisor_end")
    graph.add_edge("supervisor_end", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled
