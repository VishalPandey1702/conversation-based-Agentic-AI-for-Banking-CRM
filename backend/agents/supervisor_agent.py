"""
Supervisor Agent.

The supervisor:
- owns the workflow lifecycle (start / step / end)
- decides the next workflow step (deterministic order)
- writes start/end audit rows
- never invokes data tools directly

This agent is consumed by the LangGraph workflow as a router node.
"""
from __future__ import annotations

from typing import Optional

from backend.agents.base import BaseAgent
from backend.utils.constants import AgentRole, LogStatus, WorkflowStep
from backend.workflows.state import WorkflowState


# The fixed, deterministic order in which workers run.
WORKFLOW_ORDER = [
    WorkflowStep.DISCOVERY.value,
    WorkflowStep.SCORING.value,
    WorkflowStep.RECOMMENDATION.value,
    WorkflowStep.OUTREACH.value,
    WorkflowStep.CAMPAIGN.value,
]


class SupervisorAgent(BaseAgent):
    """Orchestrates the workflow without doing any analytical work itself."""

    role = AgentRole.SUPERVISOR.value
    agent_name = "SupervisorAgent"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def begin(self, state: WorkflowState) -> WorkflowState:
        """Mark the workflow as started and write the opening audit row."""
        state["current_step"] = WORKFLOW_ORDER[0]
        self.append_log(
            state,
            step_name="workflow_start",
            status=LogStatus.STARTED.value,
            reasoning=(
                f"Supervisor started workflow run='{state['run_id']}', "
                f"query='{state.get('user_query','')[:120]}'."
            ),
        )
        return state

    def end(self, state: WorkflowState) -> WorkflowState:
        """Mark the workflow as complete and write the closing audit row."""
        state["current_step"] = None
        completed = state.get("completed_steps", [])
        n_camp = len(state.get("campaign_results", []))
        n_msg = len(state.get("generated_messages", []))
        n_scored = len(state.get("scored_customers", []))
        n_disc = len(state.get("discovered_customers", []))
        summary = state.get("summary") or (
            f"Discovered {n_disc} customers, scored {n_scored}, "
            f"generated {n_msg} messages, dispatched {n_camp} campaigns."
        )
        state["summary"] = summary

        status = LogStatus.SUCCESS.value if not state.get("errors") else LogStatus.FAILED.value
        self.append_log(
            state,
            step_name="workflow_end",
            status=status,
            reasoning=(
                f"Supervisor finished. Completed steps={completed}; errors={state.get('errors')}. "
                f"Summary: {summary}"
            ),
        )
        return state

    # ------------------------------------------------------------------
    # Routing helpers (used by the LangGraph router)
    # ------------------------------------------------------------------
    @staticmethod
    def next_step_after(step: str) -> Optional[str]:
        """Return the next workflow step after `step`, or None if last."""
        if step not in WORKFLOW_ORDER:
            return WORKFLOW_ORDER[0]
        idx = WORKFLOW_ORDER.index(step)
        if idx + 1 >= len(WORKFLOW_ORDER):
            return None
        return WORKFLOW_ORDER[idx + 1]
