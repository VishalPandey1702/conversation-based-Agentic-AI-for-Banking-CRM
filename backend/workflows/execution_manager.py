"""
Execution manager.

Provides:
- run_full_workflow(): one-shot end-to-end run via the LangGraph
- run_single_step():   for the "rerun a step" UI flow
- in-memory cache of recent runs (run_id -> final state) used by the API

The manager owns ALL workflow lifecycle concerns so the FastAPI routes stay
thin.
"""
from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Optional

from backend.agents.campaign_agent import CampaignAgent
from backend.agents.customer_discovery_agent import CustomerDiscoveryAgent
from backend.agents.outreach_agent import OutreachAgent
from backend.agents.recommendation_agent import RecommendationAgent
from backend.agents.scoring_agent import ScoringAgent
from backend.agents.supervisor_agent import SupervisorAgent
from backend.services.logging_service import get_logger
from backend.utils.config import settings
from backend.utils.constants import WorkflowStep
from backend.utils.helpers import generate_run_id
from backend.workflows.graph import build_workflow_graph
from backend.workflows.state import WorkflowState, empty_state

logger = get_logger(__name__)


# Per-step agent registry for targeted re-runs.
_AGENT_FOR_STEP = {
    WorkflowStep.DISCOVERY.value: CustomerDiscoveryAgent,
    WorkflowStep.SCORING.value: ScoringAgent,
    WorkflowStep.RECOMMENDATION.value: RecommendationAgent,
    WorkflowStep.OUTREACH.value: OutreachAgent,
    WorkflowStep.CAMPAIGN.value: CampaignAgent,
}


class ExecutionManager:
    """Owns the compiled graph and a cache of recent run states."""

    def __init__(self) -> None:
        self._graph = None
        self._lock = Lock()
        self._runs: Dict[str, WorkflowState] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_full_workflow(
        self,
        *,
        user_query: str,
        rm_name: Optional[str] = None,
        top_n_customers: Optional[int] = None,
        min_conversion_threshold: Optional[float] = None,
        loan_type: str = "PERSONAL",
    ) -> WorkflowState:
        """End-to-end execution of the workflow."""
        run_id = generate_run_id("wf")
        state = empty_state(
            run_id=run_id,
            user_query=user_query,
            rm_name=rm_name,
            top_n_customers=top_n_customers or settings.DEFAULT_TOP_N_CUSTOMERS,
            min_conversion_threshold=(
                min_conversion_threshold
                if min_conversion_threshold is not None
                else settings.MIN_CONVERSION_THRESHOLD
            ),
            loan_type=loan_type,
        )

        graph = self._get_graph()
        logger.info("[run_full_workflow] run_id=%s query=%r", run_id, user_query)
        try:
            final_state: WorkflowState = graph.invoke(state)  # type: ignore[assignment]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Workflow crashed: %s", exc)
            state.setdefault("errors", []).append(f"workflow crashed: {exc}")
            final_state = state

        with self._lock:
            self._runs[run_id] = final_state
        return final_state

    def run_single_step(
        self,
        *,
        step_name: str,
        run_id: str,
        seed_state: Optional[WorkflowState] = None,
    ) -> WorkflowState:
        """
        Re-run a single step against an existing state.

        If `seed_state` is provided, it is used as the starting point;
        otherwise we look it up from the in-memory cache.
        """
        if step_name not in _AGENT_FOR_STEP:
            raise ValueError(f"Unknown step: {step_name}")

        with self._lock:
            state = seed_state or self._runs.get(run_id)
        if state is None:
            raise KeyError(f"No state cached for run_id={run_id}; cannot rerun '{step_name}'.")

        # Reset just this step's outputs so the rerun is idempotent
        _reset_step_output(state, step_name)
        if step_name in state.get("completed_steps", []):
            state["completed_steps"] = [s for s in state["completed_steps"] if s != step_name]

        agent = _AGENT_FOR_STEP[step_name]()
        new_state = agent.run(state)

        with self._lock:
            self._runs[run_id] = new_state
        return new_state

    def get_run(self, run_id: str) -> Optional[WorkflowState]:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> List[str]:
        with self._lock:
            return list(self._runs.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_graph(self):
        if self._graph is None:
            with self._lock:
                if self._graph is None:
                    self._graph = build_workflow_graph()
        return self._graph


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _reset_step_output(state: WorkflowState, step_name: str) -> None:
    """Clear the slice of state owned by a step so the rerun is clean."""
    mapping = {
        WorkflowStep.DISCOVERY.value: ["discovered_customers", "scored_customers", "recommendations", "generated_messages", "campaign_results"],
        WorkflowStep.SCORING.value: ["scored_customers", "recommendations", "generated_messages", "campaign_results"],
        WorkflowStep.RECOMMENDATION.value: ["recommendations", "generated_messages", "campaign_results"],
        WorkflowStep.OUTREACH.value: ["generated_messages", "campaign_results"],
        WorkflowStep.CAMPAIGN.value: ["campaign_results"],
    }
    for key in mapping.get(step_name, []):
        state[key] = []


# Module-level singleton consumed by the API
execution_manager = ExecutionManager()
