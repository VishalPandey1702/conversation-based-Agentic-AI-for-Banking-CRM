"""
Recommendation Agent.

For each scored customer above the threshold, pick the most appropriate
product from the catalog. Uses only the recommendation tool (not the LLM).
"""
from __future__ import annotations

from typing import List

from backend.agents.base import BaseAgent
from backend.utils.constants import AgentRole, LogStatus, ToolName, WorkflowStep
from backend.utils.helpers import timed_block
from backend.workflows.state import RecommendedCustomer, WorkflowState


class RecommendationAgent(BaseAgent):
    """Selects the best-fit product for each high-scoring customer."""

    role = AgentRole.RECOMMENDATION.value
    agent_name = "RecommendationAgent"

    def run(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        scored = state.get("scored_customers", [])

        if not scored:
            self.append_log(
                state,
                step_name=WorkflowStep.RECOMMENDATION.value,
                status=LogStatus.SKIPPED.value,
                reasoning="No scored customers available to recommend products for.",
            )
            state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.RECOMMENDATION.value]
            return state

        # We recommend for every scored customer (above-threshold flag preserved
        # so that downstream agents can decide). This keeps the system flexible
        # for what-if analysis.
        recs: List[RecommendedCustomer] = []
        with timed_block("recommendation") as t:
            for s in scored:
                cand = s["customer"]
                rec_resp = self.invoke_tool(
                    tool_name=ToolName.RECOMMEND_PRODUCT.value,
                    params={
                        "customer": {
                            "id": cand["id"],
                            "customer_code": cand["customer_code"],
                            "full_name": cand["full_name"],
                            "annual_income": cand["annual_income"],
                            "monthly_salary": cand["monthly_salary"],
                            "account_balance": cand["account_balance"],
                            "credit_score": cand["credit_score"],
                            "customer_segment": cand["customer_segment"],
                        },
                        "conversion_probability": float(s.get("conversion_probability", 0.0)),
                        "preferred_audience": cand["customer_segment"],
                    },
                    run_id=run_id,
                )
                if not rec_resp.get("ok"):
                    self.record_error(state, f"recommendation failed for cust {cand['id']}: {rec_resp.get('error')}")
                    continue

                recs.append(
                    RecommendedCustomer(
                        customer=cand,
                        score=float(s.get("score", 0.0)),
                        conversion_probability=float(s.get("conversion_probability", 0.0)),
                        rationale_score=s.get("rationale", ""),
                        recommendation=rec_resp["result"],
                    )
                )

        state["recommendations"] = recs

        n_with_product = sum(1 for r in recs if r.get("recommendation", {}).get("product_code"))
        self.append_log(
            state,
            step_name=WorkflowStep.RECOMMENDATION.value,
            status=LogStatus.SUCCESS.value,
            reasoning=(
                f"Generated {n_with_product} product recommendations "
                f"out of {len(scored)} scored customers."
            ),
            duration_ms=t["duration_ms"],
        )
        state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.RECOMMENDATION.value]
        return state
