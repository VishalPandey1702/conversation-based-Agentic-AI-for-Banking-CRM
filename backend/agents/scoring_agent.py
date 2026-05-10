"""
Scoring Agent.

For each discovered customer:
- pull transaction frequency / consistency
- combine with profile attributes
- run the deterministic scoring tool to produce conversion probability + rationale
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from backend.agents.base import BaseAgent
from backend.utils.constants import AgentRole, LogStatus, ToolName, WorkflowStep
from backend.utils.helpers import timed_block
from backend.workflows.state import ScoredCustomer, WorkflowState


class ScoringAgent(BaseAgent):
    """Computes conversion probability + explainability for each candidate."""

    role = AgentRole.SCORING.value
    agent_name = "ScoringAgent"

    def run(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        threshold = float(state.get("min_conversion_threshold", 0.55))
        candidates = state.get("discovered_customers", [])

        if not candidates:
            self.append_log(
                state,
                step_name=WorkflowStep.SCORING.value,
                status=LogStatus.SKIPPED.value,
                reasoning="No discovered customers to score.",
            )
            state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.SCORING.value]
            return state

        scored: List[ScoredCustomer] = []
        with timed_block("scoring") as t:
            for cand in candidates:
                cid = cand["id"]
                # 1. Transaction frequency / consistency feature
                freq_resp = self.invoke_tool(
                    tool_name=ToolName.ANALYZE_TRANSACTION_FREQUENCY.value,
                    params={"customer_id": cid, "days": 90},
                    run_id=run_id,
                )
                txn_consistency = (
                    float(freq_resp["result"].get("consistency_score", 0.0))
                    if freq_resp.get("ok")
                    else 0.0
                )

                # 2. Inquiry recency feature
                has_inquiry = bool(cand.get("has_recent_inquiry"))
                inquiry_recency_days = None
                inq = cand.get("most_recent_inquiry")
                if has_inquiry and inq:
                    try:
                        inq_dt = (
                            datetime.fromisoformat(str(inq["inquiry_date"]).replace("Z", ""))
                            if isinstance(inq["inquiry_date"], str)
                            else inq["inquiry_date"]
                        )
                        inquiry_recency_days = max(0, (datetime.utcnow() - inq_dt).days)
                    except Exception:  # noqa: BLE001
                        inquiry_recency_days = None

                # 3. Compute deterministic conversion score
                customer_payload: Dict[str, Any] = {
                    "id": cand["id"],
                    "customer_code": cand["customer_code"],
                    "full_name": cand["full_name"],
                    "annual_income": cand["annual_income"],
                    "monthly_salary": cand["monthly_salary"],
                    "account_balance": cand["account_balance"],
                    "credit_score": cand["credit_score"],
                    "customer_segment": cand["customer_segment"],
                    "has_existing_loan": cand.get("has_existing_loan", False),
                    "previous_repayment_score": 0.9,  # default; refined below if profile pulled
                }

                # Pull a tiny slice of profile data for repayment score
                profile_resp = self.invoke_tool(
                    tool_name=ToolName.FETCH_CUSTOMER_PROFILE.value,
                    params={"customer_id": cid},
                    run_id=run_id,
                )
                if profile_resp.get("ok") and profile_resp["result"]:
                    customer_payload["previous_repayment_score"] = (
                        profile_resp["result"]["customer"].get("previous_repayment_score", 0.9)
                    )

                score_resp = self.invoke_tool(
                    tool_name=ToolName.COMPUTE_CONVERSION_SCORE.value,
                    params={
                        "customer": customer_payload,
                        "txn_consistency": txn_consistency,
                        "has_recent_inquiry": has_inquiry,
                        "inquiry_recency_days": inquiry_recency_days,
                    },
                    run_id=run_id,
                )
                if not score_resp.get("ok"):
                    self.record_error(state, f"scoring failed for cust {cid}: {score_resp.get('error')}")
                    continue

                result = score_resp["result"]
                prob = float(result.get("conversion_probability", 0.0))

                scored.append(
                    ScoredCustomer(
                        customer=cand,
                        score=float(result.get("score", 0.0)),
                        conversion_probability=prob,
                        features=result.get("features", {}),
                        rationale=result.get("rationale", ""),
                        txn_consistency=txn_consistency,
                        above_threshold=prob >= threshold,
                    )
                )

        # 4. Sort by probability desc and persist
        scored.sort(key=lambda s: float(s.get("conversion_probability", 0.0)), reverse=True)
        state["scored_customers"] = scored

        passing = sum(1 for s in scored if s.get("above_threshold"))
        self.append_log(
            state,
            step_name=WorkflowStep.SCORING.value,
            status=LogStatus.SUCCESS.value,
            reasoning=(
                f"Scored {len(scored)} customers; "
                f"{passing} are above the conversion threshold of {threshold:.2f}."
            ),
            duration_ms=t["duration_ms"],
        )
        state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.SCORING.value]
        return state
