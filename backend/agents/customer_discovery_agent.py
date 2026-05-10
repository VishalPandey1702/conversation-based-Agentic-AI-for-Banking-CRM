"""
Customer Discovery Agent.

Single responsibility: identify a candidate set of customers for the
specified loan campaign by combining:
- high-income / high-credit-score customers
- customers with recent loan inquiries (last 60 days)

Outputs into state["discovered_customers"].
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.base import BaseAgent
from backend.utils.constants import AgentRole, LogStatus, ToolName, WorkflowStep
from backend.utils.helpers import timed_block
from backend.workflows.state import DiscoveredCustomer, WorkflowState


class CustomerDiscoveryAgent(BaseAgent):
    """Finds customers most likely to be receptive to a personal loan campaign."""

    role = AgentRole.DISCOVERY.value
    agent_name = "CustomerDiscoveryAgent"

    def run(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        loan_type = state.get("loan_type", "PERSONAL")
        top_n = int(state.get("top_n_customers", 10))

        with timed_block("discovery") as t:
            # 1. Pull a base universe of qualifying customers
            high_income_resp = self.invoke_tool(
                tool_name=ToolName.FETCH_HIGH_INCOME_CUSTOMERS.value,
                params={
                    "min_annual_income": 600_000.0,
                    "min_credit_score": 680,
                    "only_segments": ["HIGH", "MEDIUM"],
                    "limit": max(top_n * 4, 40),
                },
                run_id=run_id,
            )
            if not high_income_resp.get("ok"):
                self.record_error(state, f"fetch_high_income_customers failed: {high_income_resp.get('error')}")
                self.append_log(
                    state,
                    step_name=WorkflowStep.DISCOVERY.value,
                    status=LogStatus.FAILED.value,
                    reasoning="Failed to fetch high-income customers.",
                    error=high_income_resp.get("error"),
                )
                state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.DISCOVERY.value]
                return state

            base_customers: List[Dict[str, Any]] = high_income_resp["result"]
            base_ids = [c["id"] for c in base_customers]

            # 2. Pull recent loan inquiries for that universe
            inquiry_resp = self.invoke_tool(
                tool_name=ToolName.FETCH_RECENT_LOAN_INQUIRIES.value,
                params={"days": 60, "loan_type": loan_type, "customer_ids": base_ids},
                run_id=run_id,
            )
            inquiries: List[Dict[str, Any]] = (
                inquiry_resp["result"] if inquiry_resp.get("ok") else []
            )

            inquiry_index: Dict[int, Dict[str, Any]] = {}
            for inq in inquiries:
                # keep the most recent per customer
                cid = inq["customer_id"]
                if cid not in inquiry_index:
                    inquiry_index[cid] = inq

            # 3. Build the discovered list, prioritising customers with recent inquiries
            discovered: List[DiscoveredCustomer] = []
            for c in base_customers:
                inq = inquiry_index.get(c["id"])
                if inq is not None:
                    reason = (
                        f"Recent {inq['loan_type']} inquiry for ₹{inq['inquiry_amount']:,.0f} "
                        f"via {inq['channel']} channel"
                    )
                else:
                    reason = (
                        f"High-value profile (income ₹{c['annual_income']:,.0f}, "
                        f"credit {c['credit_score']}, segment {c['customer_segment']})"
                    )

                discovered.append(
                    DiscoveredCustomer(
                        id=c["id"],
                        customer_code=c["customer_code"],
                        full_name=c["full_name"],
                        email=c["email"],
                        phone=c["phone"],
                        annual_income=c["annual_income"],
                        monthly_salary=c["monthly_salary"],
                        account_balance=c["account_balance"],
                        credit_score=c["credit_score"],
                        customer_segment=c["customer_segment"],
                        has_existing_loan=c["has_existing_loan"],
                        has_recent_inquiry=inq is not None,
                        most_recent_inquiry=inq,
                        discovery_reason=reason,
                    )
                )

            # 4. Sort: customers with recent inquiry first, then by income desc
            discovered.sort(
                key=lambda d: (
                    0 if d.get("has_recent_inquiry") else 1,
                    -float(d.get("annual_income", 0.0)),
                )
            )
            discovered = discovered[:top_n]

            state["discovered_customers"] = discovered

        # 5. Audit + reasoning trace
        n_inq = sum(1 for d in discovered if d.get("has_recent_inquiry"))
        reasoning = (
            f"Selected {len(discovered)} candidates (top_n={top_n}); "
            f"{n_inq} have a recent {loan_type} inquiry, "
            f"{len(discovered) - n_inq} are high-value profile-only matches."
        )
        self.append_log(
            state,
            step_name=WorkflowStep.DISCOVERY.value,
            status=LogStatus.SUCCESS.value,
            reasoning=reasoning,
            duration_ms=t["duration_ms"],
        )

        state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.DISCOVERY.value]
        return state
