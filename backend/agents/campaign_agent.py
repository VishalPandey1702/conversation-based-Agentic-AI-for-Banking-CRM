"""
Campaign Execution Agent.

For each generated outreach message, simulate a WhatsApp send through the
WhatsApp tool. Persists the campaign records in the database and produces
a final summary in state["summary"].
"""
from __future__ import annotations

from typing import List

from backend.agents.base import BaseAgent
from backend.utils.constants import AgentRole, LogStatus, ToolName, WorkflowStep
from backend.utils.helpers import timed_block
from backend.workflows.state import CampaignResult, WorkflowState


class CampaignAgent(BaseAgent):
    """Simulates campaign execution by dispatching WhatsApp messages."""

    role = AgentRole.CAMPAIGN.value
    agent_name = "CampaignAgent"

    def run(self, state: WorkflowState) -> WorkflowState:
        run_id = state["run_id"]
        messages = state.get("generated_messages", [])

        if not messages:
            self.append_log(
                state,
                step_name=WorkflowStep.CAMPAIGN.value,
                status=LogStatus.SKIPPED.value,
                reasoning="No outreach messages to dispatch.",
            )
            state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.CAMPAIGN.value]
            state["summary"] = "Campaign skipped: no qualifying customers had a generated message."
            return state

        results: List[CampaignResult] = []
        sent_count = 0
        failed_count = 0

        with timed_block("campaign") as t:
            for m in messages:
                resp = self.invoke_tool(
                    tool_name=ToolName.SEND_WHATSAPP_MESSAGE.value,
                    params={
                        "customer_id": m["customer_id"],
                        "phone": m["phone"],
                        "message": m["message"],
                        "campaign_run_id": run_id,
                    },
                    run_id=run_id,
                )
                if resp.get("ok"):
                    payload = resp["result"]
                    status = payload.get("status", "PENDING")
                    if status == "SENT":
                        sent_count += 1
                    else:
                        failed_count += 1
                    results.append(
                        CampaignResult(
                            customer_id=m["customer_id"],
                            customer_code=m["customer_code"],
                            full_name=m["full_name"],
                            phone=m["phone"],
                            status=status,
                            sent_at=payload.get("timestamp", ""),
                            campaign_id=payload.get("campaign_id"),
                            message=m["message"],
                        )
                    )
                else:
                    failed_count += 1
                    results.append(
                        CampaignResult(
                            customer_id=m["customer_id"],
                            customer_code=m["customer_code"],
                            full_name=m["full_name"],
                            phone=m["phone"],
                            status="FAILED",
                            sent_at="",
                            campaign_id=None,
                            message=m["message"],
                        )
                    )

        state["campaign_results"] = results

        summary = (
            f"Campaign run '{run_id}': dispatched {sent_count} messages, "
            f"{failed_count} failed (total {len(results)})."
        )
        state["summary"] = summary

        self.append_log(
            state,
            step_name=WorkflowStep.CAMPAIGN.value,
            status=LogStatus.SUCCESS.value,
            reasoning=summary,
            duration_ms=t["duration_ms"],
        )
        state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.CAMPAIGN.value]
        return state
