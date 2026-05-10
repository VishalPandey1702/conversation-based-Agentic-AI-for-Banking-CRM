"""
Outreach Agent.

Generates a personalized WhatsApp message for each recommended customer.

This is the ONLY agent in the system that uses the LLM directly. The LLM
is asked to write a short, friendly, structured message strictly bounded
by the data we already computed - no autonomous reasoning, no tool calls.

If the LLM is not available, we deterministically fall back to a templated
message so that the demo always works end-to-end.
"""
from __future__ import annotations

from typing import Any, Dict, List

from backend.agents.base import BaseAgent
from backend.services.llm_service import llm_service
from backend.utils.constants import AgentRole, LogStatus, WorkflowStep
from backend.utils.helpers import fmt_currency, timed_block
from backend.workflows.state import OutreachMessage, WorkflowState


SYSTEM_PROMPT = (
    "You are a senior banking copywriter for an Indian retail bank.\n"
    "You write short, friendly, professional WhatsApp messages (under 70 words) "
    "that are personalized using the structured customer + product data provided.\n"
    "Guidelines:\n"
    "- Open by addressing the customer by first name.\n"
    "- Reference at most one personalization signal (segment / inquiry / income).\n"
    "- Mention the product name, recommended amount, and interest rate.\n"
    "- End with a clear single call-to-action and a polite sign-off as 'Team RM Bank'.\n"
    "- DO NOT promise approval. DO NOT include placeholders.\n"
    "- Output JSON: {\"message\": \"<final whatsapp message>\"}"
)


def _fallback_message(customer: Dict[str, Any], rec: Dict[str, Any]) -> str:
    """Deterministic message used when the LLM is unavailable."""
    first_name = (customer.get("full_name") or "").split(" ")[0] or "Customer"
    product_name = rec.get("product_name") or "personal loan offer"
    amount = fmt_currency(rec.get("recommended_amount") or 0.0)
    rate = rec.get("interest_rate") or 0.0
    tenure = rec.get("tenure_months") or 0
    return (
        f"Hi {first_name}, based on your relationship with us we have a pre-qualified "
        f"{product_name} of up to {amount} at {rate:.2f}% p.a. for {tenure} months. "
        f"Reply YES to have your RM walk you through the next steps. – Team RM Bank"
    )


class OutreachAgent(BaseAgent):
    """Generates the actual WhatsApp body for each recommended customer."""

    role = AgentRole.OUTREACH.value
    agent_name = "OutreachAgent"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm_service

    def run(self, state: WorkflowState) -> WorkflowState:
        recs = state.get("recommendations", [])
        threshold = float(state.get("min_conversion_threshold", 0.55))

        if not recs:
            self.append_log(
                state,
                step_name=WorkflowStep.OUTREACH.value,
                status=LogStatus.SKIPPED.value,
                reasoning="No recommendations to generate outreach for.",
            )
            state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.OUTREACH.value]
            return state

        messages: List[OutreachMessage] = []
        used_llm = 0
        used_fallback = 0

        with timed_block("outreach") as t:
            for r in recs:
                cand = r["customer"]
                rec = r.get("recommendation") or {}
                if not rec.get("product_code"):
                    continue
                # Only message customers above the conversion threshold.
                if float(r.get("conversion_probability", 0.0)) < threshold:
                    continue

                user_prompt = self._build_user_prompt(cand, rec, r)
                fallback = {"message": _fallback_message(cand, rec)}
                llm_out = self.llm.complete_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    fallback=fallback,
                )
                msg = (llm_out.get("message") or "").strip() or fallback["message"]
                if llm_out.get("_llm_used"):
                    used_llm += 1
                else:
                    used_fallback += 1

                rationale = (
                    f"Personalized message for segment '{cand.get('customer_segment')}' "
                    f"and product '{rec.get('product_name')}' "
                    f"(prob={float(r.get('conversion_probability', 0.0)):.2f})."
                )
                messages.append(
                    OutreachMessage(
                        customer_id=cand["id"],
                        customer_code=cand["customer_code"],
                        full_name=cand["full_name"],
                        phone=cand["phone"],
                        message=msg,
                        product_code=rec.get("product_code"),
                        product_name=rec.get("product_name"),
                        rationale=rationale,
                    )
                )

        state["generated_messages"] = messages

        self.append_log(
            state,
            step_name=WorkflowStep.OUTREACH.value,
            status=LogStatus.SUCCESS.value,
            reasoning=(
                f"Generated {len(messages)} outreach messages "
                f"(LLM={used_llm}, fallback={used_fallback}, threshold={threshold:.2f})."
            ),
            duration_ms=t["duration_ms"],
        )
        state["completed_steps"] = state.get("completed_steps", []) + [WorkflowStep.OUTREACH.value]
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_user_prompt(cand: Dict[str, Any], rec: Dict[str, Any], r: Dict[str, Any]) -> str:
        first_name = (cand.get("full_name") or "").split(" ")[0] or "Customer"
        prob = float(r.get("conversion_probability", 0.0))
        return (
            "Generate a WhatsApp message in JSON for this customer.\n\n"
            "CUSTOMER:\n"
            f"- first_name: {first_name}\n"
            f"- segment: {cand.get('customer_segment')}\n"
            f"- annual_income: {fmt_currency(cand.get('annual_income') or 0.0)}\n"
            f"- credit_score: {cand.get('credit_score')}\n\n"
            "RECOMMENDED PRODUCT:\n"
            f"- product_name: {rec.get('product_name')}\n"
            f"- recommended_amount: {fmt_currency(rec.get('recommended_amount') or 0.0)}\n"
            f"- interest_rate: {rec.get('interest_rate')} % p.a.\n"
            f"- tenure_months: {rec.get('tenure_months')}\n\n"
            f"CONTEXT:\n"
            f"- conversion_probability: {prob:.2f}\n"
            f"- has_recent_inquiry: {cand.get('has_recent_inquiry', False)}\n"
        )
