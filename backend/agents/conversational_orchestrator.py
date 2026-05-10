"""
Conversational Orchestrator.

A single agent that:
- talks to the RM in natural language,
- has every specialist agent and MCP tool exposed AS function-calling tools,
- decides which tools to call (and in what order) for each user turn,
- maintains short-term working memory across turns (ConversationSession),
- stops when it has enough to respond.

This replaces the rigid "always run the full pipeline" UX with a real
agentic, tool-using assistant. The fixed LangGraph workflow is still
available — it's just one of the tools (run_full_workflow) the orchestrator
can choose.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from backend.agents.base import BaseAgent
from backend.agents.orchestrator_tools import (
    TOOL_HANDLERS,
    TOOL_SCHEMAS,
    ConversationSession,
    sessions,
)
from backend.services.llm_service import llm_service
from backend.services.logging_service import get_logger
from backend.tools.audit_tools import log_agent_event
from backend.utils.constants import AgentRole, LogStatus
from backend.utils.helpers import safe_json_dumps

logger = get_logger(__name__)


_SYSTEM_PROMPT = """You are the Banking CRM Orchestrator — a senior agent helping a Relationship Manager (RM).

You have access to specialist agents and tools (see the function schemas):
- list_customers, get_customer
- discover_loan_candidates (Customer Discovery agent)
- score_customers (Scoring agent)
- recommend_product (Recommendation agent)
- generate_outreach (Outreach agent — writes WhatsApp copy)
- send_whatsapp, send_all_pending (Campaign agent)
- run_full_workflow (the canonical 5-step pipeline; use only when the RM clearly asks)
- explain_customer, get_session_summary

OPERATING RULES:
1. Decompose the RM's request into the smallest number of well-chosen tool calls.
2. Prefer specialist tools (e.g. discover → score → recommend → generate → send) over run_full_workflow.
3. Reuse the conversation's working set: if customers were just discovered/scored, refer to them by rank ("top", "rank 2") or by code ("C00042"). Do not re-discover unless asked.
4. NEVER fabricate data. If a tool returns an error, surface it briefly and ask a clarifying question.
5. Keep replies concise (<= 6 sentences). Use markdown bullets for lists. Do not paste large JSON.
6. After completing the request, give a short natural-language summary of what was done (e.g. "Discovered 5, scored, drafted message for the top one — ready to send?").
7. Be proactive but not autonomous: send WhatsApp messages only when the RM asks for it explicitly.

Today: a typical RM turn is "find HIGH segment customers" → "score them" → "outreach to the top one" → "send it"."""


class ConversationalOrchestrator(BaseAgent):
    """Single conversational agent that dynamically calls specialist agents/tools."""

    role = AgentRole.SUPERVISOR.value
    agent_name = "ConversationalOrchestrator"

    MAX_TOOL_HOPS = 6  # safety cap on the tool-call loop per turn

    def chat(
        self,
        *,
        conv_id: Optional[str],
        user_message: str,
        history: List[Dict[str, str]],
        rm_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process one user turn.

        Returns:
            {
                "conv_id": str,
                "reply": str,                      # final natural-language answer
                "tool_trace": [ {tool, args, result_brief}, ... ],
                "session_summary": {...},
                "used_llm": bool,
                "error": str | None
            }
        """
        session = sessions.get_or_create(conv_id, rm_name=rm_name)
        run_id = f"chat-{session.conv_id}-{int(time.time() * 1000)}"
        log_agent_event(
            run_id=run_id,
            agent_name=self.agent_name,
            step_name="chat_turn_start",
            status=LogStatus.STARTED.value,
            reasoning=user_message[:300],
        )

        if not llm_service.is_available:
            return {
                "conv_id": session.conv_id,
                "reply": (
                    "The conversational orchestrator needs an LLM with tool calling enabled. "
                    "Azure OpenAI is currently unreachable, so I can't reason over your request. "
                    "You can still trigger the fixed workflow via the **Quick run** button or REST API."
                ),
                "tool_trace": [],
                "session_summary": _session_snapshot(session),
                "used_llm": False,
                "error": "LLM unavailable",
            }

        # Build the message stack: system + (compressed) history + new user turn
        messages: List[Dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for h in (history or [])[-8:]:
            role = h.get("role")
            content = h.get("content") or ""
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content[:1500]})
        messages.append({"role": "user", "content": user_message})

        tool_trace: List[Dict[str, Any]] = []
        final_text = ""
        last_error: Optional[str] = None

        for hop in range(self.MAX_TOOL_HOPS):
            resp = llm_service.chat_with_tools(messages=messages, tools=TOOL_SCHEMAS)
            if not resp.get("ok"):
                last_error = resp.get("error") or "LLM error"
                break

            tool_calls = resp.get("tool_calls") or []
            content = resp.get("content")

            # If the model is done (no tool calls), capture content and exit
            if not tool_calls:
                final_text = (content or "").strip()
                break

            # Replay assistant tool-call message into the transcript
            messages.append(
                {
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc.get("_arg_str") or json.dumps(tc.get("arguments") or {})},
                        }
                        for tc in tool_calls
                        if tc.get("name")
                    ],
                }
            )

            # Execute each tool call, append its result as a tool message
            for tc in tool_calls:
                name = tc.get("name")
                args = tc.get("arguments") or {}
                handler = TOOL_HANDLERS.get(name)
                if handler is None:
                    result: Dict[str, Any] = {"error": f"unknown tool '{name}'"}
                else:
                    log_agent_event(
                        run_id=run_id,
                        agent_name=self.agent_name,
                        tool_name=name,
                        step_name="tool_call",
                        status=LogStatus.STARTED.value,
                        input_payload=args,
                    )
                    result = handler(session, args)
                    log_agent_event(
                        run_id=run_id,
                        agent_name=self.agent_name,
                        tool_name=name,
                        step_name="tool_call",
                        status=LogStatus.SUCCESS.value if "error" not in result else LogStatus.FAILED.value,
                        output_payload=_brief_result(result),
                        duration_ms=float(result.get("_duration_ms") or 0.0),
                        error_message=result.get("error"),
                    )

                tool_trace.append(
                    {
                        "tool": name,
                        "args": args,
                        "result_brief": _brief_result(result),
                        "ok": "error" not in result,
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": safe_json_dumps(_compact_for_llm(result), max_chars=4000),
                    }
                )
        else:
            # Loop exited because we hit MAX_TOOL_HOPS
            last_error = f"reached max tool hops ({self.MAX_TOOL_HOPS}) without final answer"

        # If we have no final text yet, ask the model for a final summary based on tools used
        if not final_text and not last_error:
            final_text = "Done. Anything else?"
        elif not final_text and last_error:
            final_text = f"Stopped early: {last_error}. Try rephrasing or click **Reset conversation**."

        log_agent_event(
            run_id=run_id,
            agent_name=self.agent_name,
            step_name="chat_turn_end",
            status=LogStatus.SUCCESS.value if not last_error else LogStatus.FAILED.value,
            reasoning=f"hops={len(tool_trace)} | trace={[t['tool'] for t in tool_trace]}",
            error_message=last_error,
        )

        return {
            "conv_id": session.conv_id,
            "reply": final_text,
            "tool_trace": tool_trace,
            "session_summary": _session_snapshot(session),
            "used_llm": True,
            "error": last_error,
        }


# =====================================================
# Helpers
# =====================================================
def _session_snapshot(session: ConversationSession) -> Dict[str, Any]:
    return {
        "conv_id": session.conv_id,
        "rm_name": session.rm_name,
        "discovered": len(session.discovered),
        "scored": len(session.scored),
        "recommendations": len(session.recs_by_id),
        "messages": len(session.msgs_by_id),
        "campaigns": len(session.campaigns),
        "loan_type": session.loan_type,
        "min_conversion_threshold": session.min_conversion_threshold,
        "top_n": session.top_n,
        "last_run_id": session.last_run_id,
    }


def _brief_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary used in tool_trace + audit logs (avoid huge payloads)."""
    if not isinstance(result, dict):
        return {"value": str(result)[:400]}
    keys = []
    for k in ("count", "discovered", "scored", "messages", "campaigns", "run_id",
              "status", "product_name", "recommended_amount", "interest_rate",
              "conversion_probability", "customer_code", "full_name", "error",
              "summary", "loan_type", "threshold"):
        if k in result:
            keys.append((k, result[k]))
    out = dict(keys)
    # Trim long lists
    for list_key in ("customers", "candidates", "scored", "details"):
        if list_key in result and isinstance(result[list_key], list):
            out[f"{list_key}_count"] = len(result[list_key])
            out[f"{list_key}_first"] = result[list_key][:1]
    return out


def _compact_for_llm(result: Dict[str, Any]) -> Dict[str, Any]:
    """Trim tool results before feeding back to the LLM (keep payloads small)."""
    if not isinstance(result, dict):
        return {"value": str(result)[:1000]}
    out: Dict[str, Any] = {}
    for k, v in result.items():
        if k.startswith("_"):
            continue
        if isinstance(v, list) and len(v) > 12:
            out[k] = v[:12]
            out[f"{k}_truncated_total"] = len(v)
        else:
            out[k] = v
    return out


# Module-level singleton consumed by the API
conversational_orchestrator = ConversationalOrchestrator()
