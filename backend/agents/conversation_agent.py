"""
Conversation Agent.

This is the entry-point for the chat UI. It understands the RM's natural-language
message, classifies intent, extracts parameters, and decides what to do next:

    RUN_WORKFLOW    -> trigger the multi-agent pipeline (with extracted overrides)
    QUERY_LAST_RUN  -> answer questions about the most recent run
    SHOW_CUSTOMER   -> look up a specific customer by code or id
    LIST_CUSTOMERS  -> filter the customer master (segment / min credit / min income)
    HELP            -> describe the system capabilities
    SMALLTALK       -> brief, polite reply for greetings / thanks / etc.

Routing strategy:
    1. LLM-based classification (Azure OpenAI, JSON-only response).
    2. Deterministic keyword/regex fallback when the LLM is unreachable or
       the auth circuit is open. The fallback covers the common RM phrases
       so the demo always works end-to-end.

This agent never invokes data tools directly; it RETURNS a structured
"action" object that the FastAPI /chat handler dispatches.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.agents.base import BaseAgent
from backend.services.llm_service import llm_service
from backend.services.logging_service import get_logger
from backend.utils.constants import AgentRole, LogStatus
from backend.utils.helpers import generate_run_id, timed_block

logger = get_logger(__name__)


# =====================================================
# Output schema
# =====================================================
class ConversationAction(BaseModel):
    """Structured decision returned to the API/UI layer."""

    intent: str = Field(..., description="One of RUN_WORKFLOW / QUERY_LAST_RUN / SHOW_CUSTOMER / LIST_CUSTOMERS / HELP / SMALLTALK")
    reply: str = Field(..., description="Short natural-language reply to render in the chat.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters for the chosen intent.")
    used_llm: bool = False
    rationale: str = ""


# =====================================================
# Intent classifier
# =====================================================
_INTENTS = {
    "RUN_WORKFLOW",
    "QUERY_LAST_RUN",
    "SHOW_CUSTOMER",
    "LIST_CUSTOMERS",
    "HELP",
    "SMALLTALK",
}

_HELP_TEXT = (
    "I can help you discover, score, and reach customers. Try things like:\n\n"
    "• *Find HIGH segment customers above 0.7 probability and send WhatsApp*\n"
    "• *Run a personal loan campaign for the top 5 customers*\n"
    "• *Show me customer C00042*\n"
    "• *List MEDIUM segment customers with credit > 720*\n"
    "• *Why was the top customer selected?* (after a run)\n"
    "• *How many messages did we send last time?*"
)


_SYSTEM_PROMPT = (
    "You are an intent router for a Banking CRM agentic system. Classify the user's "
    "message into ONE intent and extract structured parameters.\n\n"
    "Allowed intents (use exactly one):\n"
    "  - RUN_WORKFLOW: the user wants to discover customers, score them, recommend products, "
    "    generate outreach, or run a campaign. Examples: 'find high-value customers', "
    "    'run a personal loan campaign', 'send WhatsApp messages', 'launch outreach'.\n"
    "  - QUERY_LAST_RUN: the user asks about the previous run's results "
    "    ('explain the top customer', 'how many messages?', 'show campaign status').\n"
    "  - SHOW_CUSTOMER: the user mentions a specific customer code (Cxxxxx) or id.\n"
    "  - LIST_CUSTOMERS: the user wants to filter the customer master (segment, credit, income) "
    "    WITHOUT running a campaign.\n"
    "  - HELP: the user asks what you can do.\n"
    "  - SMALLTALK: greetings, thanks, generic chitchat.\n\n"
    "For RUN_WORKFLOW also extract any of these (omit if not stated):\n"
    "  top_n_customers (int 1-200), min_conversion_threshold (float 0..1), "
    "  loan_type (PERSONAL/HOME/CAR/EDUCATION), segment (HIGH/MEDIUM/LOW).\n"
    "For SHOW_CUSTOMER extract customer_code (string like 'C00042') or customer_id (int).\n"
    "For LIST_CUSTOMERS extract segment, min_credit_score, min_annual_income, limit.\n\n"
    "Return ONLY JSON: {\"intent\": ..., \"params\": {...}, \"reply\": \"<one-sentence "
    "confirmation of what you will do>\"}."
)


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize(intent: str) -> str:
    intent = (intent or "").upper().strip()
    return intent if intent in _INTENTS else "HELP"


def _normalize_segment(seg: Any) -> Optional[str]:
    if not seg:
        return None
    s = str(seg).upper().strip()
    return s if s in {"HIGH", "MEDIUM", "LOW"} else None


def _normalize_loan_type(lt: Any) -> Optional[str]:
    if not lt:
        return None
    s = str(lt).upper().strip()
    return s if s in {"PERSONAL", "HOME", "CAR", "EDUCATION"} else None


# =====================================================
# Deterministic fallback
# =====================================================
_RUN_KEYWORDS = (
    "run", "trigger", "launch", "start", "execute", "kick", "campaign",
    "find", "discover", "outreach", "send", "whatsapp", "loan", "convert",
    "personalize", "personalized", "personalised", "message", "messages",
    "generate", "shortlist", "target",
)
_QUERY_KEYWORDS = (
    "explain", "why", "summary", "summarize", "summarise", "how many",
    "what did", "show messages", "show campaigns", "show recommendations",
    "show scoring", "last run", "previous run",
)
_LIST_KEYWORDS = ("list customers", "filter customers", "show customers")
_HELP_KEYWORDS = ("help", "what can you", "capabilities", "how do i", "how to use")
_SMALL_KEYWORDS = ("hi", "hello", "hey", "thanks", "thank you", "thx", "good morning", "good evening")


def _fallback_classify(message: str, has_last_run: bool) -> Dict[str, Any]:
    """Keyword-based classifier used when the LLM is unavailable."""
    text = (message or "").lower().strip()
    params: Dict[str, Any] = {}

    # SHOW_CUSTOMER: look for Cxxxxx codes or "customer 42"
    code_match = re.search(r"\b([Cc]\d{2,7})\b", message)
    if code_match:
        params["customer_code"] = code_match.group(1).upper()
        return {
            "intent": "SHOW_CUSTOMER",
            "params": params,
            "reply": f"Looking up customer {params['customer_code']}.",
        }
    cust_id_match = re.search(r"\bcustomer\s+(?:id\s*)?(\d{1,7})\b", text)
    if cust_id_match:
        params["customer_id"] = int(cust_id_match.group(1))
        return {
            "intent": "SHOW_CUSTOMER",
            "params": params,
            "reply": f"Looking up customer id {params['customer_id']}.",
        }

    if any(k in text for k in _SMALL_KEYWORDS) and len(text) <= 30:
        return {"intent": "SMALLTALK", "params": {}, "reply": "Hi! Tell me what you'd like to do."}

    if any(k in text for k in _HELP_KEYWORDS):
        return {"intent": "HELP", "params": {}, "reply": "Here are some things I can do."}

    if any(k in text for k in _LIST_KEYWORDS):
        seg = _normalize_segment(re.search(r"\b(high|medium|low)\b", text).group(1)) if re.search(r"\b(high|medium|low)\b", text) else None
        if seg:
            params["segment"] = seg
        m = re.search(r"credit\s*(?:score|>=?|>|over|above)?\s*(\d{3,4})", text)
        if m:
            params["min_credit_score"] = int(m.group(1))
        return {"intent": "LIST_CUSTOMERS", "params": params, "reply": "Filtering customers."}

    # RUN_WORKFLOW heuristics: extract overrides
    is_runnish = any(k in text for k in _RUN_KEYWORDS)
    if is_runnish:
        m = re.search(r"top\s+(\d{1,3})", text)
        if m:
            params["top_n_customers"] = int(m.group(1))
        m = re.search(r"(?:above|over|greater than|>=|>)\s*(0?\.\d+|\d+\.\d+|\d{1,2})\s*(%)?", text)
        if m:
            v = float(m.group(1))
            if m.group(2) == "%" or v > 1.0:
                v = v / 100.0
            if 0.0 <= v <= 1.0:
                params["min_conversion_threshold"] = round(v, 2)
        m = re.search(r"\b(personal|home|car|education)\b", text)
        if m:
            params["loan_type"] = m.group(1).upper()
        m = re.search(r"\b(high|medium|low)\s+(?:segment|customers|tier)?\b", text)
        if m:
            params["segment"] = m.group(1).upper()
        return {
            "intent": "RUN_WORKFLOW",
            "params": params,
            "reply": "Got it - running the multi-agent workflow now.",
        }

    if has_last_run and any(k in text for k in _QUERY_KEYWORDS):
        return {
            "intent": "QUERY_LAST_RUN",
            "params": {},
            "reply": "Looking at the last run for an answer.",
        }

    # Default: ask for help
    return {
        "intent": "HELP",
        "params": {},
        "reply": "I'm not sure what you'd like to do.",
    }


# =====================================================
# Conversation agent
# =====================================================
class ConversationAgent(BaseAgent):
    """Classifies the RM's chat message into an action."""

    role = AgentRole.SUPERVISOR.value  # supervisor is the only role allowed to "decide"
    agent_name = "ConversationAgent"

    def classify(
        self,
        *,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        has_last_run: bool = False,
    ) -> ConversationAction:
        """
        Decide what to do with the user's message.

        Args:
            message: latest user message.
            history: list of {"role": "user"|"assistant", "content": str}.
            has_last_run: whether a recent workflow run exists in memory.
        """
        history = history or []
        run_id = generate_run_id("chat")

        with timed_block("conversation_classify") as t:
            llm_decision: Optional[Dict[str, Any]] = None
            if llm_service.is_available:
                user_prompt = _build_user_prompt(message, history, has_last_run)
                llm_decision = llm_service.complete_json(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    fallback={"intent": "HELP", "params": {}, "reply": "Could you rephrase that?"},
                )

            used_llm = bool(llm_decision and llm_decision.get("_llm_used"))
            decision = llm_decision if used_llm else _fallback_classify(message, has_last_run)

            # Validate / normalize
            intent = _normalize(decision.get("intent"))
            params_in = decision.get("params") or {}
            reply = (decision.get("reply") or "").strip()

            params = self._normalize_params(intent, params_in)

            # If LLM said RUN_WORKFLOW but user clearly said hi/help, coerce.
            text_lower = (message or "").lower().strip()
            if intent == "RUN_WORKFLOW" and len(text_lower) <= 4 and text_lower in {"hi", "hey", "yo", "ok"}:
                intent = "SMALLTALK"
                reply = "Hi! Tell me what you'd like to do."

            if intent == "HELP":
                reply = reply or "Here is what I can do."
            if intent == "SMALLTALK":
                reply = reply or "Got it!"

            action = ConversationAction(
                intent=intent,
                reply=reply or self._default_reply(intent, params),
                params=params,
                used_llm=used_llm,
                rationale=f"intent={intent}; via={'llm' if used_llm else 'fallback'}; params={params}",
            )

        # Audit
        try:
            self.append_log(
                state={"run_id": run_id, "logs": []},  # type: ignore[arg-type]
                step_name="conversation_classify",
                status=LogStatus.SUCCESS.value,
                reasoning=action.rationale,
                duration_ms=t["duration_ms"],
                write_audit=True,
            )
        except Exception:  # noqa: BLE001
            pass

        return action

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_params(intent: str, params_in: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if intent == "RUN_WORKFLOW":
            if (v := _safe_int(params_in.get("top_n_customers"))) is not None:
                out["top_n_customers"] = max(1, min(200, v))
            if (v := _safe_float(params_in.get("min_conversion_threshold"))) is not None:
                out["min_conversion_threshold"] = max(0.0, min(1.0, v))
            if (lt := _normalize_loan_type(params_in.get("loan_type"))):
                out["loan_type"] = lt
            if (seg := _normalize_segment(params_in.get("segment"))):
                out["segment"] = seg
        elif intent == "SHOW_CUSTOMER":
            if (cid := _safe_int(params_in.get("customer_id"))) is not None:
                out["customer_id"] = cid
            code = (params_in.get("customer_code") or "").strip().upper()
            if re.match(r"^C\d{2,7}$", code):
                out["customer_code"] = code
        elif intent == "LIST_CUSTOMERS":
            if (seg := _normalize_segment(params_in.get("segment"))):
                out["segment"] = seg
            if (v := _safe_int(params_in.get("min_credit_score"))) is not None:
                out["min_credit_score"] = max(0, min(900, v))
            if (v := _safe_float(params_in.get("min_annual_income"))) is not None:
                out["min_annual_income"] = max(0.0, v)
            if (v := _safe_int(params_in.get("limit"))) is not None:
                out["limit"] = max(1, min(500, v))
        return out

    @staticmethod
    def _default_reply(intent: str, params: Dict[str, Any]) -> str:
        if intent == "RUN_WORKFLOW":
            bits = []
            if "loan_type" in params:
                bits.append(f"loan: {params['loan_type']}")
            if "segment" in params:
                bits.append(f"segment: {params['segment']}")
            if "top_n_customers" in params:
                bits.append(f"top N: {params['top_n_customers']}")
            if "min_conversion_threshold" in params:
                bits.append(f"threshold: {params['min_conversion_threshold']:.2f}")
            tail = f" ({', '.join(bits)})" if bits else ""
            return f"Running the multi-agent workflow now{tail}."
        if intent == "QUERY_LAST_RUN":
            return "Looking up the last run."
        if intent == "SHOW_CUSTOMER":
            return "Pulling that customer's profile."
        if intent == "LIST_CUSTOMERS":
            return "Filtering the customer master."
        if intent == "HELP":
            return _HELP_TEXT
        return "Ok."

    @staticmethod
    def help_text() -> str:
        return _HELP_TEXT


def _build_user_prompt(message: str, history: List[Dict[str, str]], has_last_run: bool) -> str:
    last_5 = history[-5:] if history else []
    h = "\n".join([f"- {h.get('role','?')}: {h.get('content','')[:300]}" for h in last_5])
    return (
        f"LATEST USER MESSAGE:\n{message}\n\n"
        f"CONTEXT:\n- has_last_run: {has_last_run}\n"
        f"RECENT HISTORY (most recent last):\n{h or '(none)'}"
    )


# Module-level singleton
conversation_agent = ConversationAgent()
