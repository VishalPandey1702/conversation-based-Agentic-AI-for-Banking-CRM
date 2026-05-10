"""
Agents-as-tools layer for the conversational orchestrator.

Each function below is a thin "tool" that wraps a specialist agent or MCP tool.
The conversational orchestrator (LLM with tool calling) selects which one(s) to
invoke based on the RM's natural-language request.

A `ConversationSession` keeps short-term working memory between tool calls so
the orchestrator can refer to earlier results (e.g. "score the customers you
just discovered", "send WhatsApp to the top 3").

The tools share the same role-governed MCP server used by the fixed workflow,
so RBAC/audit/permission constraints continue to apply.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from backend.agents.campaign_agent import CampaignAgent
from backend.agents.customer_discovery_agent import CustomerDiscoveryAgent
from backend.agents.outreach_agent import OutreachAgent
from backend.agents.recommendation_agent import RecommendationAgent
from backend.agents.scoring_agent import ScoringAgent
from backend.database.db import session_scope
from backend.database.models import Customer
from backend.services.logging_service import get_logger
from backend.tools import customer_tools, recommendation_tools, scoring_tools, transaction_tools, whatsapp_tools
from backend.utils.helpers import generate_run_id
from backend.workflows.execution_manager import execution_manager
from backend.workflows.state import empty_state

logger = get_logger(__name__)


# =====================================================
# Per-conversation working memory
# =====================================================
@dataclass
class ConversationSession:
    """In-memory state carried across tool calls inside one conversation."""

    conv_id: str
    rm_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    # Short-term references the orchestrator can mention by name
    discovered: List[Dict[str, Any]] = field(default_factory=list)
    scored: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # generated outreach
    campaigns: List[Dict[str, Any]] = field(default_factory=list)
    last_run_id: Optional[str] = None
    # Settings carried forward
    loan_type: str = "PERSONAL"
    min_conversion_threshold: float = 0.55
    top_n: int = 10
    # Per-customer caches
    profiles_cache: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    scores_by_id: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    recs_by_id: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    msgs_by_id: Dict[int, Dict[str, Any]] = field(default_factory=dict)


class _SessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create(self, conv_id: Optional[str] = None, rm_name: Optional[str] = None) -> ConversationSession:
        if not conv_id:
            conv_id = f"conv-{uuid.uuid4().hex[:10]}"
        with self._lock:
            sess = self._sessions.get(conv_id)
            if sess is None:
                sess = ConversationSession(conv_id=conv_id, rm_name=rm_name)
                self._sessions[conv_id] = sess
            elif rm_name and not sess.rm_name:
                sess.rm_name = rm_name
            return sess

    def reset(self, conv_id: str) -> None:
        with self._lock:
            self._sessions.pop(conv_id, None)


sessions = _SessionRegistry()


# =====================================================
# Customer reference resolver
# =====================================================
def _resolve_customer_id(session: ConversationSession, ref: Any) -> Optional[int]:
    """
    Resolve a customer reference to a numeric id.

    Accepts:
        - int id ("42")
        - customer code ("C00042")
        - keywords: "top", "top 1", "first", "best", "highest"
        - 1-based "rank N" against the most recent scored / discovered list
    """
    if ref is None:
        return None

    # int id
    if isinstance(ref, int):
        return int(ref)

    s = str(ref).strip()
    if s.isdigit():
        return int(s)

    # customer code
    code_match = re.match(r"^[Cc]\d{2,7}$", s)
    if code_match:
        with session_scope() as sdb:
            row = sdb.query(Customer).filter(Customer.customer_code == s.upper()).first()
            return row.id if row else None

    # ranked references against the most recent list (scored > discovered)
    ranked = session.scored or session.discovered
    if ranked:
        low = s.lower()
        if low in {"top", "best", "first", "highest", "top1", "top 1", "rank 1"}:
            return _customer_id_from_entry(ranked[0])
        m = re.search(r"(?:rank|top)\s*(\d{1,2})", low)
        if m:
            idx = max(0, int(m.group(1)) - 1)
            if 0 <= idx < len(ranked):
                return _customer_id_from_entry(ranked[idx])

    # name match (case-insensitive contains)
    if ranked:
        for entry in ranked:
            cust = _entry_customer(entry)
            if cust and s.lower() in (cust.get("full_name") or "").lower():
                return cust["id"]
    return None


def _entry_customer(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A scored entry has {customer:{...}}; a discovered entry IS the customer."""
    if "customer" in entry and isinstance(entry["customer"], dict):
        return entry["customer"]
    return entry


def _customer_id_from_entry(entry: Dict[str, Any]) -> Optional[int]:
    c = _entry_customer(entry)
    return c.get("id") if c else None


# =====================================================
# Tool implementations
# =====================================================
def list_customers_tool(
    session: ConversationSession,
    *,
    segment: Optional[str] = None,
    min_credit_score: int = 0,
    min_annual_income: float = 0.0,
    limit: int = 25,
) -> Dict[str, Any]:
    """Filter the customer master."""
    with session_scope() as s:
        q = s.query(Customer)
        if segment:
            q = q.filter(Customer.customer_segment == str(segment).upper())
        if min_credit_score:
            q = q.filter(Customer.credit_score >= int(min_credit_score))
        if min_annual_income:
            q = q.filter(Customer.annual_income >= float(min_annual_income))
        rows = q.order_by(Customer.annual_income.desc()).limit(int(limit)).all()
        out = [
            {
                "id": c.id,
                "customer_code": c.customer_code,
                "full_name": c.full_name,
                "phone": c.phone,
                "city": c.city,
                "occupation": c.occupation,
                "annual_income": c.annual_income,
                "credit_score": c.credit_score,
                "customer_segment": c.customer_segment,
                "has_existing_loan": c.has_existing_loan,
            }
            for c in rows
        ]
    # Update working memory: list_customers may seed the next steps
    session.discovered = [
        {
            "id": x["id"],
            "customer_code": x["customer_code"],
            "full_name": x["full_name"],
            "phone": x["phone"],
            "annual_income": x["annual_income"],
            "credit_score": x["credit_score"],
            "customer_segment": x["customer_segment"],
            "has_existing_loan": x["has_existing_loan"],
            "monthly_salary": x["annual_income"] / 12.0,
            "account_balance": 0.0,
            "has_recent_inquiry": False,
            "discovery_reason": "Manual filter",
        }
        for x in out
    ]
    return {"count": len(out), "customers": out}


def get_customer_tool(session: ConversationSession, *, customer_ref: Any) -> Dict[str, Any]:
    """360° profile for a single customer (id, code, or name fragment)."""
    cid = _resolve_customer_id(session, customer_ref)
    if cid is None:
        return {"error": f"could not resolve customer reference '{customer_ref}'"}
    profile = customer_tools.fetch_customer_profile(cid)
    if profile is None:
        return {"error": f"customer id {cid} not found"}
    session.profiles_cache[cid] = profile
    return profile


def discover_loan_candidates_tool(
    session: ConversationSession,
    *,
    loan_type: str = "PERSONAL",
    top_n: int = 10,
    segments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the discovery agent to surface candidate customers."""
    seg_list = [str(s).upper() for s in (segments or ["HIGH", "MEDIUM"])]
    state = empty_state(
        run_id=generate_run_id("disc"),
        user_query=f"discover candidates for {loan_type}",
        rm_name=session.rm_name,
        top_n_customers=int(top_n),
        loan_type=loan_type,
    )
    # We pass the segment hint by tweaking the agent's lookup parameters via state.
    # Discovery agent uses ["HIGH","MEDIUM"] hard-coded; if user wanted only HIGH
    # we filter the result post-hoc.
    discovery = CustomerDiscoveryAgent()
    new_state = discovery.run(state)
    rows = new_state.get("discovered_customers", [])
    if seg_list != ["HIGH", "MEDIUM"]:
        rows = [r for r in rows if r.get("customer_segment") in seg_list]
    session.discovered = rows
    session.loan_type = loan_type.upper()
    session.top_n = int(top_n)
    return {
        "count": len(rows),
        "loan_type": loan_type,
        "candidates": [
            {
                "id": r["id"],
                "customer_code": r["customer_code"],
                "full_name": r["full_name"],
                "segment": r["customer_segment"],
                "credit_score": r["credit_score"],
                "annual_income": r["annual_income"],
                "has_recent_inquiry": r["has_recent_inquiry"],
                "discovery_reason": r["discovery_reason"],
            }
            for r in rows
        ],
    }


def score_customers_tool(
    session: ConversationSession,
    *,
    customer_refs: Optional[List[Any]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Score either a specific subset (by refs) or the current discovered/working set.

    Returns ranked scores (highest probability first).
    """
    if customer_refs:
        ids = [cid for ref in customer_refs if (cid := _resolve_customer_id(session, ref)) is not None]
    elif session.discovered:
        ids = [d["id"] for d in session.discovered]
    else:
        return {"error": "no customers in working set; call discover_loan_candidates or list_customers first"}

    if not ids:
        return {"error": "could not resolve any customer references"}

    thr = float(threshold) if threshold is not None else session.min_conversion_threshold
    state = empty_state(
        run_id=generate_run_id("score"),
        user_query="score subset",
        rm_name=session.rm_name,
        top_n_customers=len(ids),
        min_conversion_threshold=thr,
        loan_type=session.loan_type,
    )
    # Build the discovered_customers stub the scoring agent expects
    if session.discovered:
        seed = {d["id"]: d for d in session.discovered}
    else:
        seed = {}
    # Top-up with light profile lookups for ids not in working set
    cust_stub: List[Dict[str, Any]] = []
    for cid in ids:
        if cid in seed:
            cust_stub.append(seed[cid])
        else:
            with session_scope() as sdb:
                row = sdb.query(Customer).filter(Customer.id == cid).first()
                if not row:
                    continue
                cust_stub.append(
                    {
                        "id": row.id,
                        "customer_code": row.customer_code,
                        "full_name": row.full_name,
                        "phone": row.phone,
                        "annual_income": row.annual_income,
                        "monthly_salary": row.monthly_salary,
                        "account_balance": row.account_balance,
                        "credit_score": row.credit_score,
                        "customer_segment": row.customer_segment,
                        "has_existing_loan": row.has_existing_loan,
                        "has_recent_inquiry": False,
                        "discovery_reason": "Ad-hoc lookup",
                    }
                )
    state["discovered_customers"] = cust_stub
    scoring = ScoringAgent()
    new_state = scoring.run(state)
    scored = new_state.get("scored_customers", [])
    session.scored = scored
    session.min_conversion_threshold = thr
    for s in scored:
        cid = s["customer"]["id"]
        session.scores_by_id[cid] = s
    return {
        "count": len(scored),
        "threshold": thr,
        "scored": [
            {
                "id": s["customer"]["id"],
                "customer_code": s["customer"]["customer_code"],
                "full_name": s["customer"]["full_name"],
                "segment": s["customer"]["customer_segment"],
                "conversion_probability": round(float(s["conversion_probability"]), 4),
                "above_threshold": s.get("above_threshold"),
                "rationale": s.get("rationale"),
            }
            for s in scored
        ],
    }


def recommend_product_tool(session: ConversationSession, *, customer_ref: Any) -> Dict[str, Any]:
    """Recommend the best-fit product for one customer."""
    cid = _resolve_customer_id(session, customer_ref)
    if cid is None:
        return {"error": f"could not resolve '{customer_ref}'"}

    cust = _customer_dict_for(session, cid)
    if not cust:
        return {"error": f"customer id {cid} not found"}

    prob = float(session.scores_by_id.get(cid, {}).get("conversion_probability", 0.5))
    rec = recommendation_tools.recommend_product(
        customer=cust,
        conversion_probability=prob,
        preferred_audience=cust.get("customer_segment"),
    )
    session.recs_by_id[cid] = rec
    # also append to recs list (dedupe by id)
    session.recommendations = [r for r in session.recommendations if r.get("customer_id") != cid] + [rec]
    return rec


def generate_outreach_tool(
    session: ConversationSession,
    *,
    customer_ref: Any,
    tone: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a personalized WhatsApp message for a single customer."""
    cid = _resolve_customer_id(session, customer_ref)
    if cid is None:
        return {"error": f"could not resolve '{customer_ref}'"}
    cust = _customer_dict_for(session, cid)
    if not cust:
        return {"error": f"customer id {cid} not found"}

    # Ensure we have a recommendation
    rec = session.recs_by_id.get(cid)
    if not rec or not rec.get("product_code"):
        rec = recommend_product_tool(session, customer_ref=cid)
        if rec.get("error"):
            return rec

    # Reuse the OutreachAgent's prompt logic via a tiny one-off state
    state = empty_state(
        run_id=generate_run_id("out"),
        user_query="generate outreach",
        rm_name=session.rm_name,
        top_n_customers=1,
        min_conversion_threshold=0.0,  # always generate when explicitly asked
        loan_type=session.loan_type,
    )
    state["recommendations"] = [
        {
            "customer": cust,
            "score": float(session.scores_by_id.get(cid, {}).get("score", 0.0)),
            "conversion_probability": float(session.scores_by_id.get(cid, {}).get("conversion_probability", 0.5)),
            "rationale_score": session.scores_by_id.get(cid, {}).get("rationale", ""),
            "recommendation": rec,
        }
    ]
    if tone:
        # Light injection of tone via the user_query path (visible in logs)
        state["user_query"] = f"generate outreach with tone: {tone}"
    OutreachAgent().run(state)
    msgs = state.get("generated_messages", [])
    if not msgs:
        return {"error": "no message generated (agent returned empty)"}
    msg = msgs[0]
    session.msgs_by_id[cid] = msg
    session.messages = [m for m in session.messages if m.get("customer_id") != cid] + [msg]
    return msg


def send_whatsapp_tool(
    session: ConversationSession,
    *,
    customer_ref: Any,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Simulated WhatsApp send for one customer."""
    cid = _resolve_customer_id(session, customer_ref)
    if cid is None:
        return {"error": f"could not resolve '{customer_ref}'"}

    cust = _customer_dict_for(session, cid)
    if not cust:
        return {"error": f"customer id {cid} not found"}

    text = message or session.msgs_by_id.get(cid, {}).get("message")
    if not text:
        out = generate_outreach_tool(session, customer_ref=cid)
        if out.get("error"):
            return out
        text = out.get("message")

    result = whatsapp_tools.send_whatsapp_message(
        customer_id=cid,
        phone=cust.get("phone") or "",
        message=text,
        campaign_run_id=session.last_run_id or session.conv_id,
    )
    session.campaigns.append({**result, "customer_id": cid, "full_name": cust.get("full_name")})
    return result


def send_all_pending_tool(session: ConversationSession) -> Dict[str, Any]:
    """Dispatch every generated outreach message that hasn't been sent yet."""
    if not session.messages:
        return {"error": "no generated outreach messages in this conversation"}
    sent_ids = {c.get("customer_id") for c in session.campaigns if c.get("status") == "SENT"}
    sent = 0
    failed = 0
    out = []
    for m in session.messages:
        cid = m.get("customer_id")
        if cid in sent_ids:
            continue
        result = send_whatsapp_tool(session, customer_ref=cid, message=m.get("message"))
        out.append(result)
        if result.get("status") == "SENT":
            sent += 1
        else:
            failed += 1
    return {"dispatched": sent, "failed": failed, "details": out}


def run_full_workflow_tool(
    session: ConversationSession,
    *,
    loan_type: str = "PERSONAL",
    top_n: int = 10,
    threshold: float = 0.55,
) -> Dict[str, Any]:
    """Execute the entire fixed multi-agent pipeline (the original 'campaign' shortcut)."""
    state = execution_manager.run_full_workflow(
        user_query=f"end-to-end {loan_type} campaign top {top_n} >= {threshold:.2f}",
        rm_name=session.rm_name,
        top_n_customers=int(top_n),
        min_conversion_threshold=float(threshold),
        loan_type=loan_type,
    )
    session.discovered = list(state.get("discovered_customers") or [])
    session.scored = list(state.get("scored_customers") or [])
    session.recommendations = [r.get("recommendation") for r in (state.get("recommendations") or []) if r]
    session.messages = list(state.get("generated_messages") or [])
    session.campaigns = list(state.get("campaign_results") or [])
    session.last_run_id = state.get("run_id")
    session.loan_type = loan_type.upper()
    session.top_n = int(top_n)
    session.min_conversion_threshold = float(threshold)
    for s in session.scored:
        cid = s["customer"]["id"]
        session.scores_by_id[cid] = s
    for m in session.messages:
        session.msgs_by_id[m["customer_id"]] = m
    return {
        "run_id": state.get("run_id"),
        "completed_steps": state.get("completed_steps"),
        "discovered": len(session.discovered),
        "scored": len(session.scored),
        "messages": len(session.messages),
        "campaigns": len(session.campaigns),
        "summary": state.get("summary"),
    }


def explain_customer_tool(session: ConversationSession, *, customer_ref: Any) -> Dict[str, Any]:
    """Explain why a particular customer was selected/scored the way they were."""
    cid = _resolve_customer_id(session, customer_ref)
    if cid is None:
        return {"error": f"could not resolve '{customer_ref}'"}
    cust = _customer_dict_for(session, cid)
    score = session.scores_by_id.get(cid)
    rec = session.recs_by_id.get(cid)
    return {
        "customer_id": cid,
        "customer_code": cust.get("customer_code") if cust else None,
        "full_name": cust.get("full_name") if cust else None,
        "score": score,
        "recommendation": rec,
        "discovery_reason": next(
            (d.get("discovery_reason") for d in session.discovered if d.get("id") == cid),
            None,
        ),
    }


def get_session_summary_tool(session: ConversationSession) -> Dict[str, Any]:
    """Snapshot of what's currently in the conversation's working memory."""
    return {
        "conv_id": session.conv_id,
        "rm_name": session.rm_name,
        "loan_type": session.loan_type,
        "min_conversion_threshold": session.min_conversion_threshold,
        "top_n": session.top_n,
        "discovered": len(session.discovered),
        "scored": len(session.scored),
        "recommendations": len(session.recs_by_id),
        "messages": len(session.msgs_by_id),
        "campaigns": len(session.campaigns),
        "last_run_id": session.last_run_id,
    }


def _customer_dict_for(session: ConversationSession, cid: int) -> Optional[Dict[str, Any]]:
    """Return a CustomerSummary-like dict for the given id (cache-aware)."""
    # Try the discovered/scored caches first
    for d in session.discovered:
        if d.get("id") == cid:
            return d
    for s in session.scored:
        if s.get("customer", {}).get("id") == cid:
            return s["customer"]
    # Fallback to DB
    profile = session.profiles_cache.get(cid) or customer_tools.fetch_customer_profile(cid)
    if profile:
        session.profiles_cache[cid] = profile
        return profile["customer"]
    return None


# =====================================================
# Tool registry (name -> handler) + OpenAI tool schemas
# =====================================================
ToolHandler = Callable[..., Dict[str, Any]]


def _wrap(handler: ToolHandler) -> Callable[[ConversationSession, Dict[str, Any]], Dict[str, Any]]:
    def _call(session: ConversationSession, args: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            result = handler(session, **(args or {}))
        except TypeError as exc:
            return {"error": f"bad arguments: {exc}"}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool '%s' failed: %s", handler.__name__, exc)
            return {"error": str(exc)}
        result["_duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return result

    return _call


TOOL_HANDLERS: Dict[str, Callable[[ConversationSession, Dict[str, Any]], Dict[str, Any]]] = {
    "list_customers": _wrap(list_customers_tool),
    "get_customer": _wrap(get_customer_tool),
    "discover_loan_candidates": _wrap(discover_loan_candidates_tool),
    "score_customers": _wrap(score_customers_tool),
    "recommend_product": _wrap(recommend_product_tool),
    "generate_outreach": _wrap(generate_outreach_tool),
    "send_whatsapp": _wrap(send_whatsapp_tool),
    "send_all_pending": _wrap(send_all_pending_tool),
    "run_full_workflow": _wrap(run_full_workflow_tool),
    "explain_customer": _wrap(explain_customer_tool),
    "get_session_summary": _wrap(get_session_summary_tool),
}


# OpenAI-format tool schemas
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_customers",
            "description": (
                "Filter the customer master by segment / minimum credit score / minimum income. "
                "Use this when the RM wants to BROWSE customers (without running a campaign). "
                "Updates the conversation's working set so subsequent tools can refer to these."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "segment": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "min_credit_score": {"type": "integer"},
                    "min_annual_income": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Fetch a 360° profile for one customer (by id, code 'C00042', or 'top'/'rank N').",
            "parameters": {
                "type": "object",
                "properties": {"customer_ref": {"type": "string"}},
                "required": ["customer_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "discover_loan_candidates",
            "description": (
                "Run the Customer Discovery agent to surface high-value loan candidates "
                "(combines income, credit, and recent loan inquiries). Updates the working set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_type": {"type": "string", "enum": ["PERSONAL", "HOME", "CAR", "EDUCATION"]},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
                    "segments": {"type": "array", "items": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_customers",
            "description": (
                "Run the Scoring agent. If 'customer_refs' is omitted, scores the current "
                "working set. Returns conversion probability + per-feature rationale."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_refs": {"type": "array", "items": {"type": "string"}},
                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_product",
            "description": "Run the Recommendation agent for ONE customer (id, code, or 'top').",
            "parameters": {
                "type": "object",
                "properties": {"customer_ref": {"type": "string"}},
                "required": ["customer_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_outreach",
            "description": (
                "Run the Outreach agent: generate a personalized WhatsApp message for ONE customer. "
                "Will auto-recommend a product if none was generated yet. 'tone' is optional copy hint."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_ref": {"type": "string"},
                    "tone": {"type": "string"},
                },
                "required": ["customer_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp",
            "description": (
                "Run the Campaign agent: simulate sending a WhatsApp message for ONE customer. "
                "Uses the previously generated message if 'message' is omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_ref": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["customer_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_all_pending",
            "description": "Dispatch every generated outreach message in this conversation that hasn't been sent yet.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_full_workflow",
            "description": (
                "Execute the original end-to-end pipeline (Discovery → Scoring → Recommendation → "
                "Outreach → Campaign). Use this only when the RM explicitly asks for a 'full campaign'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_type": {"type": "string", "enum": ["PERSONAL", "HOME", "CAR", "EDUCATION"]},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_customer",
            "description": "Explain why a customer was selected/scored - returns score features + product rationale.",
            "parameters": {
                "type": "object",
                "properties": {"customer_ref": {"type": "string"}},
                "required": ["customer_ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_summary",
            "description": "Show what's currently in the conversation's working memory (counts + last run id).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
