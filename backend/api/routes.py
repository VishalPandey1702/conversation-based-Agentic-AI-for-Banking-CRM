"""
FastAPI routes for the Banking CRM agentic system.

Exposes:
    POST /chat                         -> conversational entry-point (intent routing)

    POST /workflow/run                 -> run full workflow
    POST /workflow/{run_id}/step       -> rerun a single step
    GET  /workflow/{run_id}            -> fetch cached run state
    GET  /workflow/runs                -> list known run ids

    GET  /customers                    -> list customers (paged + filterable)
    GET  /customers/{cid}              -> 360° profile

    GET  /messages/{run_id}            -> generated outreach messages for a run
    GET  /campaigns/{run_id}           -> campaign results for a run

    GET  /logs                         -> agent_logs (filterable)
    GET  /tools                        -> MCP tool catalog
    GET  /tools/{role}                 -> tools accessible to a role
    GET  /health                       -> service heartbeat
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.agents.conversational_orchestrator import conversational_orchestrator
from backend.agents.orchestrator_tools import sessions as orchestrator_sessions
from backend.database.db import session_scope
from backend.database.models import Customer
from backend.mcp.server import mcp_server
from backend.tools import audit_tools, customer_tools
from backend.utils.config import settings
from backend.utils.constants import APP_NAME, AgentRole
from backend.workflows.execution_manager import execution_manager

router = APIRouter()


# =====================================================
# Request / response schemas
# =====================================================
class RunWorkflowRequest(BaseModel):
    user_query: str = Field(..., description="Natural-language RM request.")
    rm_name: Optional[str] = Field(default=None)
    top_n_customers: Optional[int] = Field(default=None, ge=1, le=200)
    min_conversion_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    loan_type: str = Field(default="PERSONAL")


class RerunStepRequest(BaseModel):
    step_name: str = Field(..., description="Workflow step to rerun.")


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="Latest user chat message.")
    history: List[ChatHistoryItem] = Field(default_factory=list)
    rm_name: Optional[str] = Field(default=None)
    conv_id: Optional[str] = Field(default=None, description="Conversation id (server-side memory).")
    # Optional defaults the orchestrator may use
    top_n_customers: Optional[int] = Field(default=None, ge=1, le=200)
    min_conversion_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    loan_type: str = Field(default="PERSONAL")


class ResetConversationRequest(BaseModel):
    conv_id: str


# =====================================================
# Health / metadata
# =====================================================
@router.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + LLM availability probe."""
    return {
        "status": "ok",
        "app": APP_NAME,
        "env": settings.APP_ENV,
        "llm_configured": settings.llm_configured,
    }


# =====================================================
# Conversational entry-point
# =====================================================
@router.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    """
    Conversational entry-point — backed by the ConversationalOrchestrator.

    The orchestrator owns one LLM with tool-calling. Available tools are the
    specialist agents (Discovery, Scoring, Recommendation, Outreach, Campaign)
    plus DB lookups, all governed by RBAC + audit. The orchestrator decides
    which tools to call dynamically based on the RM's natural-language request.
    """
    return conversational_orchestrator.chat(
        conv_id=req.conv_id,
        user_message=req.message,
        history=[h.model_dump() for h in req.history],
        rm_name=req.rm_name,
    )


@router.post("/chat/reset")
def reset_chat(req: ResetConversationRequest) -> Dict[str, Any]:
    """Drop a conversation's server-side memory."""
    orchestrator_sessions.reset(req.conv_id)
    return {"ok": True, "conv_id": req.conv_id}


# =====================================================
# Workflow endpoints
# =====================================================
@router.post("/workflow/run")
def run_workflow(req: RunWorkflowRequest) -> Dict[str, Any]:
    """Run the entire multi-agent workflow end-to-end."""
    final_state = execution_manager.run_full_workflow(
        user_query=req.user_query,
        rm_name=req.rm_name,
        top_n_customers=req.top_n_customers,
        min_conversion_threshold=req.min_conversion_threshold,
        loan_type=req.loan_type,
    )
    return _state_to_payload(final_state)


@router.post("/workflow/{run_id}/step")
def rerun_step(run_id: str, req: RerunStepRequest) -> Dict[str, Any]:
    """Rerun a single named step against the cached state of `run_id`."""
    try:
        state = execution_manager.run_single_step(step_name=req.step_name, run_id=run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _state_to_payload(state)


@router.get("/workflow/runs")
def list_runs() -> Dict[str, Any]:
    return {"runs": execution_manager.list_runs()}


@router.get("/workflow/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    state = execution_manager.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return _state_to_payload(state)


# =====================================================
# Customer endpoints
# =====================================================
@router.get("/customers")
def list_customers(
    segment: Optional[str] = Query(default=None),
    min_credit_score: int = Query(default=0),
    min_income: float = Query(default=0.0),
    limit: int = Query(default=50, ge=1, le=500),
) -> Dict[str, Any]:
    """List customers with simple filters."""
    with session_scope() as s:
        q = s.query(Customer)
        if segment:
            q = q.filter(Customer.customer_segment == segment.upper())
        if min_credit_score:
            q = q.filter(Customer.credit_score >= min_credit_score)
        if min_income:
            q = q.filter(Customer.annual_income >= min_income)
        rows = q.order_by(Customer.annual_income.desc()).limit(limit).all()
        return {
            "count": len(rows),
            "customers": [
                {
                    "id": c.id,
                    "customer_code": c.customer_code,
                    "full_name": c.full_name,
                    "phone": c.phone,
                    "email": c.email,
                    "city": c.city,
                    "occupation": c.occupation,
                    "age": c.age,
                    "annual_income": c.annual_income,
                    "monthly_salary": c.monthly_salary,
                    "account_balance": c.account_balance,
                    "credit_score": c.credit_score,
                    "customer_segment": c.customer_segment,
                    "has_existing_loan": c.has_existing_loan,
                }
                for c in rows
            ],
        }


@router.get("/customers/{customer_id}")
def get_customer_profile(customer_id: int) -> Dict[str, Any]:
    profile = customer_tools.fetch_customer_profile(customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
    return profile


# =====================================================
# Messages / campaigns
# =====================================================
@router.get("/messages/{run_id}")
def get_messages(run_id: str) -> Dict[str, Any]:
    state = execution_manager.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return {"run_id": run_id, "messages": state.get("generated_messages", [])}


@router.get("/campaigns/{run_id}")
def get_campaigns(run_id: str) -> Dict[str, Any]:
    state = execution_manager.get_run(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return {"run_id": run_id, "campaigns": state.get("campaign_results", [])}


# =====================================================
# Logs
# =====================================================
@router.get("/logs")
def get_logs(
    run_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=2000),
) -> Dict[str, Any]:
    rows = audit_tools.fetch_logs(run_id=run_id, agent_name=agent_name, limit=limit)
    return {"count": len(rows), "logs": rows}


# =====================================================
# Tool / MCP introspection
# =====================================================
@router.get("/tools")
def list_tools() -> Dict[str, Any]:
    return {"tools": mcp_server.list_tools()}


@router.get("/tools/{role}")
def list_tools_for_role(role: str) -> Dict[str, Any]:
    valid_roles = [r.value for r in AgentRole]
    if role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"unknown role; valid: {valid_roles}")
    return {"role": role, "tools": mcp_server.list_tools(role=role)}


# =====================================================
# Helpers
# =====================================================
def _state_to_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    """Project a WorkflowState into a JSON-friendly response payload."""
    return {
        "run_id": state.get("run_id"),
        "user_query": state.get("user_query"),
        "rm_name": state.get("rm_name"),
        "loan_type": state.get("loan_type"),
        "top_n_customers": state.get("top_n_customers"),
        "min_conversion_threshold": state.get("min_conversion_threshold"),
        "completed_steps": state.get("completed_steps", []),
        "current_step": state.get("current_step"),
        "summary": state.get("summary"),
        "errors": state.get("errors", []),
        "discovered_customers": state.get("discovered_customers", []),
        "scored_customers": state.get("scored_customers", []),
        "recommendations": state.get("recommendations", []),
        "generated_messages": state.get("generated_messages", []),
        "campaign_results": state.get("campaign_results", []),
        "logs": state.get("logs", []),
    }
