"""
Thin HTTP client used by every Streamlit page.

Wraps the FastAPI backend defined in backend/api/routes.py and surfaces
helpful error messages to the UI layer.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _base_url() -> str:
    return os.environ.get("BACKEND_BASE_URL", "http://localhost:8000")


def _request(method: str, path: str, *, json: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None, timeout: int = 120) -> Dict[str, Any]:
    url = _base_url().rstrip("/") + path
    try:
        resp = requests.request(method, url, json=json, params=params, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach backend at {url}. Is FastAPI running?") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = resp.text
        raise RuntimeError(f"Backend error ({resp.status_code}): {detail}")
    return resp.json()


# =====================================================
# Health
# =====================================================
def health() -> Dict[str, Any]:
    return _request("GET", "/health", timeout=10)


# =====================================================
# Chat
# =====================================================
def chat(
    *,
    message: str,
    history: Optional[list] = None,
    rm_name: Optional[str] = None,
    conv_id: Optional[str] = None,
    top_n_customers: Optional[int] = None,
    min_conversion_threshold: Optional[float] = None,
    loan_type: str = "PERSONAL",
) -> Dict[str, Any]:
    payload = {
        "message": message,
        "history": history or [],
        "rm_name": rm_name,
        "conv_id": conv_id,
        "top_n_customers": top_n_customers,
        "min_conversion_threshold": min_conversion_threshold,
        "loan_type": loan_type,
    }
    return _request("POST", "/chat", json=payload, timeout=600)


def reset_chat(conv_id: str) -> Dict[str, Any]:
    return _request("POST", "/chat/reset", json={"conv_id": conv_id}, timeout=10)


# =====================================================
# Workflow
# =====================================================
def run_workflow(
    *,
    user_query: str,
    rm_name: Optional[str] = None,
    top_n_customers: Optional[int] = None,
    min_conversion_threshold: Optional[float] = None,
    loan_type: str = "PERSONAL",
) -> Dict[str, Any]:
    payload = {
        "user_query": user_query,
        "rm_name": rm_name,
        "top_n_customers": top_n_customers,
        "min_conversion_threshold": min_conversion_threshold,
        "loan_type": loan_type,
    }
    return _request("POST", "/workflow/run", json=payload, timeout=600)


def rerun_step(run_id: str, step_name: str) -> Dict[str, Any]:
    return _request("POST", f"/workflow/{run_id}/step", json={"step_name": step_name}, timeout=600)


def get_run(run_id: str) -> Dict[str, Any]:
    return _request("GET", f"/workflow/{run_id}")


def list_runs() -> Dict[str, Any]:
    return _request("GET", "/workflow/runs")


# =====================================================
# Customers
# =====================================================
def list_customers(
    *,
    segment: Optional[str] = None,
    min_credit_score: int = 0,
    min_income: float = 0.0,
    limit: int = 50,
) -> Dict[str, Any]:
    return _request(
        "GET",
        "/customers",
        params={
            "segment": segment,
            "min_credit_score": min_credit_score,
            "min_income": min_income,
            "limit": limit,
        },
    )


def get_customer_profile(customer_id: int) -> Dict[str, Any]:
    return _request("GET", f"/customers/{customer_id}")


# =====================================================
# Logs / messages / campaigns
# =====================================================
def get_messages(run_id: str) -> Dict[str, Any]:
    return _request("GET", f"/messages/{run_id}")


def get_campaigns(run_id: str) -> Dict[str, Any]:
    return _request("GET", f"/campaigns/{run_id}")


def get_logs(*, run_id: Optional[str] = None, agent_name: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
    return _request("GET", "/logs", params={"run_id": run_id, "agent_name": agent_name, "limit": limit})


# =====================================================
# Tools / MCP introspection
# =====================================================
def list_tools() -> Dict[str, Any]:
    return _request("GET", "/tools")


def list_tools_for_role(role: str) -> Dict[str, Any]:
    return _request("GET", f"/tools/{role}")
