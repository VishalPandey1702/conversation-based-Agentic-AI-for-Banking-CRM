"""
Deterministic, explainable scoring tool.

This is intentionally NOT an LLM call. It uses a transparent, weighted
linear model so the rationale can be returned alongside the score.

Score features (each normalized to [0,1]):
    - income_factor              (annual income vs targets)
    - balance_factor             (account balance vs salary)
    - credit_factor              (credit score in [600..820])
    - repayment_history_factor   (previous repayment score)
    - txn_consistency_factor     (transaction consistency score)
    - inquiry_recency_factor     (any recent loan inquiry boosts)
    - existing_loan_penalty      (negative for already-loaded customers)

Conversion probability is a calibrated logistic of the weighted score.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from backend.services.logging_service import get_logger
from backend.utils.helpers import clamp

logger = get_logger(__name__)


# =====================================================
# Feature weights (sum doesn't have to be 1; we logistic at the end)
# =====================================================
WEIGHTS: Dict[str, float] = {
    "income_factor": 1.4,
    "balance_factor": 1.0,
    "credit_factor": 1.6,
    "repayment_history_factor": 1.2,
    "txn_consistency_factor": 0.9,
    "inquiry_recency_factor": 1.3,
    "existing_loan_penalty": -0.6,
}

# Reference points for normalization
INCOME_TARGET_HIGH = 2_500_000.0
INCOME_TARGET_LOW = 300_000.0
CREDIT_FLOOR = 600.0
CREDIT_CEIL = 820.0


def _normalize_income(annual_income: float) -> float:
    if annual_income <= INCOME_TARGET_LOW:
        return 0.0
    if annual_income >= INCOME_TARGET_HIGH:
        return 1.0
    return (annual_income - INCOME_TARGET_LOW) / (INCOME_TARGET_HIGH - INCOME_TARGET_LOW)


def _normalize_balance(balance: float, monthly_salary: float) -> float:
    if monthly_salary <= 0:
        return 0.0
    # 0 salaries -> 0, 6+ salaries -> 1
    return clamp(balance / (monthly_salary * 6.0))


def _normalize_credit(credit_score: int) -> float:
    if credit_score <= CREDIT_FLOOR:
        return 0.0
    if credit_score >= CREDIT_CEIL:
        return 1.0
    return (credit_score - CREDIT_FLOOR) / (CREDIT_CEIL - CREDIT_FLOOR)


def _logistic(x: float) -> float:
    """Map (-inf, +inf) into (0, 1)."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def compute_conversion_score(
    *,
    customer: Dict[str, Any],
    txn_consistency: float = 0.5,
    has_recent_inquiry: bool = False,
    inquiry_recency_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute a transparent conversion score for a single customer.

    Args:
        customer:               CustomerSummary-like dict (see customer_tools).
        txn_consistency:        0..1 from analyze_transaction_frequency.
        has_recent_inquiry:     True if a loan inquiry exists within 60 days.
        inquiry_recency_days:   How recent the most recent inquiry was.

    Returns dict:
        {
            "customer_id": int,
            "score": float,                # raw weighted sum
            "conversion_probability": float,  # logistic(score)
            "features": {...},                # per-feature contributions
            "rationale": str,                 # human-readable reasoning
        }
    """
    annual_income = float(customer.get("annual_income", 0.0))
    balance = float(customer.get("account_balance", 0.0))
    monthly_salary = float(customer.get("monthly_salary", 0.0))
    credit_score = int(customer.get("credit_score", 0))
    repayment = float(customer.get("previous_repayment_score", 0.0))
    has_loan = bool(customer.get("has_existing_loan", False))

    # Normalized features
    f_income = _normalize_income(annual_income)
    f_balance = _normalize_balance(balance, monthly_salary)
    f_credit = _normalize_credit(credit_score)
    f_repay = clamp(repayment)
    f_txn = clamp(txn_consistency)

    # Inquiry recency: fresher = higher
    if has_recent_inquiry:
        if inquiry_recency_days is None:
            f_inq = 0.6
        else:
            f_inq = clamp(1.0 - (inquiry_recency_days / 60.0))
    else:
        f_inq = 0.0

    # Existing loan penalty (already a "negative" feature)
    f_existing_loan = 1.0 if has_loan else 0.0

    contributions = {
        "income_factor": round(WEIGHTS["income_factor"] * f_income, 4),
        "balance_factor": round(WEIGHTS["balance_factor"] * f_balance, 4),
        "credit_factor": round(WEIGHTS["credit_factor"] * f_credit, 4),
        "repayment_history_factor": round(WEIGHTS["repayment_history_factor"] * f_repay, 4),
        "txn_consistency_factor": round(WEIGHTS["txn_consistency_factor"] * f_txn, 4),
        "inquiry_recency_factor": round(WEIGHTS["inquiry_recency_factor"] * f_inq, 4),
        "existing_loan_penalty": round(WEIGHTS["existing_loan_penalty"] * f_existing_loan, 4),
    }

    raw = sum(contributions.values())
    # Center the logistic around the midpoint of typical positive scores (~2.0)
    prob = _logistic(raw - 2.0)

    rationale = _build_rationale(
        customer=customer,
        contributions=contributions,
        prob=prob,
        has_recent_inquiry=has_recent_inquiry,
        has_loan=has_loan,
    )

    return {
        "customer_id": customer.get("id"),
        "customer_code": customer.get("customer_code"),
        "score": round(raw, 4),
        "conversion_probability": round(prob, 4),
        "features": {
            "raw": {
                "income_factor": round(f_income, 4),
                "balance_factor": round(f_balance, 4),
                "credit_factor": round(f_credit, 4),
                "repayment_history_factor": round(f_repay, 4),
                "txn_consistency_factor": round(f_txn, 4),
                "inquiry_recency_factor": round(f_inq, 4),
                "existing_loan_penalty": round(f_existing_loan, 4),
            },
            "contributions": contributions,
        },
        "rationale": rationale,
    }


def _build_rationale(
    *,
    customer: Dict[str, Any],
    contributions: Dict[str, float],
    prob: float,
    has_recent_inquiry: bool,
    has_loan: bool,
) -> str:
    """Compose a short human-readable explanation of the score."""
    parts: List[str] = []
    name = customer.get("full_name", "Customer")
    seg = customer.get("customer_segment", "?")
    parts.append(f"{name} ({seg} segment)")

    top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:3]
    drivers = ", ".join([f"{k.replace('_', ' ')}={v:+.2f}" for k, v in top])
    parts.append(f"top drivers: {drivers}")

    if has_recent_inquiry:
        parts.append("recent loan inquiry boosts intent")
    if has_loan:
        parts.append("an existing loan slightly reduces probability")
    parts.append(f"final conversion probability ≈ {prob*100:.1f}%")
    return "; ".join(parts) + "."
