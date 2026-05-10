"""
Transaction analysis tools.

Tools exposed:
    analyze_monthly_spending      - average monthly spend + category breakdown
    analyze_balance_patterns      - balance trend + volatility
    analyze_transaction_frequency - txn density + consistency

These tools are deterministic and contain no LLM calls. Their output is
consumed by the scoring agent to compute conversion probability features.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Any, Dict, List

from sqlalchemy import desc

from backend.database.db import session_scope
from backend.database.models import Transaction
from backend.services.logging_service import get_logger

logger = get_logger(__name__)


# =====================================================
# Helpers
# =====================================================
def _month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def _fetch_recent_transactions(customer_id: int, days: int) -> List[Transaction]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    with session_scope() as s:
        rows = (
            s.query(Transaction)
            .filter(Transaction.customer_id == customer_id, Transaction.transaction_date >= cutoff)
            .order_by(desc(Transaction.transaction_date))
            .all()
        )
        # Detach so the caller can use them after the session closes.
        # We extract primitive fields rather than expunge.
        return rows


# =====================================================
# Tool: analyze_monthly_spending
# =====================================================
def analyze_monthly_spending(customer_id: int, months: int = 6) -> Dict[str, Any]:
    """
    Return the per-month spend (DEBIT only) and category breakdown.

    Output:
        {
            "avg_monthly_spend": float,
            "monthly_spend": {"YYYY-MM": float, ...},
            "category_breakdown": {"SHOPPING": 12345.67, ...},
            "n_months": int,
            "n_transactions": int
        }
    """
    days = months * 31
    rows = _fetch_recent_transactions(customer_id, days)

    monthly_spend: Dict[str, float] = defaultdict(float)
    category_breakdown: Dict[str, float] = defaultdict(float)
    debit_count = 0

    for r in rows:
        if r.transaction_type != "DEBIT":
            continue
        debit_count += 1
        mk = _month_key(r.transaction_date)
        monthly_spend[mk] += float(r.amount)
        category_breakdown[r.category] += float(r.amount)

    spend_values = list(monthly_spend.values())
    avg = round(mean(spend_values), 2) if spend_values else 0.0

    return {
        "customer_id": customer_id,
        "avg_monthly_spend": avg,
        "monthly_spend": {k: round(v, 2) for k, v in sorted(monthly_spend.items())},
        "category_breakdown": {k: round(v, 2) for k, v in sorted(category_breakdown.items())},
        "n_months": len(spend_values),
        "n_transactions": debit_count,
    }


# =====================================================
# Tool: analyze_balance_patterns
# =====================================================
def analyze_balance_patterns(customer_id: int, days: int = 90) -> Dict[str, Any]:
    """
    Examine the post-transaction balance series to estimate stability.

    Output:
        {
            "min_balance": float,
            "max_balance": float,
            "avg_balance": float,
            "balance_volatility": float,   # std dev / avg, lower is better
            "trend": "INCREASING"|"DECREASING"|"STABLE",
            "n_samples": int
        }
    """
    rows = _fetch_recent_transactions(customer_id, days)
    balances = [float(r.balance_after) for r in rows]

    if not balances:
        return {
            "customer_id": customer_id,
            "min_balance": 0.0,
            "max_balance": 0.0,
            "avg_balance": 0.0,
            "balance_volatility": 0.0,
            "trend": "STABLE",
            "n_samples": 0,
        }

    avg_bal = mean(balances)
    std_bal = pstdev(balances) if len(balances) > 1 else 0.0
    volatility = round(std_bal / avg_bal, 4) if avg_bal else 0.0

    # rows came back DESC by date, so balances[0] is most recent.
    recent_avg = mean(balances[: max(1, len(balances) // 3)])
    older_avg = mean(balances[-max(1, len(balances) // 3) :])
    if recent_avg > older_avg * 1.05:
        trend = "INCREASING"
    elif recent_avg < older_avg * 0.95:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    return {
        "customer_id": customer_id,
        "min_balance": round(min(balances), 2),
        "max_balance": round(max(balances), 2),
        "avg_balance": round(avg_bal, 2),
        "balance_volatility": volatility,
        "trend": trend,
        "n_samples": len(balances),
    }


# =====================================================
# Tool: analyze_transaction_frequency
# =====================================================
def analyze_transaction_frequency(customer_id: int, days: int = 90) -> Dict[str, Any]:
    """
    Count transactions per month and estimate consistency.

    Output:
        {
            "total_transactions": int,
            "transactions_per_month": {"YYYY-MM": int, ...},
            "avg_transactions_per_month": float,
            "consistency_score": float   # 0..1, higher is steadier
        }
    """
    rows = _fetch_recent_transactions(customer_id, days)

    per_month: Dict[str, int] = defaultdict(int)
    for r in rows:
        per_month[_month_key(r.transaction_date)] += 1

    counts = list(per_month.values())
    if not counts:
        return {
            "customer_id": customer_id,
            "total_transactions": 0,
            "transactions_per_month": {},
            "avg_transactions_per_month": 0.0,
            "consistency_score": 0.0,
        }

    avg = mean(counts)
    std = pstdev(counts) if len(counts) > 1 else 0.0
    # Lower std relative to avg -> higher consistency. Clamp to [0,1].
    if avg <= 0:
        consistency = 0.0
    else:
        consistency = max(0.0, min(1.0, 1.0 - (std / avg)))

    return {
        "customer_id": customer_id,
        "total_transactions": len(rows),
        "transactions_per_month": {k: per_month[k] for k in sorted(per_month)},
        "avg_transactions_per_month": round(avg, 2),
        "consistency_score": round(consistency, 4),
    }
