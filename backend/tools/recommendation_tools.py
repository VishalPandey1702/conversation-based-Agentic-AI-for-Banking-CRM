"""
Product recommendation tool.

Selects the best-fit product from a static catalog based on customer
attributes and conversion probability. Returns a structured rationale.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.services.logging_service import get_logger
from backend.utils.constants import PRODUCT_CATALOG

logger = get_logger(__name__)


def _eligible(product: Dict[str, Any], customer: Dict[str, Any]) -> bool:
    """Hard eligibility gates: income + credit score must meet floors."""
    return (
        float(customer.get("annual_income", 0.0)) >= float(product["min_income"])
        and int(customer.get("credit_score", 0)) >= int(product["min_credit_score"])
    )


def _amount_for(product: Dict[str, Any], customer: Dict[str, Any]) -> float:
    """
    Recommend a sensible amount: bounded by the product cap and customer's
    income/balance profile.
    """
    income = float(customer.get("annual_income", 0.0))
    balance = float(customer.get("account_balance", 0.0))
    base = min(income * 0.6, balance * 4.0)
    capped = min(base, float(product["max_amount"]))
    return round(max(capped, 50_000.0), -3)  # round to nearest 1000


def recommend_product(
    *,
    customer: Dict[str, Any],
    conversion_probability: float = 0.5,
    preferred_audience: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pick a product for a single customer.

    Returns dict:
        {
            "customer_id": int,
            "product_code": str,
            "product_name": str,
            "recommended_amount": float,
            "interest_rate": float,
            "tenure_months": int,
            "confidence": float,  # 0..1
            "rationale": str,
            "eligible_products": [...]
        }
    """
    eligible = [p for p in PRODUCT_CATALOG if _eligible(p, customer)]

    if not eligible:
        return {
            "customer_id": customer.get("id"),
            "product_code": None,
            "product_name": None,
            "recommended_amount": 0.0,
            "interest_rate": 0.0,
            "tenure_months": 0,
            "confidence": 0.0,
            "rationale": "No catalog product meets the income/credit thresholds.",
            "eligible_products": [],
        }

    # Prefer products whose audience tier matches the customer segment.
    target_segment = preferred_audience or customer.get("customer_segment")
    eligible.sort(
        key=lambda p: (
            0 if p["audience"] == target_segment else 1,  # segment match first
            -float(p["min_income"]),                       # then richest tier
        )
    )
    best = eligible[0]

    amount = _amount_for(best, customer)

    rationale = (
        f"Selected {best['product_name']} because the customer meets the "
        f"income (>= ₹{best['min_income']:,.0f}) and credit score "
        f"(>= {best['min_credit_score']}) gates, and product audience "
        f"'{best['audience']}' matches segment '{customer.get('customer_segment')}'."
    )

    return {
        "customer_id": customer.get("id"),
        "product_code": best["product_code"],
        "product_name": best["product_name"],
        "recommended_amount": amount,
        "interest_rate": best["interest_rate"],
        "tenure_months": best["tenure_months"],
        "confidence": round(min(1.0, conversion_probability + 0.1), 4),
        "rationale": rationale,
        "eligible_products": [p["product_code"] for p in eligible],
    }


def list_catalog() -> List[Dict[str, Any]]:
    """Expose the full catalog (read-only)."""
    return list(PRODUCT_CATALOG)
