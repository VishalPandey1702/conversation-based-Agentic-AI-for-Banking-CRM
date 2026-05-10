"""
Customer database tools.

These tools encapsulate **all** read access to the Customer / LoanInquiry /
CRMInteraction tables and return Pydantic-validated dictionaries.

Tools exposed:
    fetch_customer_by_id
    fetch_high_income_customers
    fetch_recent_loan_inquiries
    fetch_customer_profile

Each tool is:
- pure (no side effects)
- transactional (opens a short-lived session)
- structured (returns dicts, not ORM objects)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import desc, func

from backend.database.db import session_scope
from backend.database.models import Customer, LoanInquiry, CRMInteraction, Transaction
from backend.services.logging_service import get_logger

logger = get_logger(__name__)


# =====================================================
# Pydantic schemas (the "contract" of these tools)
# =====================================================
class CustomerSummary(BaseModel):
    id: int
    customer_code: str
    full_name: str
    email: str
    phone: str
    city: str
    age: int
    occupation: str
    annual_income: float
    monthly_salary: float
    account_balance: float
    credit_score: int
    account_type: str
    customer_segment: str
    has_existing_loan: bool
    previous_repayment_score: float


class LoanInquirySchema(BaseModel):
    id: int
    customer_id: int
    loan_type: str
    inquiry_amount: float
    inquiry_date: datetime
    channel: str
    status: str


class CustomerProfile(BaseModel):
    customer: CustomerSummary
    recent_inquiries: List[LoanInquirySchema] = Field(default_factory=list)
    recent_interactions: List[Dict[str, Any]] = Field(default_factory=list)
    transaction_count_90d: int = 0


# =====================================================
# Implementation
# =====================================================
def _customer_to_summary(c: Customer) -> Dict[str, Any]:
    return CustomerSummary(
        id=c.id,
        customer_code=c.customer_code,
        full_name=c.full_name,
        email=c.email,
        phone=c.phone,
        city=c.city,
        age=c.age,
        occupation=c.occupation,
        annual_income=c.annual_income,
        monthly_salary=c.monthly_salary,
        account_balance=c.account_balance,
        credit_score=c.credit_score,
        account_type=c.account_type,
        customer_segment=c.customer_segment,
        has_existing_loan=bool(c.has_existing_loan),
        previous_repayment_score=float(c.previous_repayment_score or 0.0),
    ).model_dump()


def fetch_customer_by_id(customer_id: int) -> Optional[Dict[str, Any]]:
    """Return a single customer summary, or None if not found."""
    with session_scope() as s:
        c = s.query(Customer).filter(Customer.id == customer_id).first()
        if c is None:
            logger.warning("Customer id=%s not found", customer_id)
            return None
        return _customer_to_summary(c)


def fetch_high_income_customers(
    *,
    min_annual_income: float = 600_000.0,
    min_credit_score: int = 680,
    only_segments: Optional[List[str]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Return the top customers above the income/credit thresholds.

    Args:
        min_annual_income: Floor for annual income.
        min_credit_score:  Floor for the credit score.
        only_segments:     If provided, restrict to these segments (e.g. ['HIGH','MEDIUM']).
        limit:             Max rows to return.
    """
    with session_scope() as s:
        q = s.query(Customer).filter(
            Customer.annual_income >= min_annual_income,
            Customer.credit_score >= min_credit_score,
        )
        if only_segments:
            q = q.filter(Customer.customer_segment.in_(only_segments))
        q = q.order_by(desc(Customer.annual_income)).limit(limit)
        rows = q.all()
        return [_customer_to_summary(c) for c in rows]


def fetch_recent_loan_inquiries(
    *,
    days: int = 60,
    loan_type: Optional[str] = None,
    customer_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Return recent loan inquiries (default last 60 days).

    Args:
        days: Lookback window.
        loan_type: Optional filter (PERSONAL / HOME / CAR / EDUCATION).
        customer_ids: Optional whitelist of customer ids.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    with session_scope() as s:
        q = s.query(LoanInquiry).filter(LoanInquiry.inquiry_date >= cutoff)
        if loan_type:
            q = q.filter(LoanInquiry.loan_type == loan_type)
        if customer_ids:
            q = q.filter(LoanInquiry.customer_id.in_(customer_ids))
        q = q.order_by(desc(LoanInquiry.inquiry_date))
        rows = q.all()
        return [
            LoanInquirySchema(
                id=r.id,
                customer_id=r.customer_id,
                loan_type=r.loan_type,
                inquiry_amount=r.inquiry_amount,
                inquiry_date=r.inquiry_date,
                channel=r.channel,
                status=r.status,
            ).model_dump()
            for r in rows
        ]


def fetch_customer_profile(customer_id: int) -> Optional[Dict[str, Any]]:
    """
    Build a 360-degree customer profile combining:
    - core customer attributes
    - recent loan inquiries (last 90 days)
    - recent CRM interactions (last 90 days)
    - 90-day transaction count
    """
    cutoff = datetime.utcnow() - timedelta(days=90)
    with session_scope() as s:
        customer = s.query(Customer).filter(Customer.id == customer_id).first()
        if customer is None:
            return None

        recent_inquiries = (
            s.query(LoanInquiry)
            .filter(
                LoanInquiry.customer_id == customer_id,
                LoanInquiry.inquiry_date >= cutoff,
            )
            .order_by(desc(LoanInquiry.inquiry_date))
            .all()
        )

        recent_interactions = (
            s.query(CRMInteraction)
            .filter(
                CRMInteraction.customer_id == customer_id,
                CRMInteraction.interaction_date >= cutoff,
            )
            .order_by(desc(CRMInteraction.interaction_date))
            .all()
        )

        txn_count = (
            s.query(func.count(Transaction.id))
            .filter(
                Transaction.customer_id == customer_id,
                Transaction.transaction_date >= cutoff,
            )
            .scalar()
            or 0
        )

        profile = CustomerProfile(
            customer=CustomerSummary(**_customer_to_summary(customer)),
            recent_inquiries=[
                LoanInquirySchema(
                    id=i.id,
                    customer_id=i.customer_id,
                    loan_type=i.loan_type,
                    inquiry_amount=i.inquiry_amount,
                    inquiry_date=i.inquiry_date,
                    channel=i.channel,
                    status=i.status,
                )
                for i in recent_inquiries
            ],
            recent_interactions=[
                {
                    "id": x.id,
                    "interaction_date": x.interaction_date.isoformat(),
                    "channel": x.channel,
                    "notes": x.notes,
                    "sentiment": x.sentiment,
                    "rm_name": x.rm_name,
                }
                for x in recent_interactions
            ],
            transaction_count_90d=int(txn_count),
        )
        return profile.model_dump()
