"""
Seed the SQLite database with realistic-looking banking CRM data.

Generates:
    - 120 customers (mix of HIGH/MEDIUM/LOW segments)
    - 6 months of transaction history per customer
    - Loan inquiries (skewed toward HIGH/MEDIUM segments)
    - CRM interactions
    - Some pre-existing recommendations and campaigns (sparse)

Usage:
    python -m backend.database.seed_data

This is idempotent: it drops and recreates the schema, then inserts data.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta
from typing import List

from faker import Faker

from backend.database.db import init_db, session_scope
from backend.database.models import (
    Customer,
    Transaction,
    LoanInquiry,
    CRMInteraction,
)
from backend.utils.constants import CustomerSegment

logger = logging.getLogger(__name__)
fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)


# =====================================================
# Helpers
# =====================================================
def _segment_for_income(income: float) -> str:
    if income >= 1_500_000:
        return CustomerSegment.HIGH.value
    if income >= 600_000:
        return CustomerSegment.MEDIUM.value
    return CustomerSegment.LOW.value


def _credit_score_for_segment(segment: str) -> int:
    if segment == CustomerSegment.HIGH.value:
        return random.randint(740, 820)
    if segment == CustomerSegment.MEDIUM.value:
        return random.randint(680, 760)
    return random.randint(580, 700)


def _occupation() -> str:
    return random.choice(
        [
            "Software Engineer",
            "Senior Manager",
            "Doctor",
            "Business Owner",
            "Architect",
            "Marketing Lead",
            "Teacher",
            "Sales Executive",
            "Data Scientist",
            "Banker",
            "Consultant",
            "Government Employee",
        ]
    )


def _account_type(segment: str) -> str:
    if segment == CustomerSegment.HIGH.value:
        return random.choice(["PREMIUM", "SALARY"])
    if segment == CustomerSegment.MEDIUM.value:
        return random.choice(["SALARY", "SAVINGS"])
    return "SAVINGS"


def _build_customers(n: int = 120) -> List[Customer]:
    """Build a list of Customer ORM objects without committing."""
    customers: List[Customer] = []
    for i in range(1, n + 1):
        # Bias the distribution: 25% high, 50% medium, 25% low
        bucket = random.random()
        if bucket < 0.25:
            annual_income = random.uniform(1_500_000, 4_500_000)
        elif bucket < 0.75:
            annual_income = random.uniform(600_000, 1_499_999)
        else:
            annual_income = random.uniform(250_000, 599_999)

        monthly_salary = round(annual_income / 12.0, 2)
        segment = _segment_for_income(annual_income)
        balance_multiplier = {"HIGH": 5.0, "MEDIUM": 2.5, "LOW": 1.0}[segment]
        account_balance = round(monthly_salary * balance_multiplier * random.uniform(0.6, 1.4), 2)

        customer = Customer(
            customer_code=f"C{i:05d}",
            full_name=fake.name(),
            email=fake.email(),
            phone=f"+91{random.randint(7000000000, 9999999999)}",
            city=fake.city(),
            age=random.randint(24, 60),
            occupation=_occupation(),
            annual_income=round(annual_income, 2),
            monthly_salary=monthly_salary,
            account_balance=account_balance,
            credit_score=_credit_score_for_segment(segment),
            account_type=_account_type(segment),
            customer_segment=segment,
            has_existing_loan=random.random() < 0.25,
            previous_repayment_score=round(random.uniform(0.7, 1.0), 2)
            if segment != CustomerSegment.LOW.value
            else round(random.uniform(0.4, 0.9), 2),
            onboarding_date=datetime.utcnow() - timedelta(days=random.randint(180, 1800)),
        )
        customers.append(customer)
    return customers


def _build_transactions(customer: Customer, months: int = 6) -> List[Transaction]:
    """Generate ~30 transactions/month for a customer."""
    txns: List[Transaction] = []
    running_balance = customer.account_balance

    today = datetime.utcnow()
    for month_offset in range(months):
        # Salary credit at the start of the month
        salary_date = today - timedelta(days=30 * month_offset + random.randint(0, 4))
        running_balance += customer.monthly_salary
        txns.append(
            Transaction(
                customer_id=customer.id,
                transaction_date=salary_date,
                amount=customer.monthly_salary,
                transaction_type="CREDIT",
                category="SALARY",
                merchant=customer.occupation,
                balance_after=round(running_balance, 2),
            )
        )

        # Random debits/credits across the month
        n_txn = random.randint(15, 35)
        for _ in range(n_txn):
            txn_date = salary_date + timedelta(days=random.randint(0, 28))
            if random.random() < 0.85:
                # debit
                amount = round(random.uniform(200, customer.monthly_salary * 0.25), 2)
                running_balance -= amount
                category = random.choice(
                    ["SHOPPING", "FOOD", "UTILITY", "TRAVEL", "ENTERTAINMENT", "EMI", "GROCERY"]
                )
                txn_type = "DEBIT"
            else:
                amount = round(random.uniform(500, customer.monthly_salary * 0.4), 2)
                running_balance += amount
                category = random.choice(["TRANSFER_IN", "REFUND", "INTEREST"])
                txn_type = "CREDIT"

            txns.append(
                Transaction(
                    customer_id=customer.id,
                    transaction_date=txn_date,
                    amount=amount,
                    transaction_type=txn_type,
                    category=category,
                    merchant=fake.company()[:120],
                    balance_after=round(running_balance, 2),
                )
            )
    return txns


def _build_loan_inquiries(customer: Customer) -> List[LoanInquiry]:
    """High-segment customers are likelier to have recent inquiries."""
    prob_map = {"HIGH": 0.7, "MEDIUM": 0.4, "LOW": 0.15}
    inquiries: List[LoanInquiry] = []
    if random.random() < prob_map[customer.customer_segment]:
        n = random.randint(1, 2)
        for _ in range(n):
            inquiries.append(
                LoanInquiry(
                    customer_id=customer.id,
                    loan_type=random.choice(["PERSONAL", "HOME", "CAR", "EDUCATION"]),
                    inquiry_amount=round(random.uniform(100_000, 2_000_000), 2),
                    inquiry_date=datetime.utcnow() - timedelta(days=random.randint(1, 60)),
                    channel=random.choice(["WEB", "APP", "BRANCH", "CALL"]),
                    status=random.choice(["OPEN", "OPEN", "CLOSED"]),
                )
            )
    return inquiries


def _build_crm_interactions(customer: Customer) -> List[CRMInteraction]:
    n = random.randint(0, 4)
    interactions: List[CRMInteraction] = []
    for _ in range(n):
        interactions.append(
            CRMInteraction(
                customer_id=customer.id,
                interaction_date=datetime.utcnow() - timedelta(days=random.randint(1, 180)),
                channel=random.choice(["CALL", "EMAIL", "WHATSAPP", "VISIT"]),
                notes=fake.sentence(nb_words=12),
                sentiment=random.choice(["POSITIVE", "NEUTRAL", "NEUTRAL", "NEGATIVE"]),
                rm_name=fake.name(),
            )
        )
    return interactions


# =====================================================
# Public seeding API
# =====================================================
def seed(n_customers: int = 120, drop_existing: bool = True) -> dict:
    """
    Re-create the schema and populate it with fake but realistic data.

    Returns counts of inserted rows for each table.
    """
    init_db(drop_existing=drop_existing)

    counts = {"customers": 0, "transactions": 0, "loan_inquiries": 0, "crm_interactions": 0}

    with session_scope() as session:
        customers = _build_customers(n_customers)
        session.add_all(customers)
        session.flush()  # populate customer.id values
        counts["customers"] = len(customers)

        for customer in customers:
            txns = _build_transactions(customer)
            session.add_all(txns)
            counts["transactions"] += len(txns)

            inquiries = _build_loan_inquiries(customer)
            session.add_all(inquiries)
            counts["loan_inquiries"] += len(inquiries)

            interactions = _build_crm_interactions(customer)
            session.add_all(interactions)
            counts["crm_interactions"] += len(interactions)

    logger.info("Seed complete: %s", counts)
    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = seed()
    print(f"Seeded database: {result}")
