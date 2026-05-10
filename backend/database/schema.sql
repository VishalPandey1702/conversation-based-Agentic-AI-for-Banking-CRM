-- ============================================================
-- Banking CRM - Reference SQL Schema
--
-- This schema is generated and maintained at runtime by SQLAlchemy
-- (see backend/database/models.py). This file is provided for
-- documentation, reviewers, and for external SQL tooling.
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_code               VARCHAR(20)  NOT NULL UNIQUE,
    full_name                   VARCHAR(120) NOT NULL,
    email                       VARCHAR(120) NOT NULL,
    phone                       VARCHAR(25)  NOT NULL,
    city                        VARCHAR(80)  NOT NULL,
    age                         INTEGER      NOT NULL,
    occupation                  VARCHAR(80)  NOT NULL,
    annual_income               REAL         NOT NULL,
    monthly_salary              REAL         NOT NULL,
    account_balance             REAL         NOT NULL,
    credit_score                INTEGER      NOT NULL,
    account_type                VARCHAR(40)  NOT NULL,
    customer_segment            VARCHAR(40)  NOT NULL,
    has_existing_loan           BOOLEAN      DEFAULT 0,
    previous_repayment_score    REAL         DEFAULT 1.0,
    onboarding_date             DATETIME,
    created_at                  DATETIME
);

CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         INTEGER NOT NULL,
    transaction_date    DATETIME,
    amount              REAL    NOT NULL,
    transaction_type    VARCHAR(20) NOT NULL,
    category            VARCHAR(40) NOT NULL,
    merchant            VARCHAR(120),
    balance_after       REAL    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE TABLE IF NOT EXISTS loan_inquiries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL,
    loan_type       VARCHAR(40)  NOT NULL,
    inquiry_amount  REAL         NOT NULL,
    inquiry_date    DATETIME,
    channel         VARCHAR(40)  NOT NULL,
    status          VARCHAR(40)  DEFAULT 'OPEN',
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE TABLE IF NOT EXISTS crm_interactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         INTEGER NOT NULL,
    interaction_date    DATETIME,
    channel             VARCHAR(40)  NOT NULL,
    notes               TEXT,
    sentiment           VARCHAR(20)  DEFAULT 'NEUTRAL',
    rm_name             VARCHAR(80),
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         INTEGER NOT NULL,
    product_code        VARCHAR(40)  NOT NULL,
    product_name        VARCHAR(120) NOT NULL,
    recommended_amount  REAL         NOT NULL,
    interest_rate       REAL         NOT NULL,
    tenure_months       INTEGER      NOT NULL,
    rationale           TEXT,
    confidence          REAL         DEFAULT 0.0,
    created_at          DATETIME,
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE TABLE IF NOT EXISTS whatsapp_campaigns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER NOT NULL,
    phone           VARCHAR(25) NOT NULL,
    message         TEXT NOT NULL,
    status          VARCHAR(20) DEFAULT 'PENDING',
    sent_at         DATETIME,
    campaign_run_id VARCHAR(60),
    FOREIGN KEY (customer_id) REFERENCES customers (id)
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          VARCHAR(60) NOT NULL,
    agent_name      VARCHAR(80) NOT NULL,
    tool_name       VARCHAR(80),
    step_name       VARCHAR(80),
    status          VARCHAR(20) NOT NULL,
    reasoning       TEXT,
    input_payload   TEXT,
    output_payload  TEXT,
    error_message   TEXT,
    duration_ms     REAL DEFAULT 0.0,
    timestamp       DATETIME
);
