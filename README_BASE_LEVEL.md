# Agentic Banking CRM

> Conversation-based Agentic AI for Banking CRM — built with **Azure OpenAI tool calling**, **LangGraph**, **MCP server architecture**, **FastAPI**, **SQLite**, and **Streamlit**.

A production-style, modular system that lets a Relationship Manager (RM) **talk** to an AI orchestrator. The orchestrator dynamically picks specialist agents (Discovery, Scoring, Recommendation, Outreach, Campaign) as tools to fulfill the RM's request — not a fixed pipeline.

> Example conversation:
> 1. *"Find HIGH segment customers with credit > 750, top 5."*
> 2. *"Score them."*
> 3. *"Generate a WhatsApp message for the top one."*
> 4. *"Send it."*
> 5. *"What did we do so far?"*

Each turn shows the **reasoning trace** — every agent/tool the orchestrator invoked, with arguments and brief results.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Execution Flow](#2-execution-flow)
3. [Agents & Tools](#3-agents--tools)
4. [MCP Server & Governance](#4-mcp-server--governance)
5. [Database Schema](#5-database-schema)
6. [Project Layout](#6-project-layout)
7. [Setup](#7-setup)
8. [Run](#8-run)
9. [Demo Scenarios](#9-demo-scenarios)
10. [Trade-offs & Limitations](#10-trade-offs--limitations)
11. [Future Improvements](#11-future-improvements)

---

## 1. Architecture

```
                ┌────────────────────────────────────────────┐
                │            Streamlit Frontend               │
                │   Chat input · Tool trace · Customer 360°    │
                └──────────────────────┬─────────────────────┘
                                       │ HTTP (JSON)
                                       ▼
                ┌────────────────────────────────────────────┐
                │             FastAPI Backend                  │
                │  /chat · /workflow · /customers · /logs      │
                └──────────────────────┬─────────────────────┘
                                       │
                                       ▼
                ┌────────────────────────────────────────────┐
                │       Conversational Orchestrator            │
                │  one LLM with tool-calling · per-conv memory │
                └──────────────────────┬─────────────────────┘
                                       │ picks tools dynamically
        ┌───────────┬───────────┬──────┴──────┬───────────┬────────────┐
        ▼           ▼           ▼             ▼           ▼            ▼
   list /     Discovery     Scoring     Recommend     Outreach     Campaign
   profile     agent        agent        agent         agent        agent
   tools        │            │            │             │            │
                └────────────┴────────────┼─────────────┴────────────┘
                                          ▼
                ┌────────────────────────────────────────────┐
                │   MCP Server (in-process, RBAC + audit)      │
                │   ToolRegistry · RoleManager · permissions    │
                └──────────────────────┬─────────────────────┘
                          ┌────────────┴────────────┐
                          ▼                         ▼
                 SQLite (banking_crm.db)     Azure OpenAI (gpt-5-nano)
```

### Key design rules

- **One conversational brain.** The `ConversationalOrchestrator` is the only agent exposed to the user. It uses Azure OpenAI tool-calling to decide *which* specialist agents to call and *in what order*, per turn.
- **Specialist agents as tools.** Discovery / Scoring / Recommendation / Outreach / Campaign each remain a single-responsibility agent with its own RBAC role in MCP. The orchestrator calls them through `orchestrator_tools.py` wrappers.
- **Server-side session memory.** A `ConversationSession` keeps the working set across turns: discovered customers, scores, recommendations, drafted messages, dispatched campaigns. The orchestrator can say "rank 2" / "the top one" / `C00042` and resolve correctly.
- **Deterministic specialists.** Scoring uses a transparent weighted-feature model. Recommendation uses catalog gates. Only the Outreach agent calls the LLM for copywriting (with a deterministic fallback).
- **Governance through MCP.** Every tool invocation goes through the MCP server which enforces role-based permissions, times execution, and writes an audit row to `agent_logs`.
- **Original LangGraph pipeline retained.** Available as the `run_full_workflow` tool — used only when the RM explicitly asks for a full campaign.

---

## 2. Execution Flow

```
RM types a message in chat
         │
         ▼
POST /chat (FastAPI)
         │
         ▼
ConversationalOrchestrator
  │  1. Build messages = system + last 8 history + new user turn
  │  2. Call Azure OpenAI with tools=[discover, score, recommend, …]
  │  3. Model returns: { content } or { tool_calls: [...] }
  │  4. If tool_calls: execute each tool against the session, append
  │     "tool" role messages with results, GOTO 2 (max 6 hops)
  │  5. If content: that's the final reply
  │
  ▼
Response { reply, tool_trace[], session_summary }
         │
         ▼
Streamlit chat renders reply + collapsible reasoning trace
```

The orchestrator never runs the entire 5-step pipeline unless the user asks. A turn can use 0, 1, or many tools depending on the request:

| User says | Tools called |
|---|---|
| *"Hi"* | (none) |
| *"List HIGH segment customers"* | `list_customers` |
| *"Find personal loan candidates and score them"* | `discover_loan_candidates` → `score_customers` |
| *"Generate outreach for the top one"* | `generate_outreach` |
| *"Send it"* | `send_whatsapp` |
| *"Send all pending"* | `send_all_pending` |
| *"Run a full campaign for top 5"* | `run_full_workflow` |
| *"Why was rank 3 picked?"* | `explain_customer` |

---

## 3. Agents & Tools

### Conversational orchestrator (only agent the user talks to)

| File | Purpose |
|---|---|
| `backend/agents/conversational_orchestrator.py` | LLM tool-calling loop, per-conversation session memory, audit logging. |
| `backend/agents/orchestrator_tools.py` | 11 function-calling tools that wrap specialist agents + DB lookups. |

### Specialist agents (called as tools by the orchestrator)

| Agent | Role id | Single responsibility |
|---|---|---|
| `CustomerDiscoveryAgent` | `discovery_agent` | Surface candidate customers (income + credit + recent inquiries). |
| `ScoringAgent` | `scoring_agent` | Deterministic conversion-probability scoring with explainability. |
| `RecommendationAgent` | `recommendation_agent` | Best-fit product from catalog (gates: income, credit, segment). |
| `OutreachAgent` | `outreach_agent` | Personalized WhatsApp copy (LLM with deterministic fallback). |
| `CampaignAgent` | `campaign_agent` | Simulate WhatsApp dispatch + persist campaign records. |
| `SupervisorAgent` | `supervisor_agent` | Owns the lifecycle when the **full** LangGraph pipeline is invoked. |

### Tools exposed to the orchestrator (function-calling schemas)

| Tool | Wraps | What it does |
|---|---|---|
| `list_customers` | DB query | Filter the customer master by segment / credit / income. Seeds working set. |
| `get_customer` | `customer_tools` | 360° profile (by id, code, or "top"). |
| `discover_loan_candidates` | `CustomerDiscoveryAgent` | Run the discovery agent; segment-filter the result. |
| `score_customers` | `ScoringAgent` | Score the working set (or a subset by refs). |
| `recommend_product` | `RecommendationAgent` | Pick a catalog product for one customer. |
| `generate_outreach` | `OutreachAgent` | Draft a WhatsApp message for one customer (auto-recommends if needed). |
| `send_whatsapp` | `CampaignAgent` | Dispatch one message (uses stored draft if `message` omitted). |
| `send_all_pending` | `CampaignAgent` | Batch-dispatch every drafted-but-unsent message in this conversation. |
| `run_full_workflow` | `ExecutionManager` | Run the original LangGraph end-to-end pipeline as a shortcut. |
| `explain_customer` | session data | Return score features + recommendation rationale for one customer. |
| `get_session_summary` | session data | Snapshot of working memory (counts + last run id). |

### MCP tools (still used under the hood by specialist agents)

`fetch_customer_by_id`, `fetch_high_income_customers`, `fetch_recent_loan_inquiries`, `fetch_customer_profile`, `analyze_monthly_spending`, `analyze_balance_patterns`, `analyze_transaction_frequency`, `compute_conversion_score`, `recommend_product`, `send_whatsapp_message`, `log_agent_event`.

---

## 4. MCP Server & Governance

The MCP server (`backend/mcp/`) is in-process but mirrors the real protocol:

| File | Responsibility |
|---|---|
| `permissions.py` | Static RBAC map: `{role → {tool_name, ...}}`. |
| `role_manager.py` | `assert_can_invoke(role, tool)` raises `PermissionDeniedError`. |
| `tool_registry.py` | `ToolSpec` dataclass + in-memory registry with metadata. |
| `server.py` | `MCPServer.invoke()` orchestrates lookup → RBAC → timed call → audit → envelope. |

Every invocation returns a structured envelope:

```json
{ "ok": true,  "tool": "compute_conversion_score", "result": {...}, "duration_ms": 12.4 }
{ "ok": false, "tool": "send_whatsapp_message",     "error": "...",  "code": "PERMISSION_DENIED" }
```

Specialist agents are still bound by RBAC even when the orchestrator triggers them. You can introspect the catalog at:

- `GET /tools` — all tools + schemas
- `GET /tools/{role}` — tools visible to a given role
- The **Logs** page in the UI

---

## 5. Database Schema

SQLite with seven tables (defined in `backend/database/models.py`, documented in `backend/database/schema.sql`):

| Table | Purpose |
|---|---|
| `customers` | Master profile + financial attributes (segment, credit, income). |
| `transactions` | ~6 months of transaction history per customer. |
| `loan_inquiries` | Customer-initiated loan inquiries (PERSONAL / HOME / CAR / EDUCATION). |
| `crm_interactions` | RM-customer touch points. |
| `recommendations` | Persisted product recommendations. |
| `whatsapp_campaigns` | Simulated WhatsApp campaign records. |
| `agent_logs` | Centralized audit trail (run_id, agent, tool, status, reasoning, IO payloads). |

Seed data is generated by `backend/database/seed_data.py` using **Faker** (120 customers · 18k+ transactions · loan inquiries · CRM interactions · weighted across HIGH / MEDIUM / LOW segments).

---

## 6. Project Layout

```
project/
├── backend/
│   ├── agents/
│   │   ├── conversational_orchestrator.py  # one agent, talks to user, picks tools
│   │   ├── orchestrator_tools.py           # agents-as-tools + session memory
│   │   ├── supervisor_agent.py             # owns the legacy LangGraph lifecycle
│   │   ├── customer_discovery_agent.py
│   │   ├── scoring_agent.py
│   │   ├── recommendation_agent.py
│   │   ├── outreach_agent.py
│   │   └── campaign_agent.py
│   ├── tools/                              # customer, txn, scoring, reco, whatsapp, audit
│   ├── mcp/                                # in-process MCP server (RBAC + registry)
│   ├── workflows/                          # state, graph (LangGraph), execution_manager
│   ├── database/                           # models, db, schema.sql, seed_data
│   ├── api/                                # routes.py (incl. /chat), server.py
│   ├── services/                           # llm_service (tool calling), logging_service
│   ├── utils/                              # config, constants, helpers
│   ├── requirements.txt
│   ├── .env
│   └── main.py
├── frontend/
│   ├── app.py                              # Streamlit entry
│   ├── api_client.py                       # thin HTTP client
│   ├── components/
│   │   ├── workflow_chat.py                # chat UI + reasoning trace
│   │   ├── orchestration_viz.py            # animated multi-agent graphic
│   │   ├── customer_table.py
│   │   └── message_preview.py
│   └── pages/
│       ├── customers.py
│       └── logs.py
├── README.md                               # Design approach & engineering decisions
├── README_BASE_LEVEL.md                    # This file (setup + run)
└── .gitignore
```

---

## 7. Setup

**Prerequisites:** Python 3.10+

```bash
# 1. Create + activate the virtual environment
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
# venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt
```

**Configure `backend/.env`** (already populated with the assessment values):

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://apps-mal2lne1-eastus2.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5-nano-5
AZURE_OPENAI_MODEL=gpt-5-nano
AZURE_OPENAI_API_VERSION=2024-10-21

DATABASE_URL=sqlite:///banking_crm.db
```

> Azure OpenAI tool calling is **required** for the conversational orchestrator. If the LLM is unreachable, the chat returns a helpful error and the RM can still use the REST workflow endpoints / quick-run buttons. The Outreach agent's copywriting also falls back to a deterministic template.

---

## 8. Run

### Step 1 — Seed the database (once)

```bash
python -m backend.main --seed --no-serve
```

Creates `banking_crm.db` with 120 customers and ~6 months of activity.

### Step 2 — Start the FastAPI backend

```bash
python -m backend.main --reload
# or:  uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000
```

API docs: <http://localhost:8000/docs>

### Step 3 — Start the Streamlit frontend (second terminal)

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

UI: <http://localhost:8501>

### Useful API calls

```bash
# Health
curl http://localhost:8000/health

# Chat (multi-turn — pass conv_id to keep working memory)
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "Find HIGH segment customers with credit > 750, top 5",
    "rm_name": "Priya Sharma"
  }'

# Reset the conversation memory
curl -X POST http://localhost:8000/chat/reset \
  -H 'Content-Type: application/json' \
  -d '{"conv_id": "conv-xxxxxxxxxx"}'

# Run the fixed LangGraph pipeline directly (also available as a chat tool)
curl -X POST http://localhost:8000/workflow/run \
  -H 'Content-Type: application/json' \
  -d '{"user_query":"full personal loan campaign","top_n_customers":5,"loan_type":"PERSONAL"}'

# Tool catalog per role (RBAC introspection)
curl http://localhost:8000/tools/scoring_agent

# Logs / audit
curl 'http://localhost:8000/logs?limit=50'
```

---

## 9. Demo Scenarios

Open the UI and run these as **multi-turn** chats (the same `conv_id` is reused, so each step builds on the previous one).

### Scenario A — Manual specialist orchestration

> 1. *"Find HIGH segment customers with credit > 750, top 5."*
> 2. *"Score them."*
> 3. *"Generate a personalized WhatsApp message for the top one."*
> 4. *"Send it."*
> 5. *"What did we do so far?"*

Expected tool trace per turn: `list_customers → score_customers` · `score_customers` · `generate_outreach` · `send_whatsapp` · `get_session_summary`.

### Scenario B — End-to-end full campaign shortcut

> 1. *"Run a full personal loan campaign for the top 5 customers above 0.6 probability."*
> 2. *"Send all pending."*
> 3. *"Why was the top customer selected?"*

Tool trace: `run_full_workflow` · `send_all_pending` · `explain_customer`.

### Scenario C — Targeted customer enquiry

> 1. *"Show me customer C00042."*
> 2. *"Recommend a product for them."*
> 3. *"Draft a friendlier outreach message."*
> 4. *"Send it."*

Tool trace: `get_customer` · `recommend_product` · `generate_outreach (tone=friendly)` · `send_whatsapp`.

---

## 10. Trade-offs & Limitations

- **LLM dependency for orchestration.** The orchestrator uses Azure OpenAI tool calling to decompose tasks. With the LLM unavailable, the chat surfaces a friendly error and the user must hit `run_full_workflow` directly via REST. Specialist agents (scoring, recommendation, simulated WhatsApp) remain deterministic and offline-capable.
- **Tool-hop cap.** Each chat turn is capped at 6 tool hops to prevent runaway loops. Increase `MAX_TOOL_HOPS` if you need deeper plans.
- **In-process MCP.** Simpler to demo, no separate process. The architecture (registry / role manager / permissions / structured envelopes) maps 1:1 onto a remote MCP server.
- **Session memory is in-process and ephemeral.** Conversation state is held in a dict keyed by `conv_id`. Persisting to Redis / Postgres is straightforward when needed.
- **SQLite + synchronous workflow.** Perfect for the demo; replace `DATABASE_URL` with Postgres and add Celery for long jobs in production. The schema already supports cross-process audit.
- **WhatsApp is simulated.** All sends are persisted to the `whatsapp_campaigns` table, so analytics + UI work end-to-end without a real provider.
- **Deterministic scoring.** A transparent weighted linear model with logistic calibration. LLMs are bad at consistent numerical scoring; we keep that path deterministic and explainable.

---

## 11. Future Improvements

- **Tool streaming** — stream each tool call/result back to the UI as soon as it executes (today the reply renders after the full loop finishes).
- **Hybrid retrieval** — vector store of past CRM interactions to ground outreach tone.
- **Cross-process MCP** — replace the in-process server with a real remote MCP for true multi-tenant governance.
- **Async parallel scoring** — score 50 customers in parallel rather than sequentially.
- **Per-RM identity + attribution** — attach `rm_id` to every audit row.
- **Plan-of-action preview** — let the orchestrator output its planned tool sequence before executing, for human approval.
- **Postgres + Alembic migrations.**
- **Unit + integration tests** for each tool and agent (the architecture is already test-friendly thanks to `session_scope`, `MCPServer` injection, and session registry).

---

### Author's note

The codebase intentionally over-invests in **structure and explainability**. The goal is to demonstrate **agentic reasoning + tool governance + dynamic decomposition** — not to maximize LLM calls. The LLM is the *orchestrator*, not the worker; numerical and policy decisions still live in deterministic tools.
