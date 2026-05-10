# Agentic Banking CRM — Conversation-Based Multi-Agent System

> A production-style **Agentic AI** system that lets a Relationship Manager (RM) **talk** to a banking assistant. One conversational orchestrator dynamically calls specialist agents (Discovery, Scoring, Recommendation, Outreach, Campaign) as tools — selecting, sequencing, and looping over them via Azure OpenAI tool calling — and explains every decision it makes.
>
> **Stack:** Python 3.10 · Azure OpenAI (tool calling) · LangGraph · MCP server (in-process, RBAC + audit) · FastAPI · SQLite + SQLAlchemy · Streamlit · Pydantic.

> Example conversation:
> 1. *"Find HIGH segment customers with credit > 750, top 5."*
> 2. *"Score them."*
> 3. *"Generate a personalized WhatsApp message for the top one."*
> 4. *"Send it."*
> 5. *"What did we do so far?"*
>
> The system shows the **reasoning trace** for every turn — exactly which agent/tool was invoked, with arguments and brief results.

---

## Table of Contents

1. [Architecture Diagram](#1-architecture-diagram)
2. [Execution Flow](#2-execution-flow)
3. [Tool Design and Usage](#3-tool-design-and-usage)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Trade-offs and Limitations](#5-trade-offs-and-limitations)
6. [Setup and Run Instructions](#6-setup-and-run-instructions)
7. [Demo Scenarios](#7-demo-scenarios)
8. [Project Layout](#8-project-layout)
9. [Future Improvements](#9-future-improvements)

---

## 1. Architecture Diagram

### 1.1 High-level system architecture

```
                  ┌──────────────────────────────────────────────────┐
                  │                Streamlit Frontend                  │
                  │   Chat UI · Tool trace · Customers · Logs page     │
                  └────────────────────────┬─────────────────────────┘
                                           │  HTTP (JSON)
                                           ▼
                  ┌──────────────────────────────────────────────────┐
                  │                  FastAPI Backend                   │
                  │   /chat · /chat/reset · /workflow/* · /customers   │
                  │   /messages · /campaigns · /logs · /tools          │
                  └────────────────────────┬─────────────────────────┘
                                           │
                                           ▼
                  ┌──────────────────────────────────────────────────┐
                  │           Conversational Orchestrator              │
                  │  - Azure OpenAI tool calling                       │
                  │  - Per-conversation `ConversationSession`           │
                  │  - 6-hop tool loop (plan → call → loop → reply)    │
                  └────────────────────────┬─────────────────────────┘
                                           │ selects & sequences tools
                                           ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                  Agents-as-Tools (11 tools)                    │
        │  list_customers · get_customer · discover_loan_candidates       │
        │  score_customers · recommend_product · generate_outreach        │
        │  send_whatsapp · send_all_pending · run_full_workflow           │
        │  explain_customer · get_session_summary                         │
        └──────────────────────────────────────────────────────────────┘
                                           │
        ┌──────────┬─────────────┬─────────┴─────────┬────────────┬─────────────┐
        ▼          ▼             ▼                   ▼            ▼             ▼
   Discovery   Scoring     Recommendation         Outreach     Campaign     run_full_workflow
     Agent      Agent          Agent                Agent        Agent      (LangGraph pipeline)
        │          │             │                   │            │             │
        └──────────┴─────────────┼───────────────────┴────────────┴─────────────┘
                                  ▼
                  ┌──────────────────────────────────────────────────┐
                  │          MCP Server (in-process, RBAC)             │
                  │   ToolRegistry · RoleManager · permissions.py      │
                  │   Audit hook → agent_logs table                    │
                  └────────────────────────┬─────────────────────────┘
                                ┌──────────┴───────────┐
                                ▼                      ▼
                       SQLite (banking_crm.db)   Azure OpenAI (gpt-5-nano)
```

### 1.2 Layered view

| Layer | Module(s) | Responsibility |
|---|---|---|
| **UI** | `frontend/app.py`, `frontend/components/workflow_chat.py`, `orchestration_viz.py` | Streamlit chat + reasoning trace + animated multi-agent visualization. |
| **API** | `backend/api/routes.py`, `backend/api/server.py` | FastAPI surface (`/chat`, `/workflow/*`, `/customers`, `/logs`, `/tools`). |
| **Conversation brain** | `backend/agents/conversational_orchestrator.py` | LLM tool-calling loop, history management, audit. |
| **Tool layer** | `backend/agents/orchestrator_tools.py` | 11 tools + JSON Schemas + `ConversationSession` working memory. |
| **Specialist agents** | `backend/agents/*_agent.py` | Discovery, Scoring, Recommendation, Outreach, Campaign, Supervisor. |
| **Workflow engine** | `backend/workflows/{graph,state,execution_manager}.py` | LangGraph compiled pipeline + step rerun + checkpoints. |
| **MCP governance** | `backend/mcp/{server,tool_registry,role_manager,permissions}.py` | RBAC + tool registry + audit envelopes. |
| **Tools (impl.)** | `backend/tools/*.py` | DB-backed business logic (customer, txn, scoring, reco, whatsapp, audit). |
| **Data** | `backend/database/{models,db,schema.sql,seed_data}.py` | SQLAlchemy ORM, schema, Faker-based seeding. |
| **Services** | `backend/services/{llm_service,logging_service}.py` | Azure OpenAI client (`chat_with_tools`, circuit breaker), structured logging. |

---

## 2. Execution Flow

### 2.1 Per-turn flow (conversation loop)

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  RM types in Streamlit chat                                        │
 │  "Find HIGH segment customers with credit > 750, top 5"            │
 └──────────────────────────────┬─────────────────────────────────────┘
                                ▼
                       POST /chat { message, conv_id, rm_name, settings }
                                ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  ConversationalOrchestrator.chat()                                 │
 │                                                                    │
 │  1.  session = sessions.get_or_create(conv_id, rm_name, settings) │
 │  2.  messages = [system_prompt, *last_8(history), {user, ...}]    │
 │  3.  loop hop = 1..MAX_TOOL_HOPS:                                  │
 │        resp = llm_service.chat_with_tools(messages, TOOL_SCHEMAS) │
 │        if resp.has_tool_calls:                                     │
 │           for tc in resp.tool_calls:                               │
 │              result  = TOOL_HANDLERS[tc.name](session, **args)    │
 │              audit_log(role="orchestrator", tool=tc.name, ...)    │
 │              messages.append({role:"tool", name, content:result}) │
 │           continue                                                 │
 │        else: final_reply = resp.content;  break                    │
 │  4.  return { conv_id, reply, tool_trace, session_summary }       │
 └──────────────────────────────┬─────────────────────────────────────┘
                                ▼
        Streamlit renders reply + collapsible reasoning trace + summary
```

Per-turn timings are recorded; each tool invocation is also persisted in `agent_logs` for full replay.

### 2.2 Inside the LangGraph "full workflow" tool

When the RM says *"run a full personal loan campaign"*, the orchestrator calls `run_full_workflow`, which executes the deterministic LangGraph pipeline:

```
START
  → supervisor_begin          (init state, run_id, log)
  → customer_discovery        (CustomerDiscoveryAgent)
  → customer_scoring          (ScoringAgent — deterministic)
  → product_recommendation    (RecommendationAgent — catalog gates)
  → outreach_generation       (OutreachAgent — LLM + fallback)
  → campaign_execution        (CampaignAgent — simulated send)
  → supervisor_end            (summarize, persist)
END
```

The compiled graph lives in `backend/workflows/graph.py`. `ExecutionManager` (in `execution_manager.py`) wraps it to:
- inject the `MCPServer` and `WorkflowState`,
- assign a unique `run_id`,
- persist a snapshot per node so individual steps can be re-run (`POST /workflow/{run_id}/step`).

### 2.3 Inside one specialist agent (example: ScoringAgent)

```
ScoringAgent.run(candidates):
  for c in candidates:
      profile = invoke_tool("fetch_customer_profile", customer_id=c.id)   ── via MCP
      spend   = invoke_tool("analyze_monthly_spending", customer_id=c.id) ── via MCP
      balance = invoke_tool("analyze_balance_patterns", customer_id=c.id) ── via MCP
      txnfreq = invoke_tool("analyze_transaction_frequency", customer_id=c.id) ── via MCP
      score   = invoke_tool("compute_conversion_score", features={...})   ── via MCP
      append result + log reasoning
```

Every `invoke_tool` call goes through:

```
BaseAgent.invoke_tool(...)
    └─► MCPServer.invoke(role, tool, **kwargs)
            ├─► RoleManager.assert_can_invoke(role, tool)        # RBAC
            ├─► ToolRegistry.get(tool)                            # discoverability
            ├─► time.perf_counter()                              # latency
            ├─► tool.callable(**kwargs)                          # business logic
            └─► audit_tools.log_agent_event(...)                 # agent_logs row
```

This produces a structured envelope: `{ ok, tool, result, duration_ms }` or `{ ok:false, error, code }`.

### 2.4 Conversation memory model

```
ConversationSession (per conv_id, in-process)
├── conv_id, rm_name, created_at
├── working sets (lists of dicts):
│     ├── discovered      ← last discovery result
│     ├── scored          ← last scoring output, ranked desc
│     ├── recommendations
│     ├── messages        ← drafted outreach
│     └── campaigns       ← dispatched
├── caches (id → dict):
│     ├── profiles_cache
│     ├── scores_by_id
│     ├── recs_by_id
│     └── msgs_by_id
├── last_run_id (links to LangGraph runs)
└── settings: loan_type, min_conversion_threshold, top_n
```

This is what lets the orchestrator resolve *"the top one"*, *"rank 2"*, *"C00042"*, or *"Faris"* deterministically — without the LLM having to remember exact ids from prior turns.

---

## 3. Tool Design and Usage

### 3.1 Tool taxonomy

The system has **three concentric layers** of tools. Each is governed by RBAC and audited:

```
┌───────────────────────────────────────────────────────────────┐
│  Orchestrator tools (11)         ← exposed to the LLM           │
│  agents-as-tools wrappers in `backend/agents/orchestrator_tools.py`
├───────────────────────────────────────────────────────────────┤
│  Specialist agents (6)            ← called by orchestrator tools
│  Discovery · Scoring · Recommendation · Outreach · Campaign · Supervisor
├───────────────────────────────────────────────────────────────┤
│  MCP tools (11)                   ← called by specialist agents
│  fetch_*, analyze_*, compute_*, recommend_*, send_*, log_*
└───────────────────────────────────────────────────────────────┘
```

### 3.2 Orchestrator tools (LLM function-calling surface)

| # | Tool | Wraps | Use case |
|---|---|---|---|
| 1 | `list_customers` | DB query (master) | *"Show me HIGH segment customers with credit > 750."* — seeds working set. |
| 2 | `get_customer` | `customer_tools` | *"Show me C00042"* — full 360° profile (txns, inquiries, CRM). |
| 3 | `discover_loan_candidates` | `CustomerDiscoveryAgent` | *"Find personal-loan candidates."* |
| 4 | `score_customers` | `ScoringAgent` | *"Score them"* — runs deterministic scoring over the working set or a subset. |
| 5 | `recommend_product` | `RecommendationAgent` | *"Recommend a product for them."* |
| 6 | `generate_outreach` | `OutreachAgent` | *"Draft a friendly WhatsApp."* (auto-recommends if needed) |
| 7 | `send_whatsapp` | `CampaignAgent` | *"Send it."* (re-uses the stored draft when `message` is omitted) |
| 8 | `send_all_pending` | `CampaignAgent` | *"Send all pending."* (batch dispatch of drafted-but-unsent) |
| 9 | `run_full_workflow` | `ExecutionManager` (LangGraph) | *"Run the whole campaign for top 5."* |
| 10 | `explain_customer` | session data | *"Why was rank 3 picked?"* — score features + reco rationale verbatim. |
| 11 | `get_session_summary` | session data | *"What did we do so far?"* — snapshot of working memory. |

Each tool follows the **same contract**:

```python
def some_tool(session: ConversationSession, *, ...) -> Dict[str, Any]:
    # 1. resolve refs ("top", "rank N", "C00042", numeric id) → real customer_id
    # 2. call specialist agent / MCP tool / direct DB lookup
    # 3. update session working memory + caches
    # 4. return a JSON-clean payload (no ORM objects, no SQLAlchemy)
```

And ships with two registrations:

- `TOOL_HANDLERS["name"] = some_tool` — wrapped with a timing decorator.
- A matching JSON Schema in `TOOL_SCHEMAS` — consumed by Azure OpenAI tool calling.

Adding a new capability (e.g. *email outreach*) is therefore a single-file change.

### 3.3 Specialist agents

| Agent | RBAC role | Single responsibility |
|---|---|---|
| `CustomerDiscoveryAgent` | `discovery_agent` | Surface candidate customers (income + credit + recent loan inquiries). |
| `ScoringAgent` | `scoring_agent` | Deterministic conversion-probability scoring + explainable features. |
| `RecommendationAgent` | `recommendation_agent` | Best-fit product from catalog (income / credit / segment gates). |
| `OutreachAgent` | `outreach_agent` | Personalized WhatsApp copy (LLM with deterministic fallback). |
| `CampaignAgent` | `campaign_agent` | Simulate WhatsApp dispatch + persist `whatsapp_campaigns` row. |
| `SupervisorAgent` | `supervisor_agent` | Lifecycle for the legacy full-pipeline (LangGraph) execution. |

All agents inherit from `BaseAgent`, which provides:

- `self.invoke_tool(name, **kwargs)` — single chokepoint through MCP.
- `self.log(reasoning, ...)` — structured row in `agent_logs`.
- `self.agent_name`, `self.role` — used everywhere for RBAC + audit.

### 3.4 MCP tools (back-of-house)

| Tool | Module | Purpose |
|---|---|---|
| `fetch_customer_by_id` | `customer_tools` | Master profile lookup. |
| `fetch_high_income_customers` | `customer_tools` | Income / segment filter. |
| `fetch_recent_loan_inquiries` | `customer_tools` | Loan-intent signal. |
| `fetch_customer_profile` | `customer_tools` | 360° aggregate (txns + inquiries + CRM). |
| `analyze_monthly_spending` | `transaction_tools` | Spend stats over N months. |
| `analyze_balance_patterns` | `transaction_tools` | Avg / volatility / trend on balance. |
| `analyze_transaction_frequency` | `transaction_tools` | Activity intensity. |
| `compute_conversion_score` | `scoring_tools` | Weighted features + logistic calibration → probability. |
| `recommend_product` | `recommendation_tools` | Catalog gates → product. |
| `send_whatsapp_message` | `whatsapp_tools` | Simulated send + DB persist. |
| `log_agent_event` | `audit_tools` | Append a row to `agent_logs`. |

### 3.5 RBAC matrix (excerpt from `backend/mcp/permissions.py`)

```
role                  → tools it may invoke
─────────────────────────────────────────────────────────────────
supervisor_agent      → log_agent_event
discovery_agent       → fetch_high_income_customers,
                        fetch_recent_loan_inquiries,
                        fetch_customer_by_id, log_agent_event
scoring_agent         → fetch_customer_profile, analyze_*, 
                        compute_conversion_score, log_agent_event
recommendation_agent  → fetch_customer_by_id, recommend_product,
                        log_agent_event
outreach_agent        → fetch_customer_by_id, log_agent_event
campaign_agent        → send_whatsapp_message, log_agent_event
```

The orchestrator itself doesn't hit MCP directly; it acts through the specialist agents and respects their role boundaries. A misconfigured tool call returns `{ ok:false, code:"PERMISSION_DENIED" }` — never a silent failure.

### 3.6 Tool-call audit

Every invocation produces a row in `agent_logs`:

```
{
  log_id, run_id, conv_id?, agent_name, role, tool_name,
  input_payload, output_payload, status, duration_ms,
  reasoning, error, created_at
}
```

The Streamlit **Logs** page (and `GET /logs`) renders these grouped by run / conv. The chat UI shows a *summarized* trace per assistant turn (under "Reasoning trace") and links to the full audit on the Logs page.

---

## 4. Key Design Decisions

### 4.1 One conversational brain + specialists as tools

The biggest design decision: **do not expose multiple agents to the user**. Only one — the `ConversationalOrchestrator`. Everything else is a tool it can call.

**Why?**

- A real RM thinks in tasks (*"send a friendlier message to the top one"*), not in pipeline steps.
- LLM tool calling is now a first-class primitive — far more flexible than a hand-rolled router.
- Single user-facing surface = one place to set the policy, the system prompt, the audit, and the safety rails.

The specialist agents stay encapsulated, single-responsibility, and RBAC-governed. The orchestrator is a *planner*, not a worker.

### 4.2 Keep MCP even though one agent drives the show

MCP isn't redundant — it solves a different problem (governance). Even when the orchestrator picks the tool, the specialist agent under the hood still hits MCP, which enforces RBAC, times the call, and writes a structured audit row. That's the layer that earns its keep in production.

### 4.3 Keep the LangGraph pipeline as a tool

The fixed Discovery → Scoring → Recommendation → Outreach → Campaign pipeline is exposed as `run_full_workflow`. Reasons:

- Some RM requests genuinely *are* "run the whole thing".
- LangGraph gives us checkpointed runs, per-node rerun, and a predictable happy path that doesn't depend on the LLM picking the right next step.
- It's an offline-capable fallback if Azure OpenAI is down.

### 4.4 Server-side conversation memory

The `ConversationSession` stores discovered candidates, scores, drafted messages, and dispatch results. The orchestrator can refer to *"the top one"*, *"rank 2"*, *"C00042"* and `_resolve_customer_id` deterministically resolves the reference.

Server-side instead of putting it all into the LLM context because:

- **Token economics.** Twelve scored customers and three drafts blow up every prompt.
- **Determinism.** "Send it" must refer to the *exact* customer's *exact* draft. Letting the LLM remember ids causes hallucinations.
- **Auditability.** Working memory + audit log = a complete, replayable picture.

### 4.5 Deterministic scoring and recommendation

The Scoring agent uses a **weighted linear model with a logistic calibration** over interpretable features (income, credit, recent inquiries, transaction frequency, balance trend). The Recommendation agent uses catalog gates (income / credit / segment).

Why not LLM?

- LLMs are bad at consistent numerical scoring.
- Regulated outputs (eligibility, product fit) must be explainable and stable.
- The orchestrator can still ask *"why?"* via `explain_customer` — which returns the actual features, not LLM speculation.

### 4.6 LLM only where copywriting matters

The only place we use the LLM for content is the **Outreach agent** (WhatsApp message drafting). It has a deterministic templated fallback so the end-to-end pipeline still runs without Azure OpenAI.

### 4.7 Circuit breaker for LLM auth failures

`llm_service.py` has a process-wide circuit breaker. Once we see `AuthenticationError`, we stop retrying for the process lifetime and fall back deterministically. This took the end-to-end pipeline from ~35 s of futile retries to ~1.7 s when the LLM is misconfigured.

### 4.8 Tool-hop cap

Each chat turn is capped at `MAX_TOOL_HOPS = 6` to prevent runaway plans. Raise it if you need deeper turns; in practice 6 hops cover *find → score → recommend → outreach → send → summarize*.

### 4.9 In-process MCP

The MCP server runs in-process for simplicity (one Python process). The interface (`MCPServer.invoke(role, tool, **kwargs)`) is identical to what a remote MCP would expose, so swapping to cross-process governance is a single import change.

### 4.10 SQLite for the demo

Zero-ops local persistence. Schema is portable. `DATABASE_URL` (in `backend/.env`) is the only thing to change for Postgres.

---

## 5. Trade-offs and Limitations

| Decision | Pro | Con / Mitigation |
|---|---|---|
| LLM-driven decomposition | Natural multi-turn UX, no hand-rolled router | Needs Azure OpenAI for chat. **Mitigation:** circuit breaker + fixed-pipeline tool + REST endpoints stay available. |
| In-process `ConversationSession` | Deterministic refs, cheap tokens, fast | Lost on restart. **Mitigation:** swap the `_SessionRegistry` for Redis / Postgres. |
| 6-hop tool loop cap | Prevents runaway plans | A complex turn may need more. **Mitigation:** bump `MAX_TOOL_HOPS` in code. |
| In-process MCP | Simple, one process, easy demo | Not cross-process governance. **Mitigation:** same interface works for remote MCP. |
| Deterministic scoring | Explainable, stable, regulator-friendly | Needs human tuning if product policy changes. **Mitigation:** weights live in one tool file. |
| Simulated WhatsApp | No external dependency | Not a real provider. **Mitigation:** swap `send_whatsapp_message` for Twilio / Gupshup / Meta Cloud API. |
| SQLite | Zero-ops local dev | Single-writer. **Mitigation:** switch `DATABASE_URL` to Postgres + add Alembic migrations. |
| Synchronous workflow | Predictable, debuggable | Long campaigns block the request. **Mitigation:** wrap LangGraph runs in Celery / RQ for prod. |
| LLM-generated WhatsApp copy | Natural, personalized | Variable quality, possible drift. **Mitigation:** deterministic template fallback; per-RM tone params; A/B logging. |
| Streamlit UI | Fast to build, easy to demo | Not a production UX. **Mitigation:** API is React-ready; rebuild front when needed. |

### Known limitations

- **No streaming of tool execution** — the reply renders once the full hop loop finishes. SSE / WebSocket streaming is a natural next step.
- **No per-RM identity** in the audit log yet (the `rm_name` is recorded on the session, but not on every row).
- **No vector store** of past CRM interactions; outreach tone is grounded only in the current customer profile and product catalog.
- **No automated tests yet** — the architecture is test-friendly (DI via `session_scope`, injectable `MCPServer`, session registry), but I haven't added a suite.

---

## 6. Setup and Run Instructions

### 6.1 Prerequisites

- **Python 3.10+** on macOS / Linux / Windows.
- An **Azure OpenAI** deployment with tool-calling support (e.g. `gpt-5-nano`, `gpt-4o`, `gpt-4-turbo`). The credentials in `backend/.env` are pre-filled for the assessment review.
- ~80 MB free disk for the seeded SQLite database.

### 6.2 Install

```bash
# 1. Clone & enter
cd /path/to/assessment

# 2. Create + activate the virtual environment
python -m venv venv

# Linux / macOS
source venv/bin/activate
# Windows
# venv\Scripts\activate

# 3. Install dependencies
pip install -r backend/requirements.txt
```

`backend/requirements.txt` pins: `fastapi`, `uvicorn`, `langgraph`, `openai`, `sqlalchemy`, `pydantic`, `python-dotenv`, `streamlit`, `pandas`, `faker`, `requests`, `httpx`, `plotly`, `typing-extensions`.

### 6.3 Configure

`backend/.env` (already populated for the assessment):

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://apps-mal2lne1-eastus2.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5-nano-5
AZURE_OPENAI_MODEL=gpt-5-nano
AZURE_OPENAI_API_VERSION=2024-10-21

DATABASE_URL=sqlite:///banking_crm.db
```

### 6.4 Seed the database (once)

```bash
python -m backend.main --seed --no-serve
```

Creates `banking_crm.db` with **120 customers**, ~**18 000 transactions**, **loan inquiries**, **CRM interactions**, weighted across HIGH / MEDIUM / LOW segments. Re-running is idempotent (`--reset-seed` to nuke + reseed).

### 6.5 Run the backend

```bash
python -m backend.main --reload
# equivalent: uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000
```

- API docs: <http://localhost:8000/docs>
- Health: `curl http://localhost:8000/health`

### 6.6 Run the frontend (second terminal)

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

- UI: <http://localhost:8501>
- Pages: **Workflow chat** (the main conversational UI), **Customers**, **Logs**.

### 6.7 Useful API calls

```bash
# Chat (multi-turn — re-use conv_id from the previous response to keep memory)
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

# Run the fixed LangGraph pipeline directly (also reachable via the chat as a tool)
curl -X POST http://localhost:8000/workflow/run \
  -H 'Content-Type: application/json' \
  -d '{"user_query":"full personal loan campaign","top_n_customers":5,"loan_type":"PERSONAL"}'

# Re-run a single workflow step
curl -X POST http://localhost:8000/workflow/<run_id>/step \
  -H 'Content-Type: application/json' \
  -d '{"step":"outreach_generation"}'

# RBAC introspection
curl http://localhost:8000/tools             # all tools + schemas
curl http://localhost:8000/tools/scoring_agent

# Audit logs
curl 'http://localhost:8000/logs?limit=50'

# Customer data
curl 'http://localhost:8000/customers?segment=HIGH&min_credit=750&limit=20'
curl http://localhost:8000/customers/42
```

### 6.8 Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Send a message in a conversation (creates `conv_id` on first call). |
| `POST` | `/chat/reset` | Clear server-side session memory for a conversation. |
| `POST` | `/workflow/run` | Run the LangGraph full pipeline directly. |
| `POST` | `/workflow/{run_id}/step` | Re-run a single node of a previous run. |
| `GET` | `/workflow/runs` | List recent runs. |
| `GET` | `/workflow/{run_id}` | Fetch a full run record. |
| `GET` | `/customers` | List / filter customers. |
| `GET` | `/customers/{id}` | 360° profile. |
| `GET` | `/messages/{run_id}` | Drafted outreach for a run. |
| `GET` | `/campaigns/{run_id}` | Dispatched campaigns for a run. |
| `GET` | `/logs` | Audit log (paginated). |
| `GET` | `/tools` | Full tool registry. |
| `GET` | `/tools/{role}` | Tools visible to a given RBAC role. |
| `GET` | `/health` | Liveness. |

---

## 7. Demo Scenarios

Run these as **multi-turn chats** in the UI (same `conv_id` is reused, so each step builds on the previous one). The "Reasoning trace" expander shows the exact tools invoked per turn.

### Scenario A — Manual specialist orchestration

> **Turn 1.** *"Find HIGH segment customers with credit > 750, top 5."*  
> **Turn 2.** *"Score them."*  
> **Turn 3.** *"Generate a personalized WhatsApp message for the top one."*  
> **Turn 4.** *"Send it."*  
> **Turn 5.** *"What did we do so far?"*

Expected trace:
1. `list_customers` → working set seeded.
2. `score_customers` → ranked, top kept in memory.
3. `generate_outreach(customer_ref="top")` → draft persisted.
4. `send_whatsapp(customer_ref="top")` → `whatsapp_campaigns` row written.
5. `get_session_summary` → counts + last run id.

### Scenario B — Full pipeline shortcut

> **Turn 1.** *"Run a full personal loan campaign for the top 5 customers above 0.6 probability."*  
> **Turn 2.** *"Send all pending."*  
> **Turn 3.** *"Why was the top customer selected?"*

Expected trace:
1. `run_full_workflow(loan_type="PERSONAL", top_n=5, min_probability=0.6)` → LangGraph pipeline runs end-to-end; results captured in session.
2. `send_all_pending` → batch dispatch of all drafted messages.
3. `explain_customer(customer_ref="top")` → score features + recommendation rationale (deterministic).

### Scenario C — Targeted customer enquiry

> **Turn 1.** *"Show me customer C00042."*  
> **Turn 2.** *"Recommend a product for them."*  
> **Turn 3.** *"Draft a friendlier outreach message."*  
> **Turn 4.** *"Send it."*

Expected trace:
1. `get_customer(customer_ref="C00042")` → 360° profile cached.
2. `recommend_product` → catalog gates evaluated.
3. `generate_outreach(tone="friendly")` → personalized copy.
4. `send_whatsapp` → simulated dispatch + DB persist.

---

## 8. Project Layout

```
assessment/
├── backend/
│   ├── agents/
│   │   ├── base.py                          # BaseAgent: invoke_tool + audit
│   │   ├── conversational_orchestrator.py   # the ONLY user-facing agent
│   │   ├── orchestrator_tools.py            # 11 tools + ConversationSession
│   │   ├── supervisor_agent.py              # LangGraph lifecycle owner
│   │   ├── customer_discovery_agent.py
│   │   ├── scoring_agent.py
│   │   ├── recommendation_agent.py
│   │   ├── outreach_agent.py
│   │   └── campaign_agent.py
│   ├── api/
│   │   ├── routes.py                        # FastAPI endpoints (incl. /chat)
│   │   └── server.py                        # FastAPI app factory
│   ├── database/
│   │   ├── db.py                            # SQLAlchemy session, engine
│   │   ├── models.py                        # 7 ORM tables
│   │   ├── schema.sql                       # human-readable schema mirror
│   │   └── seed_data.py                     # Faker-based seeding
│   ├── mcp/
│   │   ├── server.py                        # MCPServer.invoke()
│   │   ├── tool_registry.py                 # ToolSpec + registry
│   │   ├── role_manager.py                  # RBAC enforcement
│   │   └── permissions.py                   # static RBAC map
│   ├── services/
│   │   ├── llm_service.py                   # Azure OpenAI + chat_with_tools + circuit breaker
│   │   └── logging_service.py
│   ├── tools/
│   │   ├── audit_tools.py
│   │   ├── customer_tools.py
│   │   ├── transaction_tools.py
│   │   ├── scoring_tools.py
│   │   ├── recommendation_tools.py
│   │   └── whatsapp_tools.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── constants.py
│   │   └── helpers.py
│   ├── workflows/
│   │   ├── state.py                         # WorkflowState TypedDict
│   │   ├── graph.py                         # LangGraph compiled StateGraph
│   │   └── execution_manager.py             # run + step + checkpoint
│   ├── requirements.txt
│   ├── .env
│   └── main.py                              # `python -m backend.main` entrypoint
├── frontend/
│   ├── app.py                               # Streamlit entry (sys.path fix included)
│   ├── api_client.py                        # thin HTTP client
│   ├── components/
│   │   ├── workflow_chat.py                 # chat UI + reasoning trace
│   │   ├── orchestration_viz.py             # animated multi-agent graphic
│   │   ├── customer_table.py
│   │   └── message_preview.py
│   └── pages/
│       ├── customers.py
│       └── logs.py
├── README.md                                # ← THIS FILE
├── README_BASE_LEVEL.md                     # alt walkthrough (architecture-first)
└── .gitignore
```

---

## 9. Future Improvements

- **Streaming tool execution.** SSE / WebSocket to render each tool result as it executes. Today the reply renders once the full hop loop finishes.
- **Plan-then-execute mode.** Make the orchestrator emit a planned tool sequence first, then ask for RM approval before executing. Useful for high-stakes actions.
- **Cross-process MCP.** Replace the in-process server with a true remote MCP for multi-tenant governance.
- **Postgres + Alembic migrations.**
- **Async parallel scoring.** Score 50 customers concurrently instead of sequentially.
- **Per-RM identity** propagated into every audit row.
- **Vector store of past CRM interactions** to ground outreach tone.
- **More channels:** email, SMS, app push — each a new tool.
- **Unit + integration test suite.** Architecture is already test-friendly (`session_scope`, injectable `MCPServer`, session registry).
- **Per-tool latency dashboards** in the Logs page.

---

### Author's note

The codebase intentionally over-invests in **structure and explainability**. The goal is to demonstrate **agentic reasoning + tool governance + dynamic decomposition** — not to maximize LLM calls. The LLM is the *planner*, not the worker; all numerical and policy decisions live in deterministic tools and are reproducible from the audit log alone.
# conversation-based-Agentic-AI-for-Banking-CRM
