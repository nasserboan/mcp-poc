# MCP PoC — Design Spec

**Date:** 2026-04-07  
**Goal:** Demonstrate to a non-technical audience how tool usage behavior changes across three architectures: LangChain chain, LangGraph agent, and MCP server+client.

---

## Overview

Three self-contained scenarios, each in its own folder, sharing a common RAG corpus (arXiv articles) and LLM setup. MLflow traces every run so tool invocations are visible. LLM: OpenAI `gpt-4o-mini`.

---

## Folder Structure

```
mcp-poc/
├── .env                        # API keys (gitignored)
├── .env.example                # Template with all required vars
├── README.md                   # Differences between scenarios + how to run each
├── pyproject.toml
├── shared/
│   ├── llm.py                  # Returns ChatOpenAI(model="gpt-4o-mini")
│   ├── rag.py                  # ChromaDB init, retriever, rag_tool
│   └── ingest_arxiv.py         # Download ~5 arXiv abstracts, ingest into ChromaDB
├── scenario_1_chain/
│   └── main.py                 # LangChain chain — RAG always called
├── scenario_2_agent/
│   └── main.py                 # LangGraph ReAct agent — LLM decides when to use RAG
└── scenario_3_mcp/
    ├── server.py               # FastMCP server exposing 3 tools
    └── client.py               # LangGraph ReAct agent connecting to MCP server
```

---

## Shared Components

### `shared/llm.py`

Exports `get_llm()`. Returns `ChatOpenAI(model="gpt-4o-mini")`. Single place to change the model if needed.

### `shared/rag.py`

- Initializes ChromaDB with a persistent local directory (`./chroma_db`)
- Exposes `get_retriever()` and `get_rag_tool()` (a LangChain `Tool` wrapping the retriever)
- Shared by all three scenarios

### `shared/ingest_arxiv.py`

- Calls the arXiv public API (no key required) to fetch ~5 paper abstracts on the topic "LLM agents"
- Splits and embeds using `OpenAIEmbeddings`
- Persists into ChromaDB
- Run once before any scenario: `uv run python shared/ingest_arxiv.py`

---

## Scenario 1 — LangChain Chain

**File:** `scenario_1_chain/main.py`

**Behavior:** RAG is always called. No decision-making. The chain is linear:

```
user input → rag_retriever → prompt (with context) → LLM → output
```

**Key point for the demo:** The model always receives retrieved context, whether it needs it or not. Tool usage is unconditional.

**MLflow:** `mlflow.langchain.autolog()` before the chain invocation.

---

## Scenario 2 — LangGraph Agent

**File:** `scenario_2_agent/main.py`

**Behavior:** A ReAct agent. The LLM decides whether to call the RAG tool.

**Graph:**

```
START → agent → (tool_calls?) → tools → agent → ...
                     ↓ no
                    END
```

- `agent` node: LLM with RAG tool bound via `bind_tools`
- `tools` node: `ToolNode` executing the tool
- Conditional edge: `tools_condition` from `langgraph.prebuilt`

**Key point for the demo:** Ask a factual question → tool is called. Ask something general → tool is skipped. The trace in MLflow shows the difference.

**MLflow:** `mlflow.langchain.autolog()` before graph invocation.

---

## Scenario 3 — MCP Server + Client

### `scenario_3_mcp/server.py`

FastMCP server exposing 3 tools:

1. **`rag_search(query: str)`** — retrieves from ChromaDB (same corpus as scenarios 1 & 2)
2. **`get_exchange_rate(base: str, target: str)`** — fetches live rate from `open.er-api.com` (free, no key)
3. **`calculate(expression: str)`** — evaluates a safe math expression using Python's `ast` module

Run with: `uv run python scenario_3_mcp/server.py`

### `scenario_3_mcp/client.py`

LangGraph ReAct agent (same graph as scenario 2) that:

1. Connects to the MCP server via `fastmcp` client
2. Loads available tools dynamically (no hardcoded tool list)
3. Wraps MCP tools as LangChain tools using `langchain-mcp-adapters`
4. Binds tools to LLM and runs the ReAct loop

**Key point for the demo:** The client discovers tools at runtime via the MCP protocol. The graph structure is identical to scenario 2 — the difference is tool origin (local vs. remote MCP server). The LLM now has 3 tools and chooses among them.

**MLflow:** `mlflow.langchain.autolog()` before graph invocation.

---

## MLflow Tracing

- Every scenario calls `mlflow.langchain.autolog()` at startup
- All traces land in a local `./mlruns` directory
- Start the UI with: `mlflow ui` → open `http://localhost:5000`
- Traces show: which nodes ran, which tools were called, LLM input/output, latency

---

## Environment Variables (`.env.example`)

```env
OPENAI_API_KEY=sk-...
```

---

## What the README Will Cover

1. The core difference between the 3 scenarios (one paragraph each, non-technical language)
2. Prerequisites (uv, mlflow, API key)
3. Setup steps (install deps, create `.env`, ingest arXiv data)
4. How to run each scenario
5. How to view traces in MLflow
