# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The **Charm** framework (**C**ustom **H**arness **A**nd **R**egistry **M**anagement) is a multi-agent system implementing Google's Agent-to-Agent (A2A) protocol with a centralized registry for agent discovery and coordination. This is a proof-of-concept / demo platform with a three-panel web dashboard. Agents use Google ADK (`google-adk` 1.28+) with `to_a2a()` for automatic A2A protocol handling, LiteLLM for multi-provider LLM support.

## Architecture

**Registry Agent** (`svr_registry_agent.py`, port 7060) — Central registry storing agent cards in-memory (loaded from `registry.json`). Provides LLM-powered semantic search. API-key-authenticated CRUD. **Stays FastAPI** (its `/tasks/send` dispatches by `tool_name`, non-standard A2A).

**Chat Agent** (`svr_chat_agent.py`, port 7002) — Orchestrator. Receives user requests via A2A JSON-RPC, queries registry, delegates via `call_remote_agent` tool. Uses `to_a2a()`.

**Email Agent** (`svr_email_agent.py`, port 7001) — Sends emails directly via SMTP (`smtplib`). Uses `to_a2a()`.

**SMS Agent** (`svr_sms_agent.py`, port 7003) — Sends SMS via TextBelt API. Uses `to_a2a()`.

**Web Agent** (`svr_web_agent.py`, port 7004) — Live web research via the `browser-use` library. Its `browse_web(task, force_cloud)` tool drives a **local headless Chromium** first and **auto-falls back to browser-use Cloud** (stealth browser, residential proxies, CAPTCHA solving) when a site is bot-protected; callers can force the cloud path. Uses `to_a2a()`. **Dependency isolation:** `browser-use[core]` hard-pins versions that conflict with (and remove) `google-adk`/`a2a-sdk`, so it **cannot** live in the main venv. Instead it runs in a separate `.venv-web` (created by `./setup_web_venv.sh`), and `browse_web` invokes `web_browser_runner.py` there as a subprocess, parsing a sentinel-prefixed JSON result line. The agent module itself uses only lazy/subprocess access, so it imports fine in the main venv. Set `BROWSER_USE_API_KEY` in `.env` for the cloud fallback (local works without it).

**Web Dashboard** (`webagent/webagent_server.py`, port 9091) — Three-panel FastAPI web UI:
- Left: Agent roster with toggle switches (start/stop agents)
- Middle: Chat interface
- Right: Real-time conversation trace (SSE stream from all agent processes)

**Agent Logger** (`agent_logger.py`) — Two-level conversation tracing (`standard`/`conversation`) with event emitter pattern. Each agent process POSTs trace events to `TRACE_COLLECTOR_URL` (the dashboard), which streams them to the browser via SSE.

**Flow:** Browser → Dashboard `/api/chat` → Chat Agent (A2A JSON-RPC) → Registry Agent (REST) → Chat Agent → Specialist Agent (A2A JSON-RPC) → response. Trace events flow from each agent process → Dashboard `/api/trace/collect` → Browser SSE.

### A2A Protocol

Agents using `to_a2a()` serve JSON-RPC at their root URL (`POST /`). Agent cards at `GET /.well-known/agent.json` (mutual TLS, so a CHARM client cert is required). The Registry Agent uses old REST protocol (`POST /tasks/send` with `tool_name` field).

## Current work

CHARM 3.0 is built but not yet run end to end. Read `HANDOFF-3.0.md` first (local only, not in git): it has the context,
what changed, what's verified, the acceptance checks, and where things are likely to break.

## CHARM 3.0 guard (read before changing any agent)

`charm_guard.py` is the shared defense layer every process imports; `charm_policy.json` holds its rules.
Fix security behavior there, not in individual agents.

- Each agent module calls `guard.init_identity("<Name>")` at import, wraps its app with
  `app.add_middleware(guard.GuardMiddleware, agent_name="<Name>")`, and builds its callbacks with
  `guard.make_guard_callbacks(...)`. Keep all three when adding an agent, and add it to the policy with its pinned port.
- All outbound calls go through `guard.signed_post_json` / `guard.signed_post_json_sync`. Never post
  unsigned, and sign the exact bytes you send.
- Agent-to-agent traffic is mutual TLS. Servers start with `guard.tls_server_args(name)`; clients use
  `guard.tls_context()` (httpx) or `guard.requests_tls()` (requests). Certs are in `keys/tls/`. TLS admits
  CHARM cert holders, and signatures still identify the principal. Only the browser-to-dashboard
  connection on `WEB_PORT` is plain HTTP, and agents send traces to the TLS port `DASHBOARD_TRACE_PORT`.
- `call_remote_agent(agent_id, query)` takes a registry ID. Never let the model supply a URL or card.
- Side effects (email, SMS) are gated by `guard.rule_capability(...)` in the specialist and by the
  approval broker in the orchestrator. Don't add a send path that bypasses either.
- Bind to 127.0.0.1 only. Keys live in `keys/` (gitignored).
- Tests: `python -m pytest tests -q`.

## Commands

```bash
# Setup
cp env.example .env          # Edit with API keys
pip install -r requirements.txt
./setup_web_venv.sh          # One-time: isolated .venv-web for the Web Agent's browser-use + Chromium

# Start everything (dashboard first, then agents)
./startup.sh

# Stop everything
./shutdown.sh

# Individual agents
uvicorn svr_registry_agent:app --host 127.0.0.1 --port 7060 $(python charm_guard.py tls-args RegistryAgent)
uvicorn svr_chat_agent:app --host 127.0.0.1 --port 7002 $(python charm_guard.py tls-args ChatAgent)
uvicorn svr_sms_agent:app --host 127.0.0.1 --port 7003 $(python charm_guard.py tls-args SMSAgent)
uvicorn svr_web_agent:app --host 127.0.0.1 --port 7004 $(python charm_guard.py tls-args WebAgent)
cd webagent && uvicorn webagent_server:app --host 127.0.0.1 --port 9091

# CLI chat client (alternative to web UI)
python chat_client.py

# Read-only check of the agent mailbox for bounces (operator tool, never give it to an agent)
python check_agent_mailbox.py [days]

# Attack the running EmailAgent as a compromised orchestrator (targets are @example.com)
python demo/attack_compromised_orchestrator.py

# Before/after: unauthenticated SMS request. "before" runs 2.1 on loopback and sends a REAL text (stop 3.0 first)
demo/sms_unauthenticated.sh before
demo/sms_unauthenticated.sh after
```

## Configuration

- `.env` — API keys, `DEFAULT_MODEL`, `AGENT_LOG_LEVEL` (standard/conversation), `TRACE_COLLECTOR_URL`, SMS config
- `svr_*_config.json` — Per-agent registry URL, agent UUID
- `registry.json` — Pre-populated registry state

## Key Dependencies

- `google-adk` 1.28+ — `to_a2a()`, `LlmAgent`, `Runner`
- `a2a-sdk` 0.3+ — A2A types (`AgentCard`, `AgentSkill`)
- `mcp` 1.27+ — Model Context Protocol
- `litellm` >=1.83.0 — Multi-LLM (**never 1.82.7-1.82.8**)
- `fastapi` 0.135+ — Web framework

## Development Notes

- Python 3.12, `to_a2a()` returns Starlette (not FastAPI)
- `agent_logger.py` provides `make_callbacks()` → spread into `LlmAgent(**callbacks)`
- `AGENT_LOG_LEVEL=conversation` for demo emoji narrative
- `TRACE_COLLECTOR_URL` enables cross-process trace collection for the dashboard
- Dashboard manages agent processes via subprocess and can start/stop them via API
- The chat agent's `call_remote_agent` uses A2A JSON-RPC to call `to_a2a()` agents
