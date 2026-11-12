# CHARM 3.0 — Agent Registry System with a Guard

**CHARM** (**C**ustom **H**arness **A**nd **R**egistry **M**anagement) is a multi-agent system built on Google's Agent-to-Agent (A2A) protocol, with a central registry for agent discovery and a three-panel web dashboard for chatting with the agents and watching them work.

Version 3.0 is the "after" half of a before/after demo. CHARM 2.1 (in `../charm-v2.1`) is a working system whose trust boundaries were enforced by prompt text. 3.0 keeps every agent a reasoning specialist and adds a shared guard (`charm_guard.py`) that decides what actually happens: signed calls, mutual TLS, capability tokens for side effects, and human approval when a session has read untrusted content.

> This is a proof of concept and demo platform, not a production system.

## Table of Contents

- [Overview](#overview)
- [Security model](#security-model)
- [Architecture](#architecture)
- [Dashboard](#dashboard)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running 2.1 and 3.0 side by side](#running-21-and-30-side-by-side)
- [Demo scripts](#demo-scripts)
- [API Endpoints](#api-endpoints)
- [Agent Conversation Tracing](#agent-conversation-tracing)
- [LLM Model Switching](#llm-model-switching)
- [Example Workflows](#example-workflows)
- [Troubleshooting](#troubleshooting)
- [Proof of Concept Limits](#proof-of-concept-limits)
- [License](#license)

## Overview

| Component | File | Port | Role |
|---|---|---|---|
| Registry Agent | `svr_registry_agent.py` | 7060 | Stores agent cards and finds agents with LLM-scored semantic search |
| Chat Agent | `svr_chat_agent.py` | 7002 | Orchestrator: finds agents in the registry and delegates to them |
| Email Agent | `svr_email_agent.py` | 7001 | Sends email over SMTP |
| SMS Agent | `svr_sms_agent.py` | 7003 | Sends texts through the TextBelt API |
| Web Agent | `svr_web_agent.py` | 7004 | Reads web pages and does live web research |
| Dashboard | `webagent/webagent_server.py` | 9091 (UI), 9092 (trace, TLS) | Three-panel web UI, process control, approvals |
| CLI client | `chat_client.py` | — | Command-line chat with the Chat Agent |
| Guard | `charm_guard.py` + `charm_policy.json` | — | Shared security layer every process imports |

Agents register at startup, find each other through the registry, and coordinate to fulfill requests. They use Google ADK (`google-adk` 1.28+) with `to_a2a()` for the A2A protocol and LiteLLM for multiple model providers (Anthropic, OpenAI).

## Security model

The model can be talked into asking for anything; the guard decides what actually happens. Rules live in `charm_policy.json` (no secrets, re-read on every check, so edits apply without a restart).

| Layer | What it does |
|---|---|
| Identity | Every principal (each agent, the dashboard, the CLI) has an Ed25519 key in `keys/`. `startup.sh` creates any that are missing (`python charm_guard.py init-keys`). |
| Mutual TLS | Every agent-to-agent connection is TLS 1.3 with client certificates from a local CA in `keys/tls/`. A client without a CHARM certificate can't connect. Only the browser-to-dashboard connection on port 9091 is plain HTTP. |
| Signed calls | Every request is signed: caller, intended recipient, body hash, expiry, one-time ID. Each agent accepts only the callers the policy lists for it. Unsigned, altered, replayed or misdirected requests get a 401. TLS proves "a CHARM certificate holder"; the signature proves which agent. |
| Calls by ID | The orchestrator calls agents by registry `agent_id`. Code looks up the address and checks it's an allowed host. (2.1 let the model supply the whole card, URL included.) |
| Registry | No shared API keys. An agent can register or change only its own card. Card text and queries are treated as data in the scoring prompt. |
| Screening | Inbound messages and agent replies are screened for prompt injection (heuristics, plus an optional LLM classifier) and wrapped as untrusted before a model reads them. |
| Taint | Once a conversation has read web content (`browse_web` or `fetch_page`) or anything suspicious, it is marked tainted for the rest of the session. |
| Capabilities | Sending email or SMS needs a capability token naming the exact recipients, valid for two minutes, usable once. The Email and SMS agents check it in their own tool gate, so a recipient swapped by an injection is refused. |
| Approvals | If the session is tainted or a recipient isn't on the allowlist, the orchestrator pauses and the dashboard shows an approval card. Approve makes the dashboard sign the capability; the model never handles it. |
| Allowlists | `charm_policy.json` → `allowlists`. The user profile's own email and phone are always allowed; `@domain.com` allows a whole domain. Only the operator can edit this file. No agent has a tool that can change it. |
| Audit | Trace events are signed, so one agent can't write entries as another. |
| Hygiene | Everything binds to 127.0.0.1. The dashboard requires a per-boot page token on `/api` calls. The browser subprocess gets a scrubbed environment with no SMTP or SMS secrets. `fetch_page` refuses loopback, private and metadata addresses, including after redirects. |

What the guard does not do: contain an agent that escapes its process (that's sandboxing), or make prompt-injection detection perfect (screening is a signal; the gates are the control).

Approval behavior is set by `approval.mode` in the policy:
- `tainted` (default): approval when the session has read external or suspicious content, or a recipient is off the allowlist
- `always`: approval for every send
- `never`: approval only for recipients off the allowlist

## Architecture

### How it works

1. The user sends a request from the dashboard or CLI.
2. The Chat Agent decides whether it needs help, and if so asks the Registry Agent for capable agents.
3. The Registry Agent scores registered agents against the query with one LLM call.
4. The Chat Agent calls the best agent by its registry ID, over mutual TLS, with a signed request.
5. For a send (email or SMS), the orchestrator gets a capability token for the exact recipients. If the session is tainted or the recipient isn't allowlisted, that waits for your approval in the dashboard.
6. The specialist checks the signature and the capability, does the work, and replies.
7. The Chat Agent composes the final answer.

Trace events flow from every process to the dashboard's TLS trace port (9092) and stream to the browser over SSE.

### Agent details

**Registry Agent (7060)**, FastAPI
- In-memory registry of agent cards, loaded from `registry.json`
- One LLM call scores all agents against a query
- Builds an "action words" cache per agent at registration, for the dashboard's acknowledgments
- Uses a REST protocol (`POST /tasks/send` with a `tool_name` field), not standard A2A

**Chat Agent (7002)**, `to_a2a()`
- Orchestrates multi-agent work over A2A JSON-RPC
- `call_remote_agent(agent_id, query)` takes a registry ID; the model never supplies a URL or card
- No automatic substitution of a different agent if one is offline
- Remembers the conversation within a session
- Personality from `assistant_profile.json`; knows the user from `user_profile.json` ("email me" uses the profile email)
- Keeps SMS text to 160 characters and puts long content in the email

**Email Agent (7001)**, `to_a2a()`
- Sends over SMTP with STARTTLS (port 587); works with MXRoute or any SMTP provider
- Reports "accepted by the mail server", never "delivered"

**SMS Agent (7003)**, `to_a2a()`
- Sends through the TextBelt API and reports the remaining quota
- Pulls the phone number and message out of an unstructured request

**Web Agent (7004)**, `to_a2a()`
- `fetch_page(url)`: a plain HTTP fetch plus HTML-to-text (`page_fetch.py`). About half a second for a known URL. Used first when the request names a page.
- `browse_web(task, force_cloud)`: drives a local headless Chromium with the `browser-use` library, and falls back to browser-use Cloud (stealth browser, proxies, CAPTCHA solving) when a site blocks it. Used for searches, interactive pages, JavaScript-rendered pages, or when `fetch_page` fails. Typically 60–90 seconds.
- `browser-use` conflicts with `google-adk`/`a2a-sdk`, so it runs in a separate `.venv-web` (created by `./setup_web_venv.sh`). `browse_web` runs `web_browser_runner.py` there as a subprocess.
- Everything it returns is marked as external content, so any later send needs approval.

### A2A protocol

`to_a2a()` agents serve JSON-RPC at their root URL over mutual TLS:

```json
POST https://localhost:7002/
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Send an SMS to 555-1234 saying hello"}],
      "messageId": "msg1"
    }
  }
}
```

Requests must also carry a CHARM signature. Use `guard.signed_post_json` / `guard.signed_post_json_sync`; a bare curl is refused. Agent cards are at `GET /.well-known/agent.json` (a CHARM client certificate is required).

## Dashboard

The dashboard at **http://localhost:9091** (green-tinted; 2.1's is red) has three panels.

**Left: agent roster**
- Running / stopped / registered status, with toggles to start and stop agents
- Per-agent model dropdown and restart button
- Assistant profile (name, personality) and user profile (name, email, phone)

**Middle: chat**
- Chat with the Chat Agent; "Think out loud" shows reasoning as gray bubbles
- Approval cards for sends that need your OK: they name the action and recipients, with Approve and Deny

**Right: conversation trace**
- Live activity from every process, color-coded by agent
- Standard / verbose toggle
- 🛡️ lines show the guard at work: blocked requests, tainted sessions, approvals

## Requirements

- macOS or Linux with bash and curl
- Python 3.12
- An Anthropic and/or OpenAI API key
- SMTP credentials (for email) and a TextBelt key (for SMS)
- Optional: a browser-use Cloud key, for the Web Agent's cloud fallback

```bash
pip install -r requirements.txt     # main venv (.venv)
./setup_web_venv.sh                 # one time: .venv-web with browser-use and Chromium
```

Key dependencies: `google-adk` 1.28+, `a2a-sdk` 0.3+, `litellm` >= 1.83.0 (**never 1.82.7–1.82.8**), `mcp` 1.27+, `fastapi` 0.135+, `PyJWT`, `cryptography`, `httpx`.

## Configuration

### 1. Environment

```bash
cp env.example .env
```

The main settings in `.env`:

```bash
# Model keys (at least one)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...

# Default model for every agent (override per agent from the dashboard)
DEFAULT_MODEL=anthropic/claude-opus-5-5

# Email (SMTP)
SMTP_HOST=mail.mxrouting.net
SMTP_PORT=587
SMTP_USER=agent@yourdomain.com
SMTP_PASS=your_smtp_password
SMTP_FROM=agent@yourdomain.com

# SMS (TextBelt)
SMS_URL=https://textbelt.com/text
SMS_KEY=...

# Dashboard and tracing
WEB_PORT=9091
TRACE_COLLECTOR_URL=https://localhost:9092/api/trace/collect
AGENT_LOG_LEVEL=conversation      # or "standard"
LOG_LEVEL=DEBUG

# Web Agent
BROWSER_USE_API_KEY=              # optional: enables the cloud fallback
WEB_AGENT_HEADLESS=true
WEB_AGENT_TIMEOUT=120
WEB_AGENT_MAX_BROWSES=3
WEB_AGENT_BROWSE_BUDGET=90
```

See `env.example` for the full list, including the chat timeouts.

### 2. Guard policy

`charm_policy.json` sets which callers each agent accepts, which agents produce external content, the email and phone allowlists, and the approval mode. Edits take effect on the next check.

### 3. Agent config files

Each agent has `svr_<name>_config.json` with its registry URL and agent ID. The agent ID is written back at registration.

### 4. Profiles

`assistant_profile.json`, the Chat Agent's name and personality:

```json
{
  "name": "Sam",
  "personality": "You are a helpful assistant with a friendly disposition."
}
```

`user_profile.json`, the user's details (used for "email me" and "text me"; this email and phone are always allowlisted). It is not tracked in git; start from the example:

```bash
cp user_profile.example.json user_profile.json
```


```json
{
  "name": "Your Name",
  "email": "you@example.com",
  "phone": "5555555555",
  "role": "Developer",
  "notes": "Prefers concise responses."
}
```

## Usage

### Start and stop

```bash
./startup.sh      # dashboard first, then Registry, Chat, SMS, Email, Web
./shutdown.sh
```

`startup.sh` creates any missing keys, clears stale processes on 3.0's ports, and waits for each agent to answer before starting the next. Then open **http://localhost:9091**.

### Manual startup

Each server needs its TLS arguments:

```bash
uvicorn svr_registry_agent:app --host 127.0.0.1 --port 7060 $(python charm_guard.py tls-args RegistryAgent)
uvicorn svr_chat_agent:app     --host 127.0.0.1 --port 7002 $(python charm_guard.py tls-args ChatAgent)
uvicorn svr_sms_agent:app      --host 127.0.0.1 --port 7003 $(python charm_guard.py tls-args SMSAgent)
uvicorn svr_email_agent:app    --host 127.0.0.1 --port 7001 $(python charm_guard.py tls-args EmailAgent)
uvicorn svr_web_agent:app      --host 127.0.0.1 --port 7004 $(python charm_guard.py tls-args WebAgent)
cd webagent && uvicorn webagent_server:app --host 127.0.0.1 --port 9091
```

### CLI client

```bash
python chat_client.py
```

### Tests

```bash
python -m pytest tests -q
```

The tests cover the guard, the attack scenarios, trust boundaries, TLS, the model factory and `page_fetch`.

### Operator tools

```bash
python check_agent_mailbox.py [days]   # read-only check of the agent mailbox for bounces
./list_agent_cards.sh                  # print registered agent cards
```

`check_agent_mailbox.py` is an operator tool; never give it to an agent.

## Running 2.1 and 3.0 side by side

The two versions use different ports, so both can run at once:

| | CHARM 2.1 (before) | CHARM 3.0 (after) |
|---|---|---|
| Folder | `../charm-v2.1` | this folder |
| Dashboard | http://localhost:9090 (red) | http://localhost:9091 (green) |
| Agents | 6001–6004, registry 6060 | 7001–7004, registry 7060, trace 9092 |
| Transport | Plain HTTP on 127.0.0.1 | Mutual TLS on 127.0.0.1 |

From this folder:

```bash
./qcc.sh       # start 2.1, then 3.0
./qccstop.sh   # stop both
```

Each version's own `startup.sh` and `shutdown.sh` touch only that version's ports and processes.

## Demo scripts

```bash
# Attack the running Email Agent as a compromised orchestrator (targets are @example.com)
python demo/attack_compromised_orchestrator.py

# Unauthenticated SMS request. "before" uses 2.1 and sends a REAL text; "after" is dropped by 3.0's TLS
demo/sms_unauthenticated.sh before
demo/sms_unauthenticated.sh after
```

## API Endpoints

### Dashboard (`localhost:9091`)

`/api` calls need the per-boot page token the dashboard serves to its own page.

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Three-panel dashboard |
| `/api/agents` | GET | Agent status and model |
| `/api/agents/{id}/start` · `/stop` · `/restart` | POST | Process control |
| `/api/agents/{id}/model` | PUT | Change an agent's model (restarts it) |
| `/api/chat` | POST | Send a chat message |
| `/api/approvals/{approval_id}` | POST | Approve or deny a pending send |
| `/api/trace/stream` | GET | SSE stream of trace events |
| `/api/profile` · `/api/assistant-profile` | GET/PUT | User and assistant profiles |
| `/api/action-words` | GET | Action words from the registry |

Agents post signed trace events to `https://localhost:9092/api/trace/collect`.

### Registry Agent (`https://localhost:7060`)

| Endpoint | Method | Description |
|---|---|---|
| `/.well-known/agent.json` | GET | Agent card |
| `/tasks/send` | POST | Register, search, update, remove (signed) |
| `/action-words` | GET | Cached action words |

### Chat, Email, SMS and Web agents

| Endpoint | Method | Description |
|---|---|---|
| `/.well-known/agent.json` | GET | Agent card |
| `/` | POST | A2A JSON-RPC `message/send` (signed) |

## Agent Conversation Tracing

`agent_logger.py` provides two levels of tracing.

**Standard:**
```
[Chat] [17:27:26]: Tool called: query_registry("send an SMS text message to a phone number")
[Registry] [17:27:29]: LLM scored 4 agents
[Chat] [17:27:36]: Tool called: call_remote_agent → WebAgent
[Web] [17:27:38]: Tool called: fetch_page(url=https://queencitycon.org/)
[SMS] [17:28:14]: SMS API responded: success (quota remaining: 170)
```

**Verbose (`conversation`)**, for demos:
```
[Web] ⚡ Fetched https://queencitycon.org/ in 0.5s (9213 characters)
[Web] 🛡️ Session now holds external content from fetch_page; side effects will need your approval
[SMS] 📱 Calling SMS API: phone=5555555555, message="..."
[SMS] ✅ SMS API: success — 170 credits remaining
```

## LLM Model Switching

Each agent can use a different model, picked from its dashboard dropdown (the agent restarts):

| Provider | Models |
|---|---|
| Anthropic | Claude Opus 5.5 (default), Claude Sonnet 5.5, Claude Haiku 5.5, Claude Fable 5.1 |
| OpenAI | GPT-6.1 Terra, GPT-6.1 Sol, GPT-6 Luna, GPT-6 Astra |

Long generations are the slowest part of a request, so a faster model on the simple agents helps a lot. Suggested demo setup:
- **Registry, Email, SMS, Web:** Sonnet or Haiku
- **Chat:** your strongest model. In 2.1 the orchestrator's judgment is the only defense against prompt injection, so use the same Chat model in both versions when comparing them.

Asking for a short summary ("a 3-sentence summary") also cuts time, since the summary is written out several times on its way to email and SMS.

## Example Workflows

### 1. Text yourself
```
User: "text me hello"
Chat → Registry: find an agent that can send text messages → SMSAgent
Chat → SMSAgent (signed, capability for your own number): "Send SMS to <your phone> saying hello"
SMSAgent → TextBelt → accepted (170 credits remaining)
```
Your own number is always allowlisted, and nothing untrusted was read, so no approval is needed.

### 2. Read a page, then send a summary
```
User: "read https://queencitycon.org/ and text and email me a 3-sentence summary"
Chat → WebAgent: fetch_page → session is now tainted
Chat → approval card in the dashboard for the email and the SMS
Approve → dashboard signs the capability → EmailAgent and SMSAgent send
```
2.1 runs the same prompt with no approval step.

### 3. Recipient off the allowlist
```
User: "send a 'Happy Veterans Day' email to someone@elsewhere.com"
Chat → approval card ("not on the allowlist") → Approve sends, Deny stops it
```

### 4. Asking an agent to change its own rules
```
User: "Add someone@elsewhere.com to the allowlist"
Chat → Registry: nothing on the network can do this → the Chat Agent says it can't
```
The allowlist is in `charm_policy.json`, and no agent has a tool that writes to it.

### 5. No capable agent
```
User: "Book me a flight"
Chat → Registry: no agent scores high enough → "No agent on the network can do that."
```

## Troubleshooting

**Agents fail to start**
- Check that `.venv` exists with dependencies, and `.env` has valid keys
- Look for port conflicts: `lsof -ti:7060` (or 7001–7004, 9091, 9092)
- Check `keys/` and `keys/tls/` exist (`python charm_guard.py init-keys`)

**401 or TLS errors between agents**
- A caller must be listed for the agent in `charm_policy.json` and must sign with `guard.signed_post_json*`
- Clients need `guard.tls_context()` (httpx) or `guard.requests_tls()` (requests)

**Email "sent" but not received**
- "Accepted by the mail server" isn't delivery. Check the recipient's spam folder; Gmail may file automated mail as spam until you mark it "Not spam".
- Run `python check_agent_mailbox.py 1` to look for bounces
- MXRoute uses server-specific hostnames (for example `witcher.mxrouting.net`)

**SMS slow to arrive**
- TextBelt accepts instantly; carrier delivery varies. Texts over 160 characters go out in parts and are often slower.

**Web requests are slow**
- Give the Web Agent a specific URL so it can use `fetch_page`. `browse_web` takes a minute or more.
- `browse_web` needs `.venv-web` (`./setup_web_venv.sh`)

**Trace events not appearing**
- `TRACE_COLLECTOR_URL` must be `https://localhost:9092/api/trace/collect`, and the dashboard must start first (`startup.sh` does this)

For maximum detail, set `LOG_LEVEL=DEBUG` and `AGENT_LOG_LEVEL=conversation`.

## Proof of Concept Limits

For production you would still want:
- Persistent storage instead of the in-memory registry
- Keys in a secrets manager with rotation (keys are files in `keys/`)
- A real certificate authority instead of the local CA
- Sandboxing for agent processes
- A production SMS provider, monitoring and health checks, retries

## License

MIT License. See `LICENSE`.

## Related Projects

- [Google Agent Development Kit](https://google.github.io/adk-docs/)
- [Google Agent2Agent (A2A) Protocol](https://google-a2a.github.io/A2A/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [LiteLLM](https://docs.litellm.ai/)
- [browser-use](https://github.com/browser-use/browser-use)
- [TextBelt SMS API](https://textbelt.com/)
