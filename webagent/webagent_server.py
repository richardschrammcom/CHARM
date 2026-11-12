# webagent_server.py
# Three-panel web UI for the multi-agent A2A system.
# Left: Agent roster with toggle switches
# Middle: Chat interface
# Right: Real-time conversation trace

import os
import sys
import uuid
import json
import asyncio
import time
import signal
import secrets
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import contextlib
import uvicorn
import httpx
import requests
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette import EventSourceResponse
from pydantic import BaseModel

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_logger import SharedLogger
from agent_logger import AgentLogger, AgentEvent
from env_config import load_charm_env

# Load environment variables from parent directory
load_charm_env(Path(__file__).parent.parent / ".env")

# CHARM 3.0 guard: the dashboard is the human's console. It signs its calls as "Dashboard", mints
# capability tokens when you approve an action, and only accepts signed trace events from agents.
import charm_guard as guard
guard.init_identity("Dashboard")
UI_TOKEN = secrets.token_urlsafe(24)  # per boot; embedded in the page, required on every /api call

# Configuration
WEB_PORT = int(os.getenv("WEB_PORT", "9091"))
LOGO_IMAGE = os.getenv("LOGO_IMAGE", "logo_placeholder.png")
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "https://127.0.0.1:7002")
REGISTRY_URL = os.getenv("REGISTRY_URL", "https://127.0.0.1:7060")
# The browser uses WEB_PORT over plain loopback HTTP. Agents send trace events here, over mutual TLS.
TRACE_PORT = int(os.getenv("DASHBOARD_TRACE_PORT", "9092"))
MODULE_PRINCIPALS = {"svr_registry_agent": "RegistryAgent", "svr_chat_agent": "ChatAgent", "svr_email_agent": "EmailAgent",
                     "svr_sms_agent": "SMSAgent", "svr_web_agent": "WebAgent"}
# Dashboard -> Chat Agent timeout. Must be the LARGEST in the chain (it waits on the
# chat agent, which waits on specialist agents like the Web Agent's multi-step browsing).
CHAT_REQUEST_TIMEOUT = int(os.getenv("DASHBOARD_CHAT_TIMEOUT", "780"))
# Liveness watchdog for in-flight chat requests. Instead of a fixed wall-clock cap, a
# request is considered hung only if NO trace activity arrives from any agent for
# CHAT_IDLE_LIMIT seconds (the trace stream is the heartbeat). CHAT_MAX_RUNTIME is a
# hard backstop so a chatty-but-looping agent can't run forever.
CHAT_IDLE_LIMIT = int(os.getenv("CHAT_IDLE_LIMIT", "90"))
CHAT_MAX_RUNTIME = int(os.getenv("CHAT_MAX_RUNTIME", "1200"))
PARENT_DIR = str(Path(__file__).parent.parent)

# Logging
logger_name = os.path.splitext(os.path.basename(__file__))[0]
logger = SharedLogger.get_logger(logger_name)
logger.info(f"{logger_name} has started.")

USER_PROFILE_PATH = os.path.join(PARENT_DIR, "user_profile.json")
ASSISTANT_PROFILE_PATH = os.path.join(PARENT_DIR, "assistant_profile.json")

# FastAPI app
app = FastAPI(title="Agent Dashboard")
# Trace events must be signed by an agent; everything else under /api needs the page's UI token
# and a loopback Host header (blocks other sites, DNS rebinding, and stray local processes).
app.add_middleware(guard.GuardMiddleware, agent_name="Dashboard", protect_prefixes=("/api/trace/collect",),
                   on_block=lambda event_type, message, meta: _emit_trace("Dashboard", event_type, message))
app.add_middleware(guard.UITokenMiddleware, token=UI_TOKEN, port=WEB_PORT, extra_ports=(TRACE_PORT,),
                   exempt_prefixes=("/api/trace/collect",))

# Static files and templates
webagent_path = Path(__file__).parent
static_path = webagent_path / "static"
templates_path = webagent_path / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

# ---------------------------------------------------------------------------
# Agent process management
# ---------------------------------------------------------------------------

# Agent definitions: name, module, port, config
AGENT_DEFS = {
    "registry": {
        "name": "Registry Agent",
        "module": "svr_registry_agent",
        "port": 7060,
        "description": "Central agent registry with semantic search",
        "required": True,  # Cannot be toggled off
    },
    "chat": {
        "name": "Chat Agent",
        "module": "svr_chat_agent",
        "port": 7002,
        "description": "Orchestrator that discovers and delegates to other agents",
        "required": True,
    },
    "sms": {
        "name": "SMS Agent",
        "module": "svr_sms_agent",
        "port": 7003,
        "description": "Sends SMS text messages via third-party API",
        "required": False,
    },
    "email": {
        "name": "Email Agent",
        "module": "svr_email_agent",
        "port": 7001,
        "description": "Sends emails via SMTP",
        "required": False,
    },
    "web": {
        "name": "Web Agent",
        "module": "svr_web_agent",
        "port": 7004,
        "description": "Researches the live web (local headless browser + cloud fallback)",
        "required": False,
    },
}

# Available LLM models for the per-agent dropdown. Both providers are live; agents can be
# mixed provider-by-provider in a demo. Claude Opus 5.5 is the default; the OpenAI
# counterpart is GPT-6.1 Terra.
AVAILABLE_MODELS = [
    {"id": "anthropic/claude-opus-5-5", "label": "Claude Opus 5.5"},
    {"id": "anthropic/claude-sonnet-5-5", "label": "Claude Sonnet 5.5"},
    {"id": "anthropic/claude-haiku-5-5", "label": "Claude Haiku 5.5"},
    {"id": "anthropic/claude-fable-5-1", "label": "Claude Fable 5.1"},
    {"id": "openai/gpt-6.1-terra", "label": "GPT-6.1 Terra"},
    {"id": "openai/gpt-6.1-sol", "label": "GPT-6.1 Sol"},
    {"id": "openai/gpt-6-luna", "label": "GPT-6 Luna"},
    {"id": "openai/gpt-6-astra", "label": "GPT-6 Astra"},
]

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-opus-5-5")

# Track running agent processes and per-agent model selections
agent_processes: Dict[str, subprocess.Popen] = {}
agent_models: Dict[str, str] = {}  # agent_id -> model_id


def is_agent_running(agent_id: str) -> bool:
    """Check if an agent is running by testing if the process exists or the port is listening."""
    # First check: do we have a tracked subprocess that's still alive?
    proc = agent_processes.get(agent_id)
    if proc and proc.poll() is None:
        return True

    # Second check: is anything listening on the port? (handles startup.sh-launched agents)
    agent_def = AGENT_DEFS.get(agent_id)
    if not agent_def:
        return False
    port = agent_def["port"]
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=2
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def is_agent_registered(agent_id: str) -> bool:
    """Check if an agent is in the registry."""
    try:
        resp = requests.get(f"{REGISTRY_URL}/.well-known/agent.json", timeout=2, **guard.requests_tls())
        if resp.status_code != 200:
            return False
        # Use list_agent_cards to check
        payload = {
            "id": str(uuid.uuid4()),
            "tool_name": "list_agent_cards",
            "message": {"parts": [{"text": ""}]},
        }
        result = guard.signed_post_json_sync(f"{REGISTRY_URL}/tasks/send", "RegistryAgent", payload, timeout=5)
        if result.status_code == 200:
            cards = result.json()
            if isinstance(cards, list):
                agent_def = AGENT_DEFS.get(agent_id, {})
                agent_name = agent_def.get("name", "").replace(" Agent", "Agent")
                return any(c.get("name", "") == agent_name for c in cards)
        return False
    except Exception:
        return False


def start_agent(agent_id: str) -> bool:
    """Start an agent process."""
    agent_def = AGENT_DEFS.get(agent_id)
    if not agent_def:
        return False

    if is_agent_running(agent_id):
        return True  # Already running

    module = agent_def["module"]
    port = agent_def["port"]

    try:
        # Build environment: inherit current env, override model if set per-agent
        child_env = os.environ.copy()
        model = agent_models.get(agent_id, DEFAULT_MODEL)
        child_env["DEFAULT_MODEL"] = model

        # Redirect stdout/stderr to log file (not PIPE, which can block)
        log_file = os.path.join(PARENT_DIR, f"{module.replace('svr_', '')}.log")
        log_fh = open(log_file, "w")

        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", f"{module}:app",
             "--host", "127.0.0.1", "--port", str(port), *guard.tls_server_args(MODULE_PRINCIPALS[module])],
            cwd=PARENT_DIR,
            env=child_env,
            stdout=log_fh,
            stderr=log_fh,
        )
        agent_processes[agent_id] = proc
        # Update PID file so shutdown.sh can find it
        pid_file = os.path.join(PARENT_DIR, f"{module.replace('svr_', '').replace('_agent', '')}_agent.pid")
        try:
            with open(pid_file, "w") as f:
                f.write(str(proc.pid))
        except Exception:
            pass
        logger.info(f"Started {agent_def['name']} (pid={proc.pid}) on port {port} with model {model}")
        return True
    except Exception as e:
        logger.error(f"Failed to start {agent_def['name']}: {e}")
        return False


def stop_agent(agent_id: str) -> bool:
    """Stop an agent process."""
    stopped = False

    # Try tracked subprocess first
    proc = agent_processes.get(agent_id)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        stopped = True

    if agent_id in agent_processes:
        del agent_processes[agent_id]

    # Also kill the process listening on the port (handles startup.sh-launched agents)
    agent_def = AGENT_DEFS.get(agent_id)
    if agent_def:
        port = agent_def["port"]
        try:
            # Only kill the LISTEN process, not clients connected to it
            result = subprocess.run(
                ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                capture_output=True, text=True
            )
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid.strip():
                    os.kill(int(pid.strip()), signal.SIGTERM)
                    stopped = True
            if stopped:
                # Wait for port to actually free up
                import time as _time
                for _ in range(10):
                    check = subprocess.run(
                        ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
                        capture_output=True, text=True
                    )
                    if not check.stdout.strip():
                        break
                    _time.sleep(0.5)
        except Exception:
            pass

    if stopped:
        logger.info(f"Stopped {AGENT_DEFS[agent_id]['name']}")
    return stopped


# ---------------------------------------------------------------------------
# Conversation trace streaming
# ---------------------------------------------------------------------------

# SSE subscribers for conversation trace
trace_subscribers: List[asyncio.Queue] = []

# Liveness heartbeat: timestamp of the most recent trace event seen from ANY agent.
# The chat watchdog uses this to tell "working" from "hung" without a fixed timeout.
_last_trace_ts: float = time.time()
_last_trace_agent: str = ""


def _mark_activity(agent_name: str = ""):
    """Record that an agent just did something (heartbeat for the chat watchdog)."""
    global _last_trace_ts, _last_trace_agent
    _last_trace_ts = time.time()
    if agent_name:
        _last_trace_agent = agent_name


def _push_to_subscribers(item: dict):
    """Enqueue an item (trace event or SSE envelope) to all SSE subscribers."""
    for queue in trace_subscribers[:]:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass
        except Exception:
            pass


def trace_subscriber_callback(event: AgentEvent):
    """Push agent events to all SSE subscribers."""
    event_data = {
        "timestamp": event.timestamp,
        "agent_name": event.agent_name,
        "level": event.level,
        "event_type": event.event_type,
        "message": event.message,
        "trace_id": event.trace_id,
        "metadata": event.metadata,
    }
    _mark_activity(event.agent_name)
    _push_to_subscribers(event_data)


# Register our subscriber with the AgentLogger
agent_logger = AgentLogger()
agent_logger.add_subscriber(trace_subscriber_callback)


def _emit_trace(agent_name: str, event_type: str, message: str):
    """Push a trace event directly to all SSE subscribers."""
    event_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "agent_name": agent_name,
        "level": "standard",
        "event_type": event_type,
        "message": message,
        "trace_id": None,
        "metadata": {},
    }
    _mark_activity(agent_name)
    _push_to_subscribers(event_data)


def _emit_chat_result(request_id: str, error: bool, message: str, context_id: Optional[str]):
    """Deliver a finished (or stalled) chat result to the browser over the SSE stream."""
    _push_to_subscribers({
        "__sse_event__": "chat_result",
        "payload": {
            "request_id": request_id,
            "error": error,
            "message": message,
            "context_id": context_id,
        },
    })


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the three-panel dashboard."""
    session_id = str(uuid.uuid4())
    cache_bust = str(int(time.time()))
    # Load assistant name for welcome message
    try:
        with open(ASSISTANT_PROFILE_PATH, "r") as f:
            assistant = json.load(f)
    except Exception:
        assistant = {"name": "Sam"}
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "session_id": session_id,
        "logo_image": LOGO_IMAGE,
        "cache_bust": cache_bust,
        "assistant_name": assistant.get("name", "Sam"),
        "ui_token": UI_TOKEN,
    })


@app.get("/api/agents")
async def get_agents():
    """Get status of all agents."""
    agents = []
    for agent_id, agent_def in AGENT_DEFS.items():
        running = is_agent_running(agent_id)
        registered = is_agent_registered(agent_id) if running and agent_id not in ("registry",) else False
        agents.append({
            "id": agent_id,
            "name": agent_def["name"],
            "port": agent_def["port"],
            "description": agent_def["description"],
            "required": agent_def["required"],
            "running": running,
            "registered": registered,
            "model": agent_models.get(agent_id, DEFAULT_MODEL),
        })
    return {"agents": agents, "available_models": AVAILABLE_MODELS}


@app.post("/api/agents/{agent_id}/start")
async def api_start_agent(agent_id: str):
    """Start an agent."""
    if agent_id not in AGENT_DEFS:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent_display = AGENT_DEFS[agent_id]["name"].replace(" Agent", "Agent")
    _emit_trace(agent_display, "start", f"Starting...")
    success = start_agent(agent_id)
    for _ in range(15):
        if is_agent_running(agent_id):
            _emit_trace(agent_display, "start", f"✅ Started")
            return {"status": "started", "agent_id": agent_id}
        await asyncio.sleep(1)
    _emit_trace(agent_display, "start", f"❌ Failed to start")
    return {"status": "starting" if success else "failed", "agent_id": agent_id}


@app.post("/api/agents/{agent_id}/stop")
async def api_stop_agent(agent_id: str):
    """Stop an agent."""
    if agent_id not in AGENT_DEFS:
        raise HTTPException(status_code=404, detail="Agent not found")
    if AGENT_DEFS[agent_id].get("required"):
        raise HTTPException(status_code=400, detail="Cannot stop required agent")
    agent_display = AGENT_DEFS[agent_id]["name"].replace(" Agent", "Agent")
    success = stop_agent(agent_id)
    if success:
        _emit_trace(agent_display, "stop", f"🛑 Stopped")
    return {"status": "stopped" if success else "failed", "agent_id": agent_id}


@app.post("/api/agents/{agent_id}/restart")
async def api_restart_agent(agent_id: str):
    """Restart an agent (stop + start)."""
    if agent_id not in AGENT_DEFS:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent_display = AGENT_DEFS[agent_id]["name"].replace(" Agent", "Agent")
    model = agent_models.get(agent_id, DEFAULT_MODEL)

    _emit_trace(agent_display, "restart", f"Restarting with model {model.split('/')[-1]}...")

    stop_agent(agent_id)
    await asyncio.sleep(2)
    start_agent(agent_id)
    for _ in range(15):
        if is_agent_running(agent_id):
            _emit_trace(agent_display, "restart", f"✅ Restarted successfully")
            return {"status": "restarted", "agent_id": agent_id}
        await asyncio.sleep(1)

    _emit_trace(agent_display, "restart", f"❌ Restart failed")
    return {"status": "restart_failed", "agent_id": agent_id}


@app.put("/api/agents/{agent_id}/model")
async def set_agent_model(agent_id: str, request: Request):
    """Change the LLM model for an agent. Restarts the agent with the new model."""
    if agent_id not in AGENT_DEFS:
        raise HTTPException(status_code=404, detail="Agent not found")

    body = await request.json()
    new_model = body.get("model", "")
    if not new_model:
        raise HTTPException(status_code=400, detail="Model is required")

    old_model = agent_models.get(agent_id, DEFAULT_MODEL)
    agent_models[agent_id] = new_model
    agent_name = AGENT_DEFS[agent_id]["name"]

    agent_display = agent_name.replace(" Agent", "Agent")
    logger.info(f"Changing {agent_name} model: {old_model} → {new_model}")
    _emit_trace(agent_display, "model_change",
                f"Model changed: {old_model.split('/')[-1]} → {new_model.split('/')[-1]}")

    # Restart the agent if it's running
    was_running = is_agent_running(agent_id)
    if was_running:
        stop_agent(agent_id)
        await asyncio.sleep(2)
        start_agent(agent_id)
        # Wait for it to come back
        for _ in range(15):
            if is_agent_running(agent_id):
                break
            await asyncio.sleep(1)

    return {
        "status": "ok",
        "agent_id": agent_id,
        "model": new_model,
        "restarted": was_running,
    }


async def _call_chat_agent(payload: dict, context_id: Optional[str]) -> dict:
    """Make the actual A2A JSON-RPC call to the chat agent and normalize the reply.
    Raises httpx errors / asyncio.CancelledError to the caller (the worker handles them)."""
    async with httpx.AsyncClient(timeout=CHAT_REQUEST_TIMEOUT, verify=guard.tls_context()) as client:
        result = await guard.signed_post_json(client, AGENT_BASE_URL, "ChatAgent", payload)

    if result.status_code != 200:
        return {"error": True, "message": f"Agent returned {result.status_code}", "context_id": context_id}

    rpc_response = result.json()
    if "error" in rpc_response:
        return {"error": True, "message": rpc_response["error"].get("message", "Unknown error"),
                "context_id": context_id}

    task_result = rpc_response.get("result", {})
    new_context_id = task_result.get("contextId", context_id)
    reply_text = ""
    for artifact in task_result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text":
                reply_text += part.get("text", "")

    return {"error": False, "message": reply_text or "No response from agent.", "context_id": new_context_id}


async def _chat_worker(request_id: str, payload: dict, context_id: Optional[str]):
    """Run a chat request in the background, supervised by an activity watchdog.

    The request is NOT interrupted while agents keep emitting trace events (it may
    legitimately take minutes for web research). It is aborted only if the trace stream
    goes silent for CHAT_IDLE_LIMIT seconds (hung), or it exceeds CHAT_MAX_RUNTIME.
    The result (or stall notice) is delivered to the browser over the SSE stream.
    """
    _mark_activity("ChatAgent")  # reset heartbeat at launch
    started = time.time()
    task = asyncio.create_task(_call_chat_agent(payload, context_id))

    try:
        while not task.done():
            await asyncio.sleep(3)
            idle = time.time() - _last_trace_ts
            runtime = time.time() - started

            if idle > CHAT_IDLE_LIMIT:
                task.cancel()
                stuck = _last_trace_agent or "an agent"
                _emit_chat_result(
                    request_id, True,
                    f"⚠️ The request appears to be stuck — no activity from {stuck} for "
                    f"{int(idle)}s, so I stopped waiting. The agent may be hung; you can retry, "
                    f"or restart it from the roster on the left.",
                    context_id)
                return
            if runtime > CHAT_MAX_RUNTIME:
                task.cancel()
                _emit_chat_result(
                    request_id, True,
                    f"⚠️ The request ran past the {CHAT_MAX_RUNTIME}s safety limit and was stopped, "
                    f"even though agents were still active. It may be looping.",
                    context_id)
                return

        result = task.result()
        _emit_chat_result(request_id, result["error"], result["message"], result.get("context_id"))

    except asyncio.CancelledError:
        # Task was cancelled by the watchdog above; notice already emitted.
        pass
    except httpx.ConnectError:
        _emit_chat_result(request_id, True, "Cannot connect to Chat Agent. Is it running?", context_id)
    except Exception as e:
        logger.error(f"Chat worker error: {e}")
        _emit_chat_result(request_id, True, f"Chat request failed: {e}", context_id)


@app.post("/api/chat")
async def send_chat(request: Request):
    """Launch a chat request in the background and return immediately.

    The final reply is delivered asynchronously over the SSE trace stream as a
    'chat_result' event, so long-running web research no longer blocks the request.
    """
    body = await request.json()
    message = body.get("message", "")
    context_id = body.get("context_id")

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": message}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }
    if context_id:
        payload["params"]["message"]["contextId"] = context_id

    request_id = str(uuid.uuid4())
    asyncio.create_task(_chat_worker(request_id, payload, context_id))
    return {"status": "started", "request_id": request_id}


# Approvals waiting on the human, keyed by approval_id (filled from signed ChatAgent trace events).
pending_approvals: Dict[str, dict] = {}


@app.post("/api/approvals/{approval_id}")
async def resolve_approval(approval_id: str, request: Request):
    """The human's decision. The dashboard, not the model, signs the capability token."""
    body = await request.json()
    decision = "approve" if body.get("decision") == "approve" else "deny"
    meta = pending_approvals.pop(approval_id, None)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown or expired approval")
    capability = None
    if decision == "approve":
        capability = guard.mint_capability(meta["agent"], meta["action"], meta["recipients"], meta["kind"],
                                           approval_id=approval_id)
    payload = {"approval_id": approval_id, "decision": decision, "capability": capability}
    try:
        async with httpx.AsyncClient(timeout=10, verify=guard.tls_context()) as client:
            resp = await guard.signed_post_json(client, f"{AGENT_BASE_URL}/charm/approval", "ChatAgent", payload)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Chat agent unreachable: {e}")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    verb = "✅ You approved" if decision == "approve" else "🚫 You denied"
    _emit_trace("Dashboard", "approval_decision", f"{verb} {meta['action']} to {', '.join(meta['recipients'])}")
    return {"status": decision}


@app.get("/api/trace/stream")
async def stream_trace():
    """SSE endpoint streaming conversation trace events."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    trace_subscribers.append(queue)

    async def event_generator():
        try:
            yield {"event": "connected", "data": json.dumps({"message": "Trace stream connected"})}
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=30)
                    # Items may be raw trace events or tagged SSE envelopes (e.g. chat results).
                    if isinstance(item, dict) and item.get("__sse_event__"):
                        yield {"event": item["__sse_event__"], "data": json.dumps(item.get("payload", {}))}
                    else:
                        yield {"event": "trace", "data": json.dumps(item)}
                except asyncio.TimeoutError:
                    yield {"event": "keepalive", "data": json.dumps({"ts": time.time()})}
        finally:
            if queue in trace_subscribers:
                trace_subscribers.remove(queue)

    return EventSourceResponse(event_generator())


@app.get("/api/assistant-profile")
async def get_assistant_profile():
    """Get the assistant profile."""
    try:
        with open(ASSISTANT_PROFILE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"name": "Sam", "personality": "You are a helpful assistant with a friendly disposition."}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.put("/api/assistant-profile")
async def update_assistant_profile(request: Request):
    """Update the assistant profile and restart the chat agent to apply."""
    try:
        profile = await request.json()
        with open(ASSISTANT_PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Assistant profile updated: {profile.get('name', '?')}")

        # Auto-restart chat agent to pick up the new personality
        _emit_trace("ChatAgent", "profile_change",
                    f"Assistant profile updated: name={profile.get('name', '?')}. Restarting...")
        stop_agent("chat")
        await asyncio.sleep(2)
        start_agent("chat")
        for _ in range(15):
            if is_agent_running("chat"):
                _emit_trace("ChatAgent", "profile_change", f"✅ Restarted with new personality")
                break
            await asyncio.sleep(1)

        return {"status": "saved", "restarted": True}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/profile")
async def get_profile():
    """Get the user profile."""
    try:
        with open(USER_PROFILE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"name": "", "email": "", "phone": "", "role": "", "notes": ""}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.put("/api/profile")
async def update_profile(request: Request):
    """Update the user profile."""
    try:
        profile = await request.json()
        with open(USER_PROFILE_PATH, "w") as f:
            json.dump(profile, f, indent=2)
        logger.info(f"User profile updated: {profile.get('name', '?')}")
        return {"status": "saved"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/trace/collect")
async def collect_trace(request: Request):
    """Receive trace events from remote agent processes."""
    if request.url.scheme != "https":
        return JSONResponse({"error": f"trace events must use the TLS port {TRACE_PORT}"}, status_code=426)
    try:
        event_data = await request.json()
        # CHARM 3.0: the event is signed, so we know who really sent it. An agent can't speak
        # for another agent in the audit trail.
        ctx = guard.request_context()
        caller = ctx.caller if ctx else "unknown"
        claimed = event_data.get("agent_name", "")
        if claimed != caller:
            event_data["message"] = f"[claimed to be {claimed}] {event_data.get('message', '')}"
            event_data["agent_name"] = caller
        # The browser draws a card for any approval event it sees, and only the orchestrator asks
        # for approvals. Anything else is shown as what it is: an agent trying to fake one.
        if event_data.get("event_type") in ("approval_request", "approval_resolved") and caller != "ChatAgent":
            event_data["message"] = f"🛡️ Ignored a fake {event_data['event_type']} from {caller}"
            event_data["event_type"] = "guard_block"
            event_data["metadata"] = {"guard": True}
        if event_data.get("event_type") == "approval_request" and caller == "ChatAgent":
            meta = event_data.get("metadata") or {}
            if meta.get("approval_id"):
                pending_approvals[meta["approval_id"]] = meta
        # Heartbeat: a remote agent just reported activity.
        _mark_activity(event_data.get("agent_name", ""))
        # Push to all SSE subscribers
        _push_to_subscribers(event_data)
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/action-words")
async def get_action_words():
    """Fetch action word cache from the registry agent."""
    try:
        resp = requests.get(f"{REGISTRY_URL}/action-words", headers=guard.sign_request("RegistryAgent"), timeout=5,
                            **guard.requests_tls())
        if resp.status_code == 200:
            return resp.json()
        return {}
    except Exception:
        return {}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "agent_url": AGENT_BASE_URL}


class _TraceServer(uvicorn.Server):
    """Second listener in the dashboard's event loop, so traces reach the same SSE subscribers."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield  # the main server owns signal handling


_trace_server: Optional[_TraceServer] = None


@app.on_event("startup")
async def start_trace_listener():
    global _trace_server
    _trace_server = _TraceServer(uvicorn.Config(app, host="127.0.0.1", port=TRACE_PORT, lifespan="off", log_level="warning",
                                                **guard.tls_server_kwargs("Dashboard")))
    app.state.trace_listener = asyncio.create_task(_trace_server.serve())


@app.on_event("shutdown")
async def stop_trace_listener():
    if _trace_server:
        _trace_server.should_exit = True


@app.on_event("startup")
async def startup_event():
    logger.info(f"Dashboard starting on port {WEB_PORT}")
    logger.info(f"Chat Agent URL: {AGENT_BASE_URL}")
    logger.info(f"Registry URL: {REGISTRY_URL}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=WEB_PORT, log_level="info")
