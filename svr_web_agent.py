# svr_web_agent.py
# Web Agent — performs live web research / browsing for information that is not in
# the LLM's training data (current events, prices, docs, lookups, etc.).
#
# Uses the browser-use library (https://github.com/browser-use/browser-use):
#   - LOCAL: a headless Chromium driven by browser_use.Agent (free, no API key).
#   - CLOUD: browser-use Cloud (browser_use_sdk) with stealth browsers, residential
#     proxies and CAPTCHA solving, used when the local browser is blocked by bot
#     protection (auto fallback) or when the caller explicitly forces it.
#
# Uses google-adk to_a2a() for A2A protocol handling (same pattern as the SMS agent).

import os
import json
import uuid
import time
import asyncio
import requests
from env_config import load_charm_env
from contextlib import asynccontextmanager

from page_fetch import fetch_page_text, PageFetchError

from google.adk.agents import LlmAgent
from model_factory import build_llm
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

import warnings
warnings.filterwarnings("ignore")

# Load environment
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_charm_env(env_path)

# Logging
from shared_logger import SharedLogger
logger_name = os.path.splitext(os.path.basename(__file__))[0]
logger = SharedLogger.get_logger(logger_name)
logger.info(f"{logger_name} has started.")

# Agent logger (conversation tracing)
from agent_logger import agent_logger, make_callbacks

# CHARM 3.0 guard: this process acts as WebAgent. Only callers named in charm_policy.json may
# reach it, every inbound message is screened and wrapped as untrusted, and tool calls are gated.
import charm_guard as guard
guard.init_identity("WebAgent")

# Configuration
APP_NAME = "web_agent"
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
CONFIG_FILE = os.path.join(os.path.dirname(__file__), f"{SCRIPT_NAME}_config.json")
ACTIVE_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-opus-5-5")

# Browser subprocess config.
# browser-use[core] CANNOT coexist with this framework's google-adk / a2a-sdk
# dependencies (it force-downgrades/removes them), so the actual browsing runs in
# an ISOLATED virtualenv (.venv-web) via web_browser_runner.py. This agent only
# launches that runner as a subprocess and parses its JSON result.
BASE_DIR = os.path.dirname(__file__)
WEB_VENV_PYTHON = os.getenv("WEB_AGENT_VENV_PYTHON",
                            os.path.join(BASE_DIR, ".venv-web", "bin", "python"))
WEB_RUNNER = os.path.join(BASE_DIR, "web_browser_runner.py")
WEB_AGENT_TIMEOUT = int(os.getenv("WEB_AGENT_TIMEOUT", "420"))  # cloud browses can take ~5+ min
# The browser subprocess emits no trace events while it runs, which the dashboard would
# otherwise read as "hung". Emit a heartbeat trace every N seconds during a browse so the
# dashboard's activity watchdog knows real work is still happening.
WEB_AGENT_HEARTBEAT = int(os.getenv("WEB_AGENT_HEARTBEAT", "20"))
# Research budget per A2A request: the agent's LLM can otherwise loop on browse_web
# indefinitely (chasing hard-to-find facts) and blow past the caller's timeout. Cap the
# number of browse_web calls AND the cumulative wall-clock; when exceeded, browse_web stops
# browsing and tells the LLM to answer with what it has. Keep the worst case
# (BUDGET + one in-flight WEB_AGENT_TIMEOUT) safely under the chat agent's call timeout.
WEB_AGENT_MAX_BROWSES = int(os.getenv("WEB_AGENT_MAX_BROWSES", "6"))
WEB_AGENT_BROWSE_BUDGET = int(os.getenv("WEB_AGENT_BROWSE_BUDGET", "240"))
RESULT_SENTINEL = "__BROWSE_RESULT__"
EVENT_SENTINEL = "__BROWSE_EVENT__"

# Per-request browse budget state (reset at agent start via the before_agent callback).
_browse_budget = {"count": 0, "start": 0.0}

def _reset_browse_budget():
    _browse_budget["count"] = 0
    _browse_budget["start"] = time.monotonic()
CLOUD_AVAILABLE = bool(os.getenv("BROWSER_USE_API_KEY", ""))


# Agent card — dict for registry registration, AgentCard object for to_a2a()
AGENT_CARD_DICT = {
    "name": "WebAgent",
    "description": ("A web research agent that browses the live internet to find current, "
                    "up-to-date information that is not in an LLM's training data — news, "
                    "prices, schedules, documentation, facts, and general lookups. It uses a "
                    "headless browser locally and automatically falls back to a cloud browser "
                    "(with proxies and CAPTCHA solving) for bot-protected sites."),
    "url": "https://localhost:7004",
    "version": "0.1",
    "capabilities": {"streaming": False, "pushNotifications": False},
    "skills": [
        {
            "id": "browse_web",
            "name": "Browse the Web",
            "description": ("Performs live web research. Given a natural-language task or question, "
                            "browses the internet and returns the answer or extracted information. "
                            "Use for current events, real-time data, prices, lookups, and anything "
                            "that requires accessing websites."),
            "tags": ["web", "research", "browse", "internet", "lookup", "search", "current", "news"],
            "inputModes": ["application/json", "text/plain"],
            "outputModes": ["application/json"],
        }
    ],
}

AGENT_CARD = AgentCard(
    name="WebAgent",
    description=AGENT_CARD_DICT["description"],
    url="https://localhost:7004",
    version="0.1",
    capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
    defaultInputModes=["application/json", "text/plain"],
    defaultOutputModes=["application/json", "text/plain"],
    skills=[AgentSkill(
        id="browse_web",
        name="Browse the Web",
        description=("Performs live web research. Given a natural-language task or question, "
                     "browses the internet and returns the answer or extracted information."),
        tags=["web", "research", "browse", "internet", "lookup", "search", "current", "news"],
    )],
)


# ---------------------------------------------------------------------------
# Browser subprocess invocation
# ---------------------------------------------------------------------------

def _handle_browse_event(ev: dict):
    """Forward a mid-flight progress event from the runner as a dashboard trace."""
    if ev.get("kind") == "cloud":
        reason = ev.get("reason", "")
        agent_logger.standard("WebAgent", "cloud_switch",
                              f"Switching to browser-use Cloud ({reason})")
        agent_logger.conversation("WebAgent", "cloud_switch",
                                  f"☁️ Switching to browser-use Cloud (stealth browser, proxies, "
                                  f"CAPTCHA solving). Reason: {reason}")


async def _invoke_runner(task: str, force_cloud: bool) -> dict:
    """Run the browsing task in the isolated browser venv via web_browser_runner.py.
    Streams the runner's stdout so mid-flight events (e.g. switching to the cloud) surface
    immediately. Returns the runner's JSON result dict, or an error dict."""
    if not os.path.exists(WEB_VENV_PYTHON):
        return {"error": True,
                "message": (f"The browser runtime is not installed ({WEB_VENV_PYTHON} not found). "
                            "Create the isolated browser environment by running ./setup_web_venv.sh "
                            "(it installs browser-use and Chromium into .venv-web).")}

    spec = json.dumps({"task": task, "force_cloud": force_cloud, "model": ACTIVE_MODEL})
    try:
        proc = await asyncio.create_subprocess_exec(
            WEB_VENV_PYTHON, WEB_RUNNER, spec,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=BASE_DIR,
            # CHARM 3.0: the process browsing arbitrary pages gets only what it needs, never
            # SMTP or SMS credentials or other providers' keys.
            env=guard.scrubbed_env(
                allow=("BROWSER_USE_API_KEY", "DEFAULT_MODEL", "LOG_LEVEL", "AGENT_LOG_LEVEL",
                       "OPENAI_REASONING_EFFORT", "ANONYMIZED_TELEMETRY"),
                allow_prefixes=("WEB_AGENT_", "BROWSER_USE_", "PLAYWRIGHT_"),
                model=ACTIVE_MODEL),
        )
    except Exception as e:
        return {"error": True, "message": f"Failed to launch browser subprocess: {e}"}

    loop = asyncio.get_event_loop()
    deadline = loop.time() + WEB_AGENT_TIMEOUT

    # browser-use is very verbose on stderr; drain it concurrently so a full pipe buffer can
    # never deadlock the subprocess. Keep only the tail for diagnostics.
    stderr_lines: list = []
    async def _drain_stderr():
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            stderr_lines.append(line)
            if len(stderr_lines) > 200:
                del stderr_lines[0]
    stderr_task = asyncio.create_task(_drain_stderr())

    # Heartbeat: the browser subprocess is otherwise silent while it works, so emit a periodic
    # trace to keep the dashboard's activity watchdog from treating a long browse as hung.
    async def _heartbeat():
        elapsed = 0
        while True:
            await asyncio.sleep(WEB_AGENT_HEARTBEAT)
            elapsed += WEB_AGENT_HEARTBEAT
            agent_logger.standard("WebAgent", "browsing", f"Still browsing… ({elapsed}s elapsed)")
            agent_logger.conversation("WebAgent", "browsing", f"⏳ Still browsing… ({elapsed}s elapsed)")
    hb_task = asyncio.create_task(_heartbeat())

    result = None
    timed_out = False
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                break
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except asyncio.TimeoutError:
                timed_out = True
                break
            if not line:
                break  # EOF: subprocess finished
            text = line.decode("utf-8", "replace").rstrip("\r\n")
            if text.startswith(RESULT_SENTINEL):
                try:
                    result = json.loads(text[len(RESULT_SENTINEL):])
                except json.JSONDecodeError:
                    pass
                break
            elif text.startswith(EVENT_SENTINEL):
                try:
                    _handle_browse_event(json.loads(text[len(EVENT_SENTINEL):]))
                except Exception:
                    pass
            # other stdout lines are ignored
    finally:
        hb_task.cancel()

    if timed_out:
        try:
            proc.kill()
        except Exception:
            pass
        stderr_task.cancel()
        return {"error": True, "message": f"Web browsing timed out after {WEB_AGENT_TIMEOUT}s."}

    # Let the process exit and finish draining stderr.
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        await asyncio.wait_for(stderr_task, timeout=5)
    except Exception:
        stderr_task.cancel()

    if result is not None:
        return result

    err_tail = (b"".join(stderr_lines)).decode("utf-8", "replace")[-500:]
    logger.error(f"Browser runner returned no result. stderr tail: {err_tail}")
    return {"error": True,
            "message": f"The browser runner did not return a result. Details: {err_tail[-200:]}"}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

async def fetch_page(url: str) -> dict:
    """
    Quickly read one public web page by URL (a plain HTTP fetch, no browser). Takes a second or
    two, versus a minute or more for browse_web. Use it FIRST whenever you have a specific URL to
    read or summarize.

    Parameters:
        url: the full http(s) URL of the page, e.g. "https://example.com/".

    Returns:
        A dictionary with 'error' (boolean), 'message' (the page's visible text, or why the fetch
        failed), 'title', 'sources' (the URL actually read) and 'hint'. If 'error' is true or
        'hint' says the page looks empty or JavaScript-rendered, use browse_web instead.
    """
    agent_logger.standard("WebAgent", "tool_call", f"Tool called: fetch_page(url={url})")
    agent_logger.conversation("WebAgent", "tool_call", f"⚡ Fetching page directly (no browser): {url}")
    started = time.monotonic()
    try:
        page = await fetch_page_text(url)
    except PageFetchError as e:
        agent_logger.conversation("WebAgent", "tool_error", f"❌ Direct fetch failed: {str(e)[:150]}")
        return {"error": True, "message": str(e), "sources": [],
                "hint": "Use browse_web for this page instead."}
    secs = time.monotonic() - started
    hint = ("The page has very little text; it is probably rendered by JavaScript. Use browse_web."
            if page["sparse"] else "")
    agent_logger.conversation("WebAgent", "tool_response",
                              f"⚡ Fetched {page['final_url']} in {secs:.1f}s "
                              f"({len(page['text'])} characters{', truncated' if page['truncated'] else ''})")
    _emit_sources({"sources": [page["final_url"]]})
    body = page["text"]
    if page["description"]:
        body = f"Meta description: {page['description']}\n\n{body}"
    return {"error": False, "via": "fetch", "title": page["title"], "message": body,
            "truncated": page["truncated"], "sources": [page["final_url"]], "hint": hint}


async def browse_web(task: str, force_cloud: bool = False) -> dict:
    """
    Browse the live internet to research/answer a task. Runs a local headless browser
    first and automatically falls back to the browser-use Cloud (stealth browser with
    proxies and CAPTCHA solving) when the local browser is blocked by bot protection.

    Parameters:
        task: a natural-language description of what to find or do on the web,
              e.g. "Find the current price of the latest iPhone on apple.com".
        force_cloud: set True to skip the local browser and go straight to the cloud
              browser (use when the user explicitly asks for the cloud browser or says
              a site is bot-protected).

    Returns:
        A dictionary with 'error' (boolean), 'message' (the research result or an
        explanation), and 'via' ('local' or 'cloud') indicating which path was used.
    """
    # Enforce the per-request research budget so the agent can't loop forever and blow past
    # the caller's timeout.
    if _browse_budget["start"] == 0.0:
        _reset_browse_budget()
    elapsed = time.monotonic() - _browse_budget["start"]
    if _browse_budget["count"] >= WEB_AGENT_MAX_BROWSES or elapsed > WEB_AGENT_BROWSE_BUDGET:
        agent_logger.standard("WebAgent", "budget",
                              f"Research budget reached ({_browse_budget['count']} browses, {int(elapsed)}s)")
        agent_logger.conversation("WebAgent", "budget",
                                  f"🛑 Research budget reached ({_browse_budget['count']} browses, {int(elapsed)}s) — answering with what was found")
        return {"error": False, "via": "budget",
                "message": ("RESEARCH BUDGET REACHED — do not call browse_web again. "
                            "Compose your final answer NOW using the information already gathered, "
                            "and clearly note anything you were unable to find.")}
    _browse_budget["count"] += 1

    agent_logger.standard("WebAgent", "tool_call",
                          f"Tool called: browse_web(force_cloud={force_cloud}) task=\"{task[:80]}\"")
    if force_cloud:
        agent_logger.conversation("WebAgent", "tool_call",
                                  f"🌐 Cloud browsing (forced): \"{task}\"")
    else:
        agent_logger.conversation("WebAgent", "tool_call",
                                  f"🌐 Browsing locally (headless), cloud fallback if blocked: \"{task}\"")

    result = await _invoke_runner(task, force_cloud)

    via = result.get("via", "")
    if result.get("error"):
        agent_logger.conversation("WebAgent", "tool_error",
                                  f"❌ Browse failed: {str(result.get('message', ''))[:150]}")
    else:
        msg = str(result.get("message", ""))
        icon = "☁️" if via == "cloud" else "✅"
        label = "Cloud browser (bot protection bypassed)" if via == "cloud" else "Local browser"
        agent_logger.conversation("WebAgent", "tool_response",
                                  f"{icon} {label} returned: \"{msg[:150]}{'...' if len(msg) > 150 else ''}\"")
        if via == "cloud":
            _emit_cloud_usage(result)
        _emit_sources(result)
    return result


def _emit_sources(result: dict):
    """Emit a trace listing the pages visited during this browse (for citations)."""
    sources = result.get("sources") or []
    if not sources:
        return
    shown = ", ".join(sources[:5]) + (f" (+{len(sources) - 5} more)" if len(sources) > 5 else "")
    agent_logger.standard("WebAgent", "sources",
                          f"Visited {len(sources)} page(s)", metadata={"sources": sources})
    agent_logger.conversation("WebAgent", "sources",
                              f"🔗 Visited {len(sources)} page(s): {shown}")


def _emit_cloud_usage(result: dict):
    """Emit a trace summarizing this cloud run's cost and the remaining account balance."""
    cost = result.get("cost") or {}
    balance = result.get("balance") or {}
    spent = cost.get("total_cost_usd")
    remaining = balance.get("total_credits_balance_usd")

    parts = []
    if spent is not None:
        parts.append(f"${spent:.4f} spent this run")
    if remaining is not None:
        parts.append(f"${remaining:.2f} credits remaining")
    summary = " · ".join(parts) if parts else "usage details unavailable from the cloud API"

    breakdown = ""
    if cost.get("llm_cost_usd") is not None or cost.get("proxy_cost_usd") is not None:
        breakdown = (f" (llm ${cost.get('llm_cost_usd', 0):.4f}, "
                     f"proxy ${cost.get('proxy_cost_usd', 0):.4f}, "
                     f"browser ${cost.get('browser_cost_usd', 0):.4f})")

    # Structured numbers travel in metadata (only on the standard event) so the dashboard
    # can render a clean usage bubble in the chat without parsing the message string.
    usage_meta = {"cloud_usage": {"spent_usd": spent, "remaining_usd": remaining}}
    agent_logger.standard("WebAgent", "cloud_cost", f"Cloud usage: {summary}", metadata=usage_meta)
    agent_logger.conversation("WebAgent", "cloud_cost", f"💳 Cloud usage: {summary}{breakdown}")


# ---------------------------------------------------------------------------
# Config & Registration
# ---------------------------------------------------------------------------

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        logger.error(f"Could not find config file {CONFIG_FILE}! Exiting...")
        exit(1)

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

async def register(appname: str):
    logger.debug(f"Registering App: {appname}")
    config = load_config()
    agent_id = config.get("agent_id", "")
    reg_url = config.get("reg_url")
    if not reg_url:
        logger.error("Could not load the registration URL from config file.")
        exit(1)

    payload = {
        "id": str(uuid.uuid4()),
        "agent_id": agent_id,
        "tool_name": "register_agent",
        "message": {"parts": [{"text": json.dumps(AGENT_CARD_DICT)}]},
    }

    response = guard.signed_post_json_sync(reg_url, "RegistryAgent", payload)

    if response.status_code == 200:
        try:
            response_data = response.json()
            agent_id = response_data.get("id")
            config["agent_id"] = agent_id
            save_config(config)
            agent_logger.standard("WebAgent", "registration", f"Registered with registry (id: {agent_id})")
            agent_logger.conversation("WebAgent", "registration",
                                      f"🚀 Starting up, registered with registry (id: {agent_id})")
        except ValueError:
            logger.error(f"Failed to parse registration response: {response.text}")
    else:
        logger.error(f"Registration failed with status {response.status_code}: {response.text}")


# ---------------------------------------------------------------------------
# Agent creation and A2A app
# ---------------------------------------------------------------------------

AGENT_PROMPT = f"""
You are a web research agent powered by {ACTIVE_MODEL}. Your primary responsibility is to find
current, real-world information from the live internet using your browse_web tool. You operate
within a multi-agent environment; other trusted agents delegate research tasks to you.

The request will be a natural-language description of something to look up or research on the web
(e.g. "What is the current weather in Tokyo?", "Find the latest price of NVDA stock",
"Summarize the top story on Hacker News right now"). Read the request, understand what information
is needed, and call your browse_web tool with a clear, specific task string.

CHOOSE THE FASTEST TOOL:
- If the request gives a specific URL to read, summarize or check, call fetch_page with that URL
  FIRST. It returns the page text in a second or two. Only fall back to browse_web if fetch_page
  returns an error, or its hint says the page is empty or rendered by JavaScript, or the task needs
  clicking, searching, logging in or following links to other pages.
- Use browse_web when there is no URL (you need to search), or the task needs a real browser.
- Sources from fetch_page count the same as sources from browse_web.

USING THE CLOUD BROWSER:
- By default, browse_web tries a fast local headless browser and automatically falls back to a
  cloud browser (which can bypass bot protection, CAPTCHAs, and region blocks) if the site blocks it.
- If the user EXPLICITLY asks you to use the cloud browser, or says a site is bot-protected /
  behind Cloudflare / blocking automation, call browse_web with force_cloud=True.

After browsing, summarize the findings clearly and directly answer the original request. If the
browse_web result indicates the site was blocked and no cloud fallback was available, explain that
to the user honestly rather than guessing an answer.

Do NOT answer from your own training data when the request asks for current or real-time
information — always use browse_web so the answer reflects the live web.

SOURCES / CITATIONS:
- Each browse_web result includes a "sources" field — the actual list of web pages that were
  visited. These are the ONLY valid citations. Never invent, guess, or add URLs that are not in
  a "sources" field.
- At the END of your response, always include a "Sources:" section that lists every URL from the
  "sources" fields of all your browse_web calls (de-duplicated). If you made several browse_web
  calls, consolidate their sources into one list. If there were no sources, say "Sources: none".
- This lets the requesting agent provide citations on demand. Keep the list complete and accurate.

BE EFFICIENT — AVOID ENDLESS LOOPING:
- Use as few browse_web calls as possible. Do NOT keep retrying the same lookup with slightly
  reworded queries; if two or three attempts do not find a specific detail (e.g. an obscure code),
  accept that it is unavailable and move on.
- Partial results are acceptable. It is better to return what you found and clearly state what you
  could NOT find than to keep browsing indefinitely.
- If a browse_web result says "RESEARCH BUDGET REACHED", STOP calling browse_web immediately and
  compose your final answer from what you already have, noting anything missing.

MOST IMPORTANTLY:
- THINK DEEPLY, but be decisive. A correct, well-sourced answer that admits gaps beats an endless search.
"""

# Build callbacks from agent_logger, then reset the research budget at the start of each request.
callbacks = make_callbacks("WebAgent", ACTIVE_MODEL)
_orig_before_agent = callbacks.get("before_agent_callback")

def _before_agent_with_budget(callback_context, **kwargs):
    _reset_browse_budget()
    if _orig_before_agent:
        return _orig_before_agent(callback_context, **kwargs)
    return None

callbacks["before_agent_callback"] = _before_agent_with_budget

# Guard: screen and wrap inbound requests; wrap browse results as untrusted web content.
callbacks = guard.make_guard_callbacks("WebAgent", inbound="peer", base_callbacks=callbacks,
                                       untrusted_tools=["browse_web", "fetch_page"])

# Create the agent
web_agent = LlmAgent(
    name="web_agent",
    model=build_llm(ACTIVE_MODEL),
    description="An agent that performs live web research/browsing using the browser-use library, with local headless and cloud fallback.",
    instruction=AGENT_PROMPT + guard.GUARD_PREAMBLE,
    tools=[fetch_page, browse_web],
    **callbacks,
)

# Lifespan: register with registry on startup
@asynccontextmanager
async def lifespan(app):
    if CLOUD_AVAILABLE:
        logger.info("browser-use Cloud fallback is available (BROWSER_USE_API_KEY set).")
    else:
        logger.info("browser-use Cloud fallback NOT configured — local headless only.")
    await register(APP_NAME)
    yield
    logger.info("Shutting down Web agent.")

# Create the A2A Starlette app
app = to_a2a(
    web_agent,
    host="127.0.0.1",
    port=7004,
    agent_card=AGENT_CARD,
    lifespan=lifespan,
)
app.add_middleware(guard.GuardMiddleware, agent_name="WebAgent")

if __name__ == "__main__":
    print("Run with: uvicorn svr_web_agent:app --host 127.0.0.1 --port 7004")
