#!/usr/bin/env python
# web_browser_runner.py
#
# Standalone browser-use runner — executed by svr_web_agent.py as a subprocess in an
# ISOLATED virtualenv (.venv-web). browser-use[core] hard-pins dependency versions that
# conflict with (and remove) the Charm framework's google-adk / a2a-sdk packages, so it
# cannot live in the framework venv. This script must therefore import ONLY browser-use
# (no framework modules).
#
# Protocol:
#   - Input:  a JSON spec as argv[1] (or on stdin), e.g.
#             {"task": "...", "force_cloud": false, "model": "anthropic/claude-opus-5-5"}
#   - Output: exactly one line on stdout prefixed with RESULT_SENTINEL containing the
#             JSON result {"error": bool, "via": "local"|"cloud", "message": str, ...}.
#             All browser-use logging goes to stderr, keeping stdout clean for the result.

import os
import sys
import json
import asyncio

RESULT_SENTINEL = "__BROWSE_RESULT__"


def _cfg_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


HEADLESS = _cfg_bool("WEB_AGENT_HEADLESS", "true")
CLOUD_FALLBACK = _cfg_bool("WEB_AGENT_CLOUD_FALLBACK", "true")
BROWSER_USE_API_KEY = os.getenv("BROWSER_USE_API_KEY", "")
CLOUD_MODEL = os.getenv("WEB_AGENT_CLOUD_MODEL", "")
CLOUD_AVAILABLE = bool(BROWSER_USE_API_KEY)

_DEFAULT_BLOCK_MARKERS = (
    "captcha,cloudflare,verify you are human,are you a robot,access denied,"
    "403 forbidden,unusual traffic,enable javascript,bot detection,security check"
)
BLOCK_MARKERS = [m.strip().lower() for m in
                 os.getenv("WEB_AGENT_BLOCK_MARKERS", _DEFAULT_BLOCK_MARKERS).split(",") if m.strip()]

# Markers that indicate the LLM provider failed (quota/auth/rate limit) rather than the
# site being bot-protected. These should be reported clearly, not retried on the cloud.
_LLM_ERROR_MARKERS = (
    "insufficient_quota", "exceeded your current quota", "rate limit", "ratelimit",
    "error code: 429", "invalid api key", "incorrect api key", "authentication",
    "permission", "billing",
)


EVENT_SENTINEL = "__BROWSE_EVENT__"


def emit(obj: dict) -> None:
    """Write the single (final) result line to stdout."""
    sys.stdout.write(RESULT_SENTINEL + json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_event(obj: dict) -> None:
    """Write a mid-flight progress event line that the Web Agent forwards as a trace."""
    sys.stdout.write(EVENT_SENTINEL + json.dumps(obj) + "\n")
    sys.stdout.flush()


def _build_llm(model_str: str):
    """Map a litellm-style model string to a browser-use chat model object."""
    provider, _, model_name = model_str.partition("/")
    if not model_name:
        provider, model_name = "openai", model_str
    provider = provider.lower()
    if provider == "anthropic":
        from browser_use import ChatAnthropic
        # Claude x-5-5 models reject forced tool use. browser-use only switches to
        # tool_choice "auto" for them when thinking is set, so name their default.
        if model_name.endswith("-5-5"):
            return ChatAnthropic(model=model_name, thinking={"type": "adaptive"})
        return ChatAnthropic(model=model_name)
    if provider in ("google", "gemini"):
        from browser_use import ChatGoogle
        return ChatGoogle(model=model_name)
    from browser_use import ChatOpenAI
    # browser-use drives the model with tools/structured output, which OpenAI's
    # chat-completions endpoint rejects while a reasoning effort is active on the
    # gpt-5.x family. Same quirk (and same fix) as model_factory.llm_kwargs.
    if model_name.startswith("gpt-5"):
        return ChatOpenAI(model=model_name, reasoning_effort="none")
    # gpt-6.x accepts only its default sampling, and browser-use sends its own otherwise.
    if model_name.startswith("gpt-6"):
        return ChatOpenAI(model=model_name, temperature=None, frequency_penalty=None)
    return ChatOpenAI(model=model_name)


def _first_block_marker(text: str) -> str:
    if not text:
        return ""
    low = text.lower()
    for marker in BLOCK_MARKERS:
        if marker in low:
            return marker
    return ""


_SKIP_URL_PREFIXES = ("about:", "data:", "chrome:", "chrome-error:", "javascript:")


def _clean_urls(urls) -> list:
    """Dedupe (preserving order) and drop non-source URLs (about:blank, data:, etc.)."""
    seen = set()
    out = []
    for u in urls or []:
        if not u or not isinstance(u, str):
            continue
        u = u.strip().rstrip("/")  # normalize trailing slash so /x and /x/ dedupe
        if not u or u.lower().startswith(_SKIP_URL_PREFIXES):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


async def _run_browser(task: str, model: str, use_cloud: bool):
    """Drive a browser-use Agent (reasoning by the given Claude/LLM) over either a local
    headless Chromium (use_cloud=False) or a browser-use Cloud stealth browser
    (use_cloud=True — same agent/LLM, just a remote anti-bot browser via CDP).
    Returns (text, status, detail, sources); status is 'ok', 'blocked', or 'llm_error'."""
    from browser_use import Agent as BrowserAgent, Browser

    llm = _build_llm(model)
    if use_cloud:
        # cloud_browser=True → browser-use spins up a stealth cloud browser (residential
        # proxies, anti-fingerprinting) and the local agent drives it over CDP.
        browser = Browser(cloud_browser=True)
    else:
        browser = Browser(headless=HEADLESS)

    agent = BrowserAgent(task=task, llm=llm, browser=browser)
    history = await agent.run()

    final = (history.final_result() or "").strip()
    successful = history.is_successful()  # True / False / None
    errors_text = " ".join(str(e) for e in (history.errors() or []) if e)
    try:
        sources = _clean_urls(history.urls())
    except Exception:
        sources = []

    # An LLM provider failure (quota/auth/rate limit) is a configuration problem, not a
    # bot wall — report it clearly instead of guessing the site blocked us.
    low_err = errors_text.lower()
    for em in _LLM_ERROR_MARKERS:
        if em in low_err:
            return final, "llm_error", f"the language model API failed ('{em}')", sources

    marker = _first_block_marker(final) or _first_block_marker(errors_text)
    if marker:
        return final, "blocked", f"bot-protection signal: '{marker}'", sources
    if successful is False:
        return final, "blocked", "agent reported the task was not completed successfully", sources
    if not final:
        return final, "blocked", "no content could be extracted", sources
    return final, "ok", "", sources


async def run_local(task: str, model: str):
    """Local headless browser path."""
    return await _run_browser(task, model, use_cloud=False)


def _num(v):
    """Coerce a possibly-None cost/balance value to float, or None."""
    try:
        return round(float(v), 4) if v is not None else None
    except (TypeError, ValueError):
        return None


async def _get_balance():
    """Fetch the remaining browser-use Cloud credit balance (USD), or None."""
    try:
        from browser_use_sdk.v3 import AsyncBrowserUse
        client = AsyncBrowserUse(api_key=BROWSER_USE_API_KEY)
        try:
            acct = await client.billing.account()
            return _num(getattr(acct, "total_credits_balance_usd", None))
        finally:
            try:
                await client.close()
            except Exception:
                pass
    except Exception:
        return None


async def run_cloud(task: str, reason: str, model: str) -> dict:
    """Run the SAME Claude-driven agent as the local path, but over a browser-use Cloud
    stealth browser (residential proxies + anti-fingerprinting). The reasoning is Claude's;
    the cloud only provides the anti-bot browser. Cost is derived from the credit-balance
    delta across the run."""
    if not CLOUD_AVAILABLE:
        return {"error": True, "via": "cloud",
                "message": ("This site appears to be bot-protected and the local browser was blocked, "
                            "but the browser-use Cloud fallback is not configured (no BROWSER_USE_API_KEY). "
                            f"Reason: {reason}")}

    # Tell the Web Agent (and thus the dashboard) that we are switching to the cloud NOW.
    emit_event({"kind": "cloud", "reason": reason})

    before = await _get_balance()
    try:
        text, status, detail, sources = await _run_browser(task, model, use_cloud=True)
    except Exception as e:
        return {"error": True, "via": "cloud", "message": f"Cloud browse failed: {e}"}
    after = await _get_balance()

    spent = round(max(0.0, before - after), 4) if (before is not None and after is not None) else None
    cost = {"total_cost_usd": spent}
    balance = {"total_credits_balance_usd": after}

    if status == "ok":
        return {"error": False, "via": "cloud", "message": text, "reason": reason,
                "sources": sources, "cost": cost, "balance": balance}
    if status == "llm_error":
        return {"error": True, "via": "cloud",
                "message": (f"Cloud research could not complete because {detail}. This is a model/API "
                            "configuration issue (e.g. quota or key), not a website problem."),
                "cost": cost, "balance": balance}
    # status == "blocked": even the cloud stealth browser could not finish the task
    return {"error": True, "via": "cloud",
            "message": (f"The cloud browser could not complete the task ({detail}). "
                        "The site may use protection beyond stealth/proxy, or the page lacked the info."),
            "sources": sources, "cost": cost, "balance": balance}


async def main() -> dict:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    spec = json.loads(raw)
    task = spec["task"]
    force_cloud = bool(spec.get("force_cloud", False))
    model = spec.get("model", "anthropic/claude-opus-5-5")

    if force_cloud:
        return await run_cloud(task, reason="caller requested the cloud browser", model=model)

    try:
        text, status, detail, sources = await run_local(task, model)
    except Exception as e:
        if CLOUD_FALLBACK and CLOUD_AVAILABLE:
            return await run_cloud(task, reason=f"local browser raised an error: {e}", model=model)
        return {"error": True, "via": "local", "message": f"Local browse failed: {e}"}

    if status == "ok":
        return {"error": False, "via": "local", "message": text, "sources": sources}

    # LLM provider problem — do not silently retry on the (paid) cloud; report it.
    if status == "llm_error":
        return {"error": True, "via": "local",
                "message": (f"Web research could not complete because {detail}. This is a model/API "
                            "configuration issue (e.g. quota or key), not a website problem. Check the "
                            "LLM API key/billing for the configured model.")}

    # status == "blocked"
    if CLOUD_FALLBACK:
        return await run_cloud(task, reason=detail, model=model)

    return {"error": True, "via": "local",
            "message": (f"The site appears to be bot-protected ({detail}) and the cloud fallback "
                        "is disabled. Enable WEB_AGENT_CLOUD_FALLBACK with a BROWSER_USE_API_KEY.")}


if __name__ == "__main__":
    try:
        emit(asyncio.run(main()))
    except Exception as e:
        emit({"error": True, "message": f"Browser runner crashed: {e}"})
