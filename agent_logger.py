# agent_logger.py
# Two-level conversation tracing for multi-agent A2A system.
# Levels: "standard" (operational) and "conversation" (narrative for demos).
# Designed as an event emitter with subscriber pattern for Phase 2 web UI integration.

import os
import time
import json
from datetime import datetime
from typing import Any, Callable, Optional
from dataclasses import dataclass, field, asdict
from shared_logger import SharedLogger

# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------

@dataclass
class AgentEvent:
    timestamp: str
    agent_name: str
    level: str          # "standard" or "conversation"
    event_type: str     # e.g. "agent_start", "llm_call", "tool_call", "tool_response", "llm_decision", "agent_end", "registry_change"
    message: str
    trace_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Subscriber protocol — any callable(AgentEvent) -> None
# ---------------------------------------------------------------------------

Subscriber = Callable[[AgentEvent], None]

# ---------------------------------------------------------------------------
# AgentLogger
# ---------------------------------------------------------------------------

class AgentLogger:
    """Singleton event emitter for agent conversation tracing."""

    _instance: Optional["AgentLogger"] = None
    _subscribers: list[Subscriber] = []
    _level: str = "standard"
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = []
            cls._instance._level = os.getenv("AGENT_LOG_LEVEL", "standard").lower()
            cls._instance._logger = SharedLogger.get_logger("agent_trace")
            # Auto-register console subscriber
            cls._instance.add_subscriber(cls._console_subscriber)
        return cls._instance

    @property
    def level(self) -> str:
        return self._level

    def add_subscriber(self, subscriber: Subscriber):
        if subscriber not in self._subscribers:
            self._subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: Subscriber):
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def emit(self, event: AgentEvent):
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception:
                pass  # Don't let a bad subscriber break the agent

    # --- Convenience methods ---

    def log(self, agent_name: str, event_type: str, message: str,
            trace_id: Optional[str] = None, metadata: Optional[dict] = None,
            level: str = "standard"):
        """Emit an event. All events are emitted to subscribers (for remote collection).
        Console subscriber filters based on AGENT_LOG_LEVEL."""
        event = AgentEvent(
            timestamp=datetime.now().isoformat(),
            agent_name=agent_name,
            level=level,
            event_type=event_type,
            message=message,
            trace_id=trace_id,
            metadata=metadata or {},
        )
        self.emit(event)

    def standard(self, agent_name: str, event_type: str, message: str,
                 trace_id: Optional[str] = None, metadata: Optional[dict] = None):
        """Always emitted regardless of level."""
        self.log(agent_name, event_type, message, trace_id, metadata, level="standard")

    def conversation(self, agent_name: str, event_type: str, message: str,
                     trace_id: Optional[str] = None, metadata: Optional[dict] = None):
        """Only emitted in conversation mode."""
        self.log(agent_name, event_type, message, trace_id, metadata, level="conversation")

    # --- Built-in subscribers ---

    @staticmethod
    def _console_subscriber(event: AgentEvent):
        """Print to console via SharedLogger. Respects AGENT_LOG_LEVEL for filtering."""
        # Filter: in standard mode, skip conversation-level entries on console
        al = AgentLogger()
        if event.level == "conversation" and al._level != "conversation":
            return
        logger = SharedLogger.get_logger("agent_trace")
        prefix = f"[{event.agent_name}]"
        if event.trace_id:
            prefix = f"[{event.agent_name}][{event.trace_id[:8]}]"
        logger.info(f"{prefix} {event.message}")

    def to_dict(self, event: AgentEvent) -> dict:
        """Serialize an event to dict (for Phase 2 WebSocket streaming)."""
        return asdict(event)


# ---------------------------------------------------------------------------
# Shared callback factory
# ---------------------------------------------------------------------------

def make_callbacks(agent_name: str, model_name: str = ""):
    """
    Returns a dict of ADK callbacks wired to the AgentLogger.
    Usage:
        cbs = make_callbacks("ChatAgent", ACTIVE_MODEL)
        agent = LlmAgent(..., **cbs)
    """
    al = AgentLogger()

    def before_agent_cb(callback_context, **kwargs):
        trace_id = getattr(callback_context, '_trace_id', None)
        al.standard(agent_name, "agent_start",
                     "Agent activated", trace_id)

    def after_agent_cb(callback_context, **kwargs):
        trace_id = getattr(callback_context, '_trace_id', None)
        al.standard(agent_name, "agent_end",
                     "Agent completed", trace_id)
        al.conversation(agent_name, "agent_end",
                        "✅ Agent finished processing", trace_id)

    def before_model_cb(callback_context, llm_request, **kwargs):
        name = agent_name
        trace_id = getattr(callback_context, '_trace_id', None)
        model = model_name
        # Try to get model from request
        if hasattr(llm_request, 'model') and llm_request.model:
            model = str(llm_request.model)

        al.standard(name, "llm_call", f"Calling LLM ({model})", trace_id)
        al.conversation(name, "llm_call",
                        f"💭 Asking LLM ({model})", trace_id)

        # In conversation mode, log prompt content
        if al.level == "conversation" and hasattr(llm_request, 'contents') and llm_request.contents:
            try:
                last_content = llm_request.contents[-1]
                if hasattr(last_content, 'parts') and last_content.parts:
                    for part in last_content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_preview = part.text[:200] + "..." if len(part.text) > 200 else part.text
                            al.conversation(name, "llm_prompt",
                                           f"   📝 Prompt: \"{text_preview}\"", trace_id)
                            break
            except Exception:
                pass
        return None

    def after_model_cb(callback_context, llm_response, **kwargs):
        name = agent_name
        trace_id = getattr(callback_context, '_trace_id', None)

        # Check for tool calls in the response
        if hasattr(llm_response, 'content') and llm_response.content:
            content = llm_response.content
            if hasattr(content, 'parts') and content.parts:
                for part in content.parts:
                    # Detect function calls
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        func_name = fc.name if hasattr(fc, 'name') else str(fc)
                        args = fc.args if hasattr(fc, 'args') else {}

                        al.standard(name, "llm_decision",
                                    f"LLM responded — tool call: {func_name}", trace_id)

                        # Conversation mode: narrate the decision
                        args_preview = ""
                        if args:
                            try:
                                args_str = json.dumps(dict(args), default=str)
                                args_preview = args_str[:150]
                            except Exception:
                                args_preview = str(args)[:150]
                        al.conversation(name, "llm_decision",
                                        f"🧠 LLM decided → calling {func_name}({args_preview})",
                                        trace_id)
                        return None

                    # Detect text response (final answer, no tool call)
                    if hasattr(part, 'text') and part.text:
                        al.standard(name, "llm_response",
                                    "LLM composing response", trace_id)
                        al.conversation(name, "llm_response",
                                        "🧠 LLM composing final response", trace_id)
                        return None

        # Log usage metadata if available
        if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
            usage = llm_response.usage_metadata
            if hasattr(usage, 'total_token_count') and usage.total_token_count:
                al.standard(name, "llm_usage",
                            f"Tokens used: {usage.total_token_count}", trace_id,
                            metadata={"tokens": usage.total_token_count})
        return None

    return {
        "before_agent_callback": before_agent_cb,
        "after_agent_callback": after_agent_cb,
        "before_model_callback": before_model_cb,
        "after_model_callback": after_model_cb,
    }


# ---------------------------------------------------------------------------
# Remote HTTP subscriber — forwards events to the web dashboard
# ---------------------------------------------------------------------------

def _make_http_subscriber(url: str) -> Subscriber:
    """Create a subscriber that POSTs events to a remote URL (fire-and-forget)."""
    import threading
    import requests as _requests

    def http_subscriber(event: AgentEvent):
        def _post():
            try:
                # CHARM 3.0: trace events are signed, so the audit trail can't be forged.
                import charm_guard
                name = charm_guard.current_identity_name()
                if not name or name == "Dashboard":
                    return  # no identity yet, or the dashboard itself (it records events in-process)
                body = charm_guard.json_body(asdict(event))
                headers = {"Content-Type": "application/json", **charm_guard.sign_request("Dashboard", body)}
                _requests.post(url, data=body, headers=headers, timeout=2, **charm_guard.requests_tls())
            except Exception:
                pass  # Fire and forget
        threading.Thread(target=_post, daemon=True).start()

    return http_subscriber


def _auto_register_remote_subscriber():
    """If TRACE_COLLECTOR_URL is set, auto-register an HTTP subscriber."""
    url = os.getenv("TRACE_COLLECTOR_URL")
    if url:
        al = AgentLogger()
        sub = _make_http_subscriber(url)
        al.add_subscriber(sub)

_auto_register_remote_subscriber()


# Module-level singleton for direct import
agent_logger = AgentLogger()
