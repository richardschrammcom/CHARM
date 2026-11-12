"""Run OpenAI reasoning models through /v1/responses so tools and reasoning coexist.

Why this exists
---------------
OpenAI's gpt-5.x models reason by default, but `/v1/chat/completions` — the only
endpoint ADK's `LiteLlm` wrapper calls — rejects function tools while a reasoning
effort is active:

    400  Function tools with reasoning_effort are not supported for gpt-5.6-terra
         in /v1/chat/completions. To use function tools, use /v1/responses or set
         reasoning_effort to 'none'.

Every Charm agent registers tools, so the default path (see `model_factory`) sends
`reasoning_effort="none"` and the models run without reasoning. `/v1/responses`
supports both together, and LiteLLM already speaks it (`litellm.aresponses`), but
ADK imports `acompletion` directly and never targets it.

`LiteLlm` accepts an injectable `llm_client`, which is the seam used here:
`ResponsesLLMClient.acompletion` keeps ADK's chat-completions signature, translates
the call to the Responses API, and translates the result back into the
`ModelResponse` shape ADK expects. Nothing in ADK has to change, and no dependency
needs upgrading.

Opt-in
------
Set `OPENAI_REASONING_EFFORT` to low/medium/high in `.env`. Unset (or "none")
leaves the existing behaviour exactly as it was.

Cost: reasoning is slow. A single high-effort call measured ~50s against ~1.6s
with reasoning off, and an agent turn makes several calls. Keep the timeout chain
(`WEB_AGENT_TIMEOUT` < `CHAT_REMOTE_TIMEOUT` < `DASHBOARD_CHAT_TIMEOUT`) in mind
before turning this on.
"""

import json
import os
from typing import Any, Dict, List, Optional

from google.adk.models.lite_llm import LiteLLMClient

VALID_EFFORTS = ("none", "low", "medium", "high")


def configured_effort() -> str:
    """Reasoning effort requested via the environment, normalised."""
    effort = os.getenv("OPENAI_REASONING_EFFORT", "none").strip().lower()
    return effort if effort in VALID_EFFORTS else "none"


def _text_of(content: Any) -> str:
    """Flatten a chat `content` field (str, or list of content parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                chunks.append(part.get("text") or part.get("content") or "")
        return "".join(chunks)
    return str(content)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from an object that may be a dict or a pydantic model."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def messages_to_input(messages: List[Any]) -> List[Dict[str, Any]]:
    """Convert chat-completions messages into Responses API input items.

    Tool round-trips differ between the two APIs: chat completions carries an
    assistant message with `tool_calls` plus a `tool` role reply, while the
    Responses API expects flat `function_call` / `function_call_output` items.
    """
    items: List[Dict[str, Any]] = []

    for msg in messages or []:
        role = _get(msg, "role")
        content = _get(msg, "content")

        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": _get(msg, "tool_call_id") or "",
                "output": _text_of(content),
            })
            continue

        if role == "assistant":
            for call in _get(msg, "tool_calls") or []:
                fn = _get(call, "function") or {}
                items.append({
                    "type": "function_call",
                    "call_id": _get(call, "id") or "",
                    "name": _get(fn, "name") or "",
                    "arguments": _get(fn, "arguments") or "{}",
                })
            text = _text_of(content)
            if text:
                items.append({"role": "assistant", "content": text})
            continue

        # system / developer / user all pass through as plain roles.
        items.append({
            "role": "system" if role == "developer" else (role or "user"),
            "content": _text_of(content),
        })

    return items


def tools_to_responses_format(tools: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    """Flatten chat-style tool definitions into the Responses API's flat shape."""
    if not tools:
        return None

    converted = []
    for tool in tools:
        fn = _get(tool, "function")
        if fn is None:
            converted.append(tool if isinstance(tool, dict) else dict(tool))
            continue
        converted.append({
            "type": "function",
            "name": _get(fn, "name") or "",
            "description": _get(fn, "description") or "",
            "parameters": _get(fn, "parameters") or {"type": "object", "properties": {}},
        })
    return converted


def responses_to_model_response(response: Any, model: str):
    """Rebuild a litellm ModelResponse from a Responses API result.

    ADK reads `choices[0].message` (content, tool_calls), `finish_reason`, `model`
    and `usage`, so those are the fields that must be populated faithfully. The
    reasoning summary is placed on `reasoning_content`, which ADK already surfaces
    as thought parts.
    """
    from litellm.types.utils import (
        ChatCompletionMessageToolCall,
        Choices,
        Function,
        Message,
        ModelResponse,
        Usage,
    )

    text_chunks: List[str] = []
    reasoning_chunks: List[str] = []
    tool_calls = []

    for item in _get(response, "output") or []:
        kind = _get(item, "type")

        if kind == "function_call":
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=_get(item, "call_id") or _get(item, "id") or "",
                    type="function",
                    function=Function(
                        name=_get(item, "name") or "",
                        arguments=_get(item, "arguments") or "{}",
                    ),
                )
            )
        elif kind == "reasoning":
            for summary in _get(item, "summary") or []:
                reasoning_chunks.append(_get(summary, "text") or "")
        elif kind == "message":
            for part in _get(item, "content") or []:
                text_chunks.append(_get(part, "text") or "")

    message = Message(
        role="assistant",
        content="".join(text_chunks) or None,
        tool_calls=tool_calls or None,
    )
    reasoning = "".join(reasoning_chunks)
    if reasoning:
        message.reasoning_content = reasoning

    usage_obj = _get(response, "usage")
    usage = None
    if usage_obj is not None:
        input_tokens = _get(usage_obj, "input_tokens", 0) or 0
        output_tokens = _get(usage_obj, "output_tokens", 0) or 0
        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=_get(usage_obj, "total_tokens", input_tokens + output_tokens) or 0,
        )
        details = _get(usage_obj, "output_tokens_details")
        reasoning_tokens = _get(details, "reasoning_tokens", 0) if details is not None else 0
        if reasoning_tokens:
            usage.completion_tokens_details = {"reasoning_tokens": reasoning_tokens}

    result = ModelResponse(
        choices=[Choices(
            message=message,
            finish_reason="tool_calls" if tool_calls else "stop",
            index=0,
        )],
        model=_get(response, "model") or model,
    )
    if usage is not None:
        result.usage = usage
    return result


class ResponsesLLMClient(LiteLLMClient):
    """LiteLLMClient that fulfils ADK's chat-completions calls via /v1/responses."""

    def __init__(self, effort: str = "medium"):
        super().__init__()
        self._effort = effort if effort in VALID_EFFORTS else "medium"
        self._summaries_allowed = True

    async def acompletion(self, model, messages, tools, **kwargs):
        import litellm

        # Chat-completions-only parameters the Responses API does not accept.
        for unsupported in ("reasoning_effort", "stream", "drop_params",
                            "temperature", "top_p", "n", "stop"):
            kwargs.pop(unsupported, None)

        # `summary` asks OpenAI to return a readable digest of the hidden reasoning,
        # which ADK surfaces as thought parts (the dashboard's "think out loud").
        # Some accounts are not permitted to request summaries, so a refusal falls
        # back to reasoning without them rather than failing the call.
        reasoning: Dict[str, Any] = {"effort": self._effort}
        if self._summaries_allowed:
            reasoning["summary"] = "auto"

        request: Dict[str, Any] = {
            "model": model,
            "input": messages_to_input(messages),
            "reasoning": reasoning,
        }
        converted_tools = tools_to_responses_format(tools)
        if converted_tools:
            request["tools"] = converted_tools
        request.update(kwargs)

        try:
            response = await litellm.aresponses(**request)
        except Exception as exc:
            if self._summaries_allowed and "summary" in str(exc).lower():
                # Account not cleared for reasoning summaries — retry without them
                # and stop asking for the rest of this process's lifetime.
                self._summaries_allowed = False
                request["reasoning"] = {"effort": self._effort}
                response = await litellm.aresponses(**request)
            else:
                raise

        return responses_to_model_response(response, model)
