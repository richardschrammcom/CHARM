"""Provider-aware LLM construction shared by all Charm agents.

Different providers need different request parameters for the same logical call.
Centralizing that here keeps every agent's model setup identical and means a new
provider quirk is fixed in one place instead of five.

Current quirk — OpenAI gpt-5.x reasoning models:
    These models reason by default. OpenAI's /v1/chat/completions endpoint (the one
    LiteLLM uses) rejects two things when a reasoning effort is active:
      1. function tools  -> 400 "Function tools with reasoning_effort are not
         supported ... use /v1/responses or set reasoning_effort to 'none'"
      2. temperature != 1 -> 400 "Unsupported value: 'temperature' does not
         support 0 with this model"
    Every Charm agent registers tools, and the Registry Agent scores with
    temperature=0.0, so both hit. Sending reasoning_effort="none" resolves both.

    Trade-off: this disables the models' reasoning mode. Restoring it would mean
    routing these models through OpenAI's /v1/responses endpoint, which ADK's
    LiteLlm wrapper does not currently target.

Current quirk — OpenAI gpt-6.x:
    Same 400s as gpt-5.x, but the installed LiteLLM refuses to send reasoning_effort
    for them, so "none" isn't available on chat completions. Calls with tools always
    use /v1/responses, and plain calls just omit temperature.
"""

import re
from typing import Any, Dict, Optional

# Anthropic's Claude 5 family (claude-opus-5, claude-sonnet-5, claude-fable-5, ...)
# rejects an explicit `temperature`: "`temperature` is deprecated for this model."
# Only the model default is accepted, so the parameter must be omitted entirely.
_CLAUDE_5 = re.compile(r"^anthropic/claude-\w+-5(?:$|-)")
_GPT_6 = re.compile(r"^openai/gpt-6(?:$|[.-])")


def llm_kwargs(model_str: str) -> Dict[str, Any]:
    """Extra litellm parameters required to call `model_str` successfully."""
    if model_str.startswith("openai/gpt-5"):
        return {"reasoning_effort": "none"}
    return {}


def supports_temperature(model_str: str) -> bool:
    """Whether `model_str` accepts an explicit temperature value."""
    return not (_CLAUDE_5.match(model_str) or _GPT_6.match(model_str))


def completion_kwargs(model_str: str, temperature: Optional[float] = None) -> Dict[str, Any]:
    """Build litellm call parameters for `model_str`, dropping any it rejects.

    `temperature` is requested, not guaranteed: models that only accept their
    default sampling temperature simply have it omitted rather than erroring.
    """
    kwargs = llm_kwargs(model_str)
    if temperature is not None and supports_temperature(model_str):
        kwargs["temperature"] = temperature
    return kwargs


def build_llm(model_str: str):
    """Build an ADK LiteLlm for `model_str` with any provider-specific parameters.

    OpenAI gpt-5.x models are routed through `/v1/responses` instead of chat
    completions when `OPENAI_REASONING_EFFORT` asks for actual reasoning, since
    that is the only endpoint accepting tools and reasoning together. Unset (or
    "none") keeps the default chat-completions path with reasoning disabled.
    """
    from google.adk.models.lite_llm import LiteLlm

    if model_str.startswith("openai/gpt-5") or _GPT_6.match(model_str):
        from responses_llm import ResponsesLLMClient, configured_effort

        effort = configured_effort()
        if effort != "none" or _GPT_6.match(model_str):
            return LiteLlm(model=model_str, llm_client=ResponsesLLMClient(effort))

    return LiteLlm(model=model_str, **llm_kwargs(model_str))
