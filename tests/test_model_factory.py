"""Provider quirks: a wrong parameter here is a 400 on every call an agent makes."""
import pytest

import model_factory as mf


@pytest.mark.parametrize("model", ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5",
                                   "anthropic/claude-haiku-5-5", "anthropic/claude-fable-5-1",
                                   "openai/gpt-6.1-terra", "openai/gpt-6.1-sol", "openai/gpt-6-luna",
                                   "openai/gpt-6-astra"])
def test_current_models_omit_temperature(model):
    assert "temperature" not in mf.completion_kwargs(model, temperature=0.0)


def test_gpt6_does_not_send_reasoning_effort():
    assert mf.llm_kwargs("openai/gpt-6.1-terra") == {}


def test_gpt5_keeps_its_chat_completions_setting():
    assert mf.llm_kwargs("openai/gpt-5.6-terra") == {"reasoning_effort": "none"}


@pytest.mark.parametrize("effort", ["none", "low"])
def test_gpt6_always_uses_the_responses_route(monkeypatch, effort):
    from responses_llm import ResponsesLLMClient
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", effort)
    assert isinstance(mf.build_llm("openai/gpt-6.1-terra").llm_client, ResponsesLLMClient)
