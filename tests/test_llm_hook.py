"""The LLM hook is tested with fakes only - no network, no API key."""

import pytest

from kb_assist.answer import KnowledgeAssistant
from kb_assist.kb import load_articles
from kb_assist.llm import HttpChatProvider, LLMConfigError, provider_from_env
from kb_assist.retrieval import BM25Index


class FakeProvider:
    name = "fake"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture(scope="module")
def index():
    return BM25Index(load_articles())


def test_disabled_by_default():
    assert provider_from_env({}) is None
    assert provider_from_env({"KB_ASSIST_LLM_ENABLED": "false", "KB_ASSIST_LLM_API_KEY": "x"}) is None


def test_enabled_requires_https_and_key():
    with pytest.raises(LLMConfigError, match="https"):
        provider_from_env({"KB_ASSIST_LLM_ENABLED": "true", "KB_ASSIST_LLM_ENDPOINT": "http://x",
                           "KB_ASSIST_LLM_API_KEY": "k"})
    with pytest.raises(LLMConfigError, match="API_KEY"):
        provider_from_env({"KB_ASSIST_LLM_ENABLED": "true",
                           "KB_ASSIST_LLM_ENDPOINT": "https://gateway.example/v1/chat/completions"})


def test_enabled_provider_hides_key_in_repr():
    fake_key = "placeholder-not-a-real-key"  # pragma: allowlist secret
    p = provider_from_env({"KB_ASSIST_LLM_ENABLED": "TRUE",
                           "KB_ASSIST_LLM_ENDPOINT": "https://gateway.example/v1/chat/completions",
                           "KB_ASSIST_LLM_API_KEY": fake_key, "KB_ASSIST_LLM_MODEL": "m"})
    assert isinstance(p, HttpChatProvider)
    assert fake_key not in repr(p)


def test_grounded_llm_answer_is_used_and_prompt_is_redacted(index):
    fake = FakeProvider("Switch to the SSL VPN profile on TCP 443 [1], then restart the router [1].")
    answer = KnowledgeAssistant(index, provider=fake).ask(
        "VPN error 809 from home, my card is 4111 1111 1111 1111")
    assert answer.mode == "llm"
    assert answer.text.startswith("Switch to the SSL VPN profile")
    system, prompt = fake.calls[0]
    assert "Answer ONLY from the numbered sources" in system
    assert "4111" not in prompt and "[REDACTED_CARD]" in prompt
    assert "KB0010002" in prompt or "KB0010003" in prompt


@pytest.mark.parametrize("reply", [
    "Just reinstall Windows.",                 # no citation
    "Reinstall the client [7].",               # cites a source that was not provided
    "NO_ANSWER",
])
def test_ungrounded_llm_answer_falls_back_to_extractive(index, reply):
    answer = KnowledgeAssistant(index, provider=FakeProvider(reply)).ask("VPN error 809")
    assert answer.mode == "extractive"
    assert "[1]" in answer.text


def test_provider_failure_falls_back(index):
    answer = KnowledgeAssistant(index, provider=FakeProvider(error=TimeoutError())).ask("VPN error 809")
    assert answer.mode == "extractive"


def test_provider_not_called_when_abstaining(index):
    fake = FakeProvider("anything [1]")
    answer = KnowledgeAssistant(index, provider=fake).ask("what is the capital of France?")
    assert answer.mode == "abstained"
    assert fake.calls == []
