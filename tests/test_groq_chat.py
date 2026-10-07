"""GroqChat wiring, with a fake `groq` module: no network and no API key needed."""

import sys
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from legal_rag import generate
from legal_rag.generate import GroqChat


def _install_fake_groq(monkeypatch, content: str | None) -> dict:
    seen: dict = {}

    class Completions:
        def create(self, **kwargs):
            seen["request"] = kwargs
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            seen["api_key"] = api_key
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=FakeGroq))
    return seen


def test_groq_chat_sends_system_and_user_messages_deterministically(monkeypatch):
    seen = _install_fake_groq(monkeypatch, "جواب")
    monkeypatch.setattr(generate.settings, "groq_api_key", SecretStr("k-test"))
    assert GroqChat().complete("نظام", "سؤال") == "جواب"
    assert seen["api_key"] == "k-test"
    request = seen["request"]
    assert request["temperature"] == 0
    assert request["messages"] == [
        {"role": "system", "content": "نظام"},
        {"role": "user", "content": "سؤال"},
    ]


def test_groq_chat_turns_an_empty_completion_into_an_empty_string(monkeypatch):
    _install_fake_groq(monkeypatch, None)
    monkeypatch.setattr(generate.settings, "groq_api_key", SecretStr("k-test"))
    assert GroqChat().complete("s", "u") == ""


def test_groq_chat_fails_clearly_when_the_key_is_missing(monkeypatch):
    _install_fake_groq(monkeypatch, "x")
    monkeypatch.setattr(generate.settings, "groq_api_key", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqChat()
