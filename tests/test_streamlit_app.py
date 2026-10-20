"""Smoke test for companion/ui/streamlit_app.py via Streamlit's AppTest.

Runs the real script (real config resolution, real persona files) with
fake backends patched into `companion.app`'s `build_*` functions, which
the script imports at run time, then sends one chat message and checks
the round trip: the reply renders, and the exchange goes through
`Orchestrator.handle_text_turn` into memory. Skips without the
`ui-streamlit` extra.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from companion.core.interfaces import (
    LLMBackend,
    MemoryStore,
    SpeechToText,
    TextToSpeech,
    VoiceActivityDetector,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "companion" / "ui" / "streamlit_app.py")


class _FakeLLM(LLMBackend):
    async def generate_stream(self, prompt, context):
        yield "Hello from the "
        yield "fake model."


class _FakeMemory(MemoryStore):
    def __init__(self) -> None:
        self.written = []

    async def query(self, user_id, persona_id, query, k=5):
        return []

    async def write(self, user_id, persona_id, turn):
        self.written.append((user_id, persona_id, turn))

    async def delete(self, user_id, persona_id=None):
        pass

    async def list_turns(self, user_id, persona_id):
        return []

    async def replace_turns(self, user_id, persona_id, old, new):
        return False


@pytest.fixture
def fake_backends(monkeypatch):
    import streamlit as st

    from companion import app

    # The script caches backends with st.cache_resource, which outlives one
    # AppTest run in the same process.
    st.cache_resource.clear()
    memory = _FakeMemory()
    monkeypatch.setattr(app, "build_stt", lambda config: Mock(spec=SpeechToText))
    monkeypatch.setattr(app, "build_llm", lambda config: _FakeLLM())
    monkeypatch.setattr(app, "build_vad", lambda config: Mock(spec=VoiceActivityDetector))
    monkeypatch.setattr(app, "build_memory", lambda config: memory)
    monkeypatch.setattr(app, "build_tts_for_persona", lambda config, persona: Mock(spec=TextToSpeech))
    # No agent by default, whatever the config says, so these tests never
    # reach a real model; test_tool_use_is_shown installs a fake one.
    monkeypatch.setattr(app, "build_agent", lambda config, memory: None)
    return memory


class _FakeAgent:
    def handles(self, persona):
        return bool(persona.tools)

    async def run_stream(self, persona, user_id, built):
        from companion.core.agent import TextDelta, ToolFinished, ToolStarted

        yield ToolStarted("get_datetime")
        yield ToolFinished("get_datetime", ok=True)
        yield TextDelta("It's Monday.")


def test_chat_message_round_trip(fake_backends):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]

    at.chat_input[0].set_value("Hi there").run()

    assert not at.exception, at.exception
    rendered = [m.value for m in at.markdown]
    assert "Hi there" in rendered
    assert "Hello from the fake model." in rendered
    assert [(t.role, t.text) for _, _, t in fake_backends.written] == [
        ("user", "Hi there"),
        ("assistant", "Hello from the fake model."),
    ]


def test_clear_button_empties_the_chat_window(fake_backends):
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.chat_input[0].set_value("Hi there").run()

    at.sidebar.button[0].click().run()

    assert not at.exception, at.exception
    assert "Hello from the fake model." not in [m.value for m in at.markdown]


def test_tool_use_is_shown_for_a_persona_with_tools(fake_backends, monkeypatch):
    from companion import app

    monkeypatch.setattr(app, "build_agent", lambda config, memory: _FakeAgent())
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].set_value("planner").run()

    at.chat_input[0].set_value("What day is it?").run()

    assert not at.exception, at.exception
    assert any(c.value.startswith("Tools: ") for c in at.caption)
    assert "It's Monday." in [m.value for m in at.markdown]
    assert "🔧 get_datetime" in [m.value for m in at.markdown]
