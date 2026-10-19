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


@pytest.fixture
def fake_backends(monkeypatch):
    import companion.app as app

    memory = _FakeMemory()
    monkeypatch.setattr(app, "build_stt", lambda config: Mock(spec=SpeechToText))
    monkeypatch.setattr(app, "build_llm", lambda config: _FakeLLM())
    monkeypatch.setattr(app, "build_vad", lambda config: Mock(spec=VoiceActivityDetector))
    monkeypatch.setattr(app, "build_memory", lambda config: memory)
    monkeypatch.setattr(app, "build_tts_for_persona", lambda config, persona: Mock(spec=TextToSpeech))
    return memory


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
