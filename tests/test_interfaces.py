"""Interface-conformance smoke tests — not backend-specific, so these run
without any optional dependency installed. Prefer adding tests here over
tests that assume one specific backend, per docs/CONTRIBUTING.md."""

import pytest

from companion.core.interfaces import (
    LLMBackend,
    MemoryStore,
    SpeechToText,
    TextToSpeech,
    Turn,
    VoiceActivityDetector,
)
from companion.core.interfaces.types import Persona


@pytest.mark.parametrize(
    "interface",
    [SpeechToText, TextToSpeech, LLMBackend, VoiceActivityDetector, MemoryStore],
)
def test_interfaces_are_abstract(interface):
    with pytest.raises(TypeError):
        interface()


def test_turn_defaults_have_utc_timestamp():
    turn = Turn(role="user", text="hello")
    assert turn.timestamp.tzinfo is not None


def test_persona_round_trips_fields():
    persona = Persona(
        id="mentor",
        name="The Old Mentor",
        tone="calm",
        boundaries="not a therapist",
        voice_id="piper:en_US-mentor-medium",
        example_dialogue=["User: hi\nPersona: hey"],
    )
    assert persona.id == "mentor"
    assert len(persona.example_dialogue) == 1
