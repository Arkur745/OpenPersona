"""Tests for companion/core/prompt_builder/prompt_builder.py.

Fully offline, no optional dependencies, no MemoryStore mock — per
docs/PROMPT_BUILDER.md's Option A, PromptBuilder never touches MemoryStore
itself, so a plain list[Turn] literal stands in for whatever a real
MemoryStore.query() call would have returned.
"""

from datetime import datetime, timedelta, timezone

import pytest

from companion.core.interfaces.types import Persona, Turn
from companion.core.prompt_builder.prompt_builder import BuiltPrompt, PromptBuilder


def _persona(**overrides) -> Persona:
    defaults = dict(
        id="mentor",
        name="The Old Mentor",
        tone="Calm and plainspoken.",
        boundaries="Not a therapist.",
        voice_id="piper:en_US-mentor-medium",
        example_dialogue=[],
    )
    defaults.update(overrides)
    return Persona(**defaults)


def _turn(role: str, text: str, minutes_ago: int = 0) -> Turn:
    timestamp = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return Turn(role=role, text=text, timestamp=timestamp)


def test_build_returns_built_prompt_with_prompt_and_context():
    result = PromptBuilder().build(_persona(), [], [], "hello")

    assert isinstance(result, BuiltPrompt)
    assert isinstance(result.prompt, str)
    assert isinstance(result.context, list)


def test_prompt_includes_persona_name_tone_and_boundaries():
    persona = _persona(name="Casual Friend", tone="Warm and funny.", boundaries="No medical advice.")

    result = PromptBuilder().build(persona, [], [], "hi")

    assert "Casual Friend" in result.prompt
    assert "Warm and funny." in result.prompt
    assert "No medical advice." in result.prompt


def test_prompt_includes_example_dialogue_when_present():
    persona = _persona(example_dialogue=["User: hi\nPersona: hey there"])

    result = PromptBuilder().build(persona, [], [], "hi")

    assert "hey there" in result.prompt
    assert "Example interactions" in result.prompt


def test_prompt_omits_example_dialogue_section_when_empty():
    persona = _persona(example_dialogue=[])

    result = PromptBuilder().build(persona, [], [], "hi")

    assert "Example interactions" not in result.prompt


def test_prompt_omits_memory_section_when_no_memory_retrieved():
    result = PromptBuilder().build(_persona(), [], [], "hi")

    assert "Relevant memories" not in result.prompt


def test_prompt_includes_memory_section_when_memory_retrieved():
    memory = [_turn("user", "I mentioned feeling anxious about exams")]

    result = PromptBuilder().build(_persona(), [], memory, "hi")

    assert "Relevant memories" in result.prompt
    assert "feeling anxious about exams" in result.prompt


def test_memory_bounded_to_max_memory_turns_keeping_most_relevant_first():
    # Most-relevant-first per MemoryStore's contract — bounding must keep
    # the FIRST k, not the last k.
    memory = [_turn("user", f"memory-{i}") for i in range(5)]
    builder = PromptBuilder(max_memory_turns=2)

    result = builder.build(_persona(), [], memory, "hi")

    assert "memory-0" in result.prompt
    assert "memory-1" in result.prompt
    assert "memory-2" not in result.prompt
    assert "memory-4" not in result.prompt


def test_recent_turns_bounded_to_max_recent_turns_keeping_most_recent():
    # Chronological — bounding must keep the LAST k (most recent), in order.
    recent = [_turn("user", f"turn-{i}", minutes_ago=5 - i) for i in range(5)]
    builder = PromptBuilder(max_recent_turns=2)

    result = builder.build(_persona(), recent, [], "new message")

    texts = [t.text for t in result.context]
    assert texts == ["turn-3", "turn-4", "new message"]


def test_max_recent_turns_zero_yields_only_the_new_user_message():
    recent = [_turn("user", "old"), _turn("assistant", "reply")]
    builder = PromptBuilder(max_recent_turns=0)

    result = builder.build(_persona(), recent, [], "hi")

    assert len(result.context) == 1
    assert result.context[0].text == "hi"


def test_new_user_message_appended_as_final_context_turn_with_user_role():
    recent = [_turn("user", "earlier"), _turn("assistant", "earlier reply")]

    result = PromptBuilder().build(_persona(), recent, [], "what now?")

    assert result.context[-1].role == "user"
    assert result.context[-1].text == "what now?"
    assert [t.text for t in result.context] == ["earlier", "earlier reply", "what now?"]


def test_user_message_is_not_folded_into_prompt_string():
    result = PromptBuilder().build(_persona(), [], [], "this is a very specific new question")

    assert "this is a very specific new question" not in result.prompt


def test_recent_turns_not_folded_into_prompt_string():
    recent = [_turn("user", "a distinctly recognizable recent turn")]

    result = PromptBuilder().build(_persona(), recent, [], "hi")

    assert "a distinctly recognizable recent turn" not in result.prompt


def test_long_memory_text_truncated_with_marker():
    long_text = "x" * 5000
    builder = PromptBuilder(max_chars_per_turn=100)

    result = builder.build(_persona(), [], [_turn("user", long_text)], "hi")

    assert "[...truncated]" in result.prompt
    assert "x" * 5000 not in result.prompt


def test_long_recent_turn_text_truncated_in_context():
    long_text = "y" * 5000
    builder = PromptBuilder(max_chars_per_turn=100)

    result = builder.build(_persona(), [_turn("user", long_text)], [], "hi")

    truncated = result.context[0].text
    assert len(truncated) <= 100
    assert truncated.endswith("[...truncated]")


def test_short_turn_text_not_modified_by_truncation():
    builder = PromptBuilder(max_chars_per_turn=100)
    recent = [_turn("user", "short message")]

    result = builder.build(_persona(), recent, [], "hi")

    assert result.context[0].text == "short message"


@pytest.mark.parametrize("empty_message", ["", "   "])
def test_empty_user_message_still_produces_a_context_turn(empty_message):
    result = PromptBuilder().build(_persona(), [], [], empty_message)

    assert result.context[-1].role == "user"
    assert result.context[-1].text == ""
