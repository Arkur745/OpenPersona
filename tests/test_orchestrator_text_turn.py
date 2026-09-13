"""Tests for Orchestrator.handle_text_turn — real now that MemoryStore
exists (see docs/MEMORY_STORE.md). Request/response, unlike
handle_voice_turn's whole-session loop: the caller passes `recent_turns`
fresh each call, and this method queries memory, de-duplicates against
that caller-supplied history, builds the prompt, generates the full
reply, and persists both sides of the exchange.

All fakes — same reasoning as tests/test_orchestrator_voice_turn.py: the
turn logic's correctness doesn't depend on real inference or a real
vector DB. PromptBuilder is the one real component (cheap, pure, already
tested on its own).
"""

import logging

import pytest

from companion.core.interfaces import (
    LLMBackend,
    MemoryStore,
    MemoryStoreError,
    SpeechToText,
    TextToSpeech,
    Turn,
)
from companion.core.interfaces.types import Persona
from companion.core.orchestrator import Orchestrator
from companion.core.prompt_builder import PromptBuilder


class _FakeMemoryStore(MemoryStore):
    """query() returns every turn ever written for the queried
    (user_id, persona_id) scope, plus any `extra_results` configured up
    front — deliberately simulating a real MemoryStore that does NOT
    itself filter out turns already present in the live conversation
    (the exact scenario Orchestrator's dedup step exists to handle, see
    docs/MEMORY_STORE.md section 4), so a dedup regression would actually
    be caught here rather than tested around."""

    def __init__(self, extra_results: list[Turn] | None = None) -> None:
        self.extra_results = list(extra_results) if extra_results else []
        self.query_calls: list[tuple[str, str, str, int]] = []
        self.written: list[tuple[str, str, Turn]] = []

    async def query(self, user_id: str, persona_id: str, query: str, k: int = 5) -> list[Turn]:
        self.query_calls.append((user_id, persona_id, query, k))
        already_written = [
            turn for (uid, pid, turn) in self.written if uid == user_id and pid == persona_id
        ]
        return (self.extra_results + already_written)[:k]

    async def write(self, user_id: str, persona_id: str, turn: Turn) -> None:
        self.written.append((user_id, persona_id, turn))

    async def delete(self, user_id: str, persona_id: str | None = None) -> None:
        raise NotImplementedError


class _FakeLLM(LLMBackend):
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.received_calls: list[tuple[str, list[Turn]]] = []

    async def generate_stream(self, prompt: str, context: list[Turn]):
        self.received_calls.append((prompt, context))
        for token in self._tokens:
            yield token


def _make_orchestrator(
    *,
    llm: LLMBackend | None = None,
    memory: MemoryStore | None = None,
    prompt_builder: PromptBuilder | None = None,
):
    from unittest.mock import Mock

    return Orchestrator(
        stt=Mock(spec=SpeechToText),
        tts=Mock(spec=TextToSpeech),
        llm=llm or _FakeLLM(["Sure, ", "happy to help."]),
        vad=Mock(),
        memory=memory or _FakeMemoryStore(),
        prompt_builder=prompt_builder or PromptBuilder(),
    )


def _persona() -> Persona:
    return Persona(
        id="mentor",
        name="The Old Mentor",
        tone="Calm.",
        boundaries="Not a therapist.",
        voice_id="piper:en_US-mentor-medium",
    )


@pytest.mark.asyncio
async def test_full_text_turn_queries_memory_generates_and_persists_both_turns():
    memory = _FakeMemoryStore()
    llm = _FakeLLM(["Sure, ", "happy to help."])
    orchestrator = _make_orchestrator(llm=llm, memory=memory)

    reply = await orchestrator.handle_text_turn(
        user_id="u1", persona=_persona(), recent_turns=[], message="Can you help me?"
    )

    assert reply == "Sure, happy to help."
    assert memory.query_calls == [("u1", "mentor", "Can you help me?", 5)]  # k = max_memory_turns
    assert len(llm.received_calls) == 1
    _prompt, context = llm.received_calls[0]
    assert context[-1].role == "user" and context[-1].text == "Can you help me?"

    assert len(memory.written) == 2
    (uid1, pid1, user_turn), (uid2, pid2, assistant_turn) = memory.written
    assert (uid1, pid1) == ("u1", "mentor")
    assert (uid2, pid2) == ("u1", "mentor")
    assert user_turn.role == "user" and user_turn.text == "Can you help me?"
    assert assistant_turn.role == "assistant" and assistant_turn.text == "Sure, happy to help."


class _FailingWriteMemoryStore(_FakeMemoryStore):
    """query() behaves exactly like _FakeMemoryStore; write() always
    raises MemoryStoreError, simulating a real persistence-layer failure
    (e.g. MemArkPersistenceError, which subclasses this) — never a bug,
    never something else entirely."""

    async def write(self, user_id: str, persona_id: str, turn: Turn) -> None:
        raise MemoryStoreError("simulated disk failure")


@pytest.mark.asyncio
async def test_write_failure_does_not_affect_the_returned_reply(caplog):
    """The core requirement: a MemoryStoreError from write() must never
    reach the caller, and the reply the LLM already generated must come
    back completely unaffected -- the response the user gets should
    never depend on whether long-term persistence happened to work."""
    memory = _FailingWriteMemoryStore()
    llm = _FakeLLM(["Sure, ", "happy to help."])
    orchestrator = _make_orchestrator(llm=llm, memory=memory)

    with caplog.at_level(logging.ERROR):
        reply = await orchestrator.handle_text_turn(
            user_id="u1", persona=_persona(), recent_turns=[], message="Can you help me?"
        )

    assert reply == "Sure, happy to help."  # unaffected by the write failure


@pytest.mark.asyncio
async def test_write_failure_is_logged_loudly_not_swallowed_silently(caplog):
    """Per the section-8-style decision: end the turn (response still
    reaches the user), never silently swallow beyond a log line without
    even that -- confirms the failure actually gets a diagnosable ERROR
    log entry naming the scope and what was being written, not just
    "something went wrong"."""
    memory = _FailingWriteMemoryStore()
    orchestrator = _make_orchestrator(memory=memory)

    with caplog.at_level(logging.ERROR):
        await orchestrator.handle_text_turn(
            user_id="u1", persona=_persona(), recent_turns=[], message="Can you help me?"
        )

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 2  # one for the user turn, one for the assistant turn
    for record in error_records:
        message = record.getMessage()
        assert "u1" in message
        assert "mentor" in message
    assert any(r.exc_info is not None for r in error_records)  # traceback attached


@pytest.mark.asyncio
async def test_write_failure_does_not_end_the_session_next_turn_still_works():
    """No retry and no ending the session, per the chosen behavior: a
    write failure on one turn must not prevent the next turn (even
    against the same failing store) from generating and returning a
    reply normally."""
    memory = _FailingWriteMemoryStore()
    llm = _FakeLLM(["ok"])
    orchestrator = _make_orchestrator(llm=llm, memory=memory)

    first = await orchestrator.handle_text_turn(
        user_id="u1", persona=_persona(), recent_turns=[], message="first message"
    )
    second = await orchestrator.handle_text_turn(
        user_id="u1", persona=_persona(), recent_turns=[], message="second message"
    )

    assert first == "ok"
    assert second == "ok"


@pytest.mark.asyncio
async def test_retrieved_memory_is_deduplicated_against_recent_turns():
    """Pins the actual filtering effect on the prompt, not just presence
    somewhere — see docs/MEMORY_STORE.md section 4 and
    Orchestrator._dedup_retrieved_memory."""
    duplicate = Turn(role="user", text="I already said this earlier in the conversation")
    separate_memory = Turn(role="user", text="a genuinely separate old memory")
    memory = _FakeMemoryStore(extra_results=[duplicate, separate_memory])
    llm = _FakeLLM(["ok."])
    orchestrator = _make_orchestrator(llm=llm, memory=memory)

    await orchestrator.handle_text_turn(
        user_id="u1",
        persona=_persona(),
        recent_turns=[duplicate],  # already in this conversation's own history
        message="new message",
    )

    prompt, _context = llm.received_calls[0]
    assert "a genuinely separate old memory" in prompt
    assert "I already said this earlier in the conversation" not in prompt


@pytest.mark.asyncio
async def test_empty_reply_does_not_write_an_assistant_turn():
    memory = _FakeMemoryStore()
    llm = _FakeLLM([])  # LLM produced nothing
    orchestrator = _make_orchestrator(llm=llm, memory=memory)

    reply = await orchestrator.handle_text_turn(
        user_id="u1", persona=_persona(), recent_turns=[], message="hello"
    )

    assert reply == ""
    assert len(memory.written) == 1  # only the user turn, no empty assistant turn
    assert memory.written[0][2].role == "user"


@pytest.mark.asyncio
async def test_memory_query_uses_prompt_builders_max_memory_turns_as_k():
    memory = _FakeMemoryStore()
    orchestrator = _make_orchestrator(
        llm=_FakeLLM(["hi."]), memory=memory, prompt_builder=PromptBuilder(max_memory_turns=2)
    )

    await orchestrator.handle_text_turn(
        user_id="u1", persona=_persona(), recent_turns=[], message="hi"
    )

    assert memory.query_calls[0][3] == 2
