"""Tests for companion/memory/summarizer.py (RollingSummarizer), the
`MemoryStore.list_turns`/`replace_turns` methods it relies on, and the
Orchestrator trigger. See docs/ROLLING_SUMMARIZER.md section 7.

The LLM is a fake (fixed summary text, records its input): these tests
prove the mechanism — which turns get replaced, ordering, wipe safety,
background scheduling — not summary quality. The stores are real, in
`tmp_path`, parametrized over both backends: Chroma with its real default
embedding (as tests/test_memory_store.py uses), MemArk with a small
deterministic embedding function (as tests/test_memark_retrieval.py does).
Each backend skips cleanly if its extra isn't installed.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from companion.core.interfaces import LLMBackend, SpeechToText, TextToSpeech, Turn
from companion.core.interfaces.types import Persona
from companion.core.orchestrator import Orchestrator
from companion.core.prompt_builder import PromptBuilder
from companion.memory.summarizer import RollingSummarizer

BASE = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


class _HashEmbedding:
    """Deterministic bag-of-words embedding: each word hashes to one of 64
    dimensions. Texts sharing words get higher cosine similarity."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * 64
            for word in text.lower().replace(".", " ").replace(":", " ").split():
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


def make_store(backend: str, persist_dir: str):
    if backend == "chroma":
        pytest.importorskip("chromadb")
        from companion.memory.store import VectorMemoryStore

        return VectorMemoryStore(persist_dir=persist_dir)
    pytest.importorskip("networkx")
    pytest.importorskip("faiss")
    from companion.memark.retrieval import MemArkMemoryStore

    return MemArkMemoryStore(persist_dir=persist_dir, embedding_function=_HashEmbedding())


@pytest.fixture(params=["chroma", "memark"])
def backend(request) -> str:
    return request.param


@pytest.fixture
def store(backend, tmp_path):
    return make_store(backend, str(tmp_path))


class _FakeLLM(LLMBackend):
    def __init__(self, reply: str = "The user likes hiking and has a sister named Maya.") -> None:
        self.reply = reply
        self.calls: list[tuple[str, list[Turn]]] = []
        # When set, generate_stream waits on it before answering — lets a
        # test act "while the LLM is running".
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def generate_stream(self, prompt: str, context: list[Turn]):
        self.calls.append((prompt, context))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        for word in self.reply.split(" "):
            yield word + " "


def _turn(i: int, text: str | None = None) -> Turn:
    role = "user" if i % 2 == 0 else "assistant"
    return Turn(role=role, text=text or f"message number {i}", timestamp=BASE + timedelta(minutes=i))


async def _write_turns(store, n: int, start: int = 0, user="u1", persona="p1") -> list[Turn]:
    turns = [_turn(i) for i in range(start, start + n)]
    for turn in turns:
        await store.write(user, persona, turn)
    return turns


# --- list_turns / replace_turns contract --------------------------------------


@pytest.mark.asyncio
async def test_list_turns_is_oldest_first_and_round_trips_exactly(store):
    turns = [_turn(i) for i in range(5)]
    for turn in reversed(turns):  # written newest first on purpose
        await store.write("u1", "p1", turn)

    assert await store.list_turns("u1", "p1") == turns


@pytest.mark.asyncio
async def test_list_turns_empty_scope_and_scoping(store):
    assert await store.list_turns("u1", "p1") == []
    await _write_turns(store, 3, persona="other")
    assert await store.list_turns("u1", "p1") == []


@pytest.mark.asyncio
async def test_replace_turns_replaces_exactly_the_given_turns(store):
    turns = await _write_turns(store, 4)
    summary = Turn(role="system", text="summary of the first two", timestamp=turns[1].timestamp)

    assert await store.replace_turns("u1", "p1", turns[:2], summary) is True
    assert await store.list_turns("u1", "p1") == [summary, *turns[2:]]


@pytest.mark.asyncio
async def test_replace_turns_is_all_or_nothing(store):
    turns = await _write_turns(store, 3)
    missing = _turn(99)
    summary = Turn(role="system", text="should never be stored", timestamp=turns[0].timestamp)

    assert await store.replace_turns("u1", "p1", [turns[0], missing], summary) is False
    assert await store.list_turns("u1", "p1") == turns


@pytest.mark.asyncio
async def test_replace_turns_after_wipe_writes_nothing(store):
    turns = await _write_turns(store, 3)
    await store.delete("u1", "p1")
    summary = Turn(role="system", text="resurrected memory", timestamp=turns[0].timestamp)

    assert await store.replace_turns("u1", "p1", turns, summary) is False
    assert await store.list_turns("u1", "p1") == []


@pytest.mark.asyncio
async def test_replaced_state_survives_a_fresh_store_instance(backend, tmp_path):
    first = make_store(backend, str(tmp_path))
    turns = await _write_turns(first, 4)
    summary = Turn(role="system", text="summary", timestamp=turns[1].timestamp)
    await first.replace_turns("u1", "p1", turns[:2], summary)

    second = make_store(backend, str(tmp_path))
    assert await second.list_turns("u1", "p1") == [summary, *turns[2:]]


# --- summarize_if_due -----------------------------------------------------------


@pytest.mark.asyncio
async def test_not_due_changes_nothing_and_skips_the_llm(store):
    turns = await _write_turns(store, 5)
    llm = _FakeLLM()

    ran = await RollingSummarizer(store, llm, trigger_turn_count=6).summarize_if_due("u1", "p1")

    assert ran is False
    assert llm.calls == []
    assert await store.list_turns("u1", "p1") == turns


@pytest.mark.asyncio
async def test_due_replaces_oldest_half_with_one_summary_turn(store):
    turns = await _write_turns(store, 6)
    llm = _FakeLLM("The user counted to six.")

    ran = await RollingSummarizer(store, llm, trigger_turn_count=6).summarize_if_due("u1", "p1")

    assert ran is True
    stored = await store.list_turns("u1", "p1")
    summary, rest = stored[0], stored[1:]
    # Half of 6 is 3, extended to 4 so the batch ends on a reply: two whole
    # exchanges summarized, the third exchange kept verbatim.
    assert rest == turns[4:]
    assert summary.role == "system"
    assert summary.timestamp == turns[3].timestamp  # newest replaced turn
    assert summary.text == "Summary of earlier conversation (2026-01-01): The user counted to six."

    # The LLM saw exactly the oldest four turns, in order.
    (_prompt, context), = llm.calls
    transcript = context[0].text
    assert [line.split(": ", 1)[1] for line in transcript.splitlines()] == [
        t.text for t in turns[:4]
    ]


@pytest.mark.asyncio
async def test_summaries_are_not_counted_or_resummarized(store):
    await _write_turns(store, 6)
    summarizer = RollingSummarizer(store, _FakeLLM(), trigger_turn_count=6)
    assert await summarizer.summarize_if_due("u1", "p1") is True  # 2 raw + 1 summary left

    await _write_turns(store, 3, start=6)  # 5 raw: not due, summary isn't counted
    assert await summarizer.summarize_if_due("u1", "p1") is False

    await _write_turns(store, 1, start=9)  # 6 raw: due again
    assert await summarizer.summarize_if_due("u1", "p1") is True

    stored = await store.list_turns("u1", "p1")
    assert [t.role for t in stored[:2]] == ["system", "system"]
    # Second batch: raw turns 4-7 (two whole exchanges); 8 and 9 kept.
    assert stored[2:] == [_turn(8), _turn(9)]
    # The second run's input held only raw turns, never the first summary.
    _prompt, context = summarizer.llm.calls[1]
    assert "Summary of earlier conversation" not in context[0].text


@pytest.mark.asyncio
async def test_empty_llm_output_replaces_nothing(store):
    turns = await _write_turns(store, 4)

    ran = await RollingSummarizer(store, _FakeLLM("   "), trigger_turn_count=4).summarize_if_due(
        "u1", "p1"
    )

    assert ran is False
    assert await store.list_turns("u1", "p1") == turns


@pytest.mark.asyncio
async def test_wipe_during_summarization_is_not_undone(store):
    await _write_turns(store, 4)
    llm = _FakeLLM()
    llm.gate = asyncio.Event()
    task = asyncio.create_task(
        RollingSummarizer(store, llm, trigger_turn_count=4).summarize_if_due("u1", "p1")
    )
    await llm.started.wait()

    await store.delete("u1", "p1")  # user wipes memory while the LLM runs
    llm.gate.set()

    assert await task is False
    assert await store.list_turns("u1", "p1") == []


@pytest.mark.asyncio
async def test_turn_written_during_summarization_is_kept(store):
    turns = await _write_turns(store, 4)
    llm = _FakeLLM()
    llm.gate = asyncio.Event()
    task = asyncio.create_task(
        RollingSummarizer(store, llm, trigger_turn_count=4).summarize_if_due("u1", "p1")
    )
    await llm.started.wait()

    late = _turn(10, "written while the summary was generating")
    await store.write("u1", "p1", late)
    llm.gate.set()

    assert await task is True
    stored = await store.list_turns("u1", "p1")
    assert stored[1:] == [*turns[2:], late]


@pytest.mark.asyncio
async def test_summary_is_retrievable_through_query(store):
    await _write_turns(store, 4)
    llm = _FakeLLM("The user has a sister named Maya who runs marathons.")
    await RollingSummarizer(store, llm, trigger_turn_count=4).summarize_if_due("u1", "p1")

    results = await store.query("u1", "p1", "sister Maya marathons", k=5)

    assert results[0].role == "system"
    assert "Maya" in results[0].text


def test_trigger_below_two_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        RollingSummarizer(Mock(), Mock(), trigger_turn_count=1)


# --- Orchestrator trigger (docs/ROLLING_SUMMARIZER.md section 5) ----------------


class _StubSummarizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.gate = asyncio.Event()
        self.fail = fail

    async def summarize_if_due(self, user_id: str, persona_id: str) -> bool:
        self.calls.append((user_id, persona_id))
        await self.gate.wait()
        if self.fail:
            raise RuntimeError("summarizer exploded")
        return True


class _ListMemory:
    """Minimal in-memory MemoryStore stand-in for orchestrator tests."""

    def __init__(self) -> None:
        self.turns: list[Turn] = []

    async def query(self, user_id, persona_id, query, k=5):
        return []

    async def write(self, user_id, persona_id, turn):
        self.turns.append(turn)


def _orchestrator(summarizer) -> Orchestrator:
    return Orchestrator(
        stt=Mock(spec=SpeechToText),
        tts=Mock(spec=TextToSpeech),
        llm=_FakeLLM("Hello there."),
        vad=Mock(),
        memory=_ListMemory(),
        prompt_builder=PromptBuilder(),
        summarizer=summarizer,
    )


def _persona() -> Persona:
    return Persona(id="mentor", name="Mentor", tone="Calm.", boundaries="None.",
                   voice_id="piper:x")


@pytest.mark.asyncio
async def test_text_turn_schedules_summarizer_without_waiting_for_it():
    summarizer = _StubSummarizer()  # blocks until gate is set
    orchestrator = _orchestrator(summarizer)

    reply = await asyncio.wait_for(
        orchestrator.handle_text_turn("u1", _persona(), [], "hi"), timeout=2
    )
    assert reply.strip() == "Hello there."
    await asyncio.sleep(0)  # let the background task start
    assert summarizer.calls == [("u1", "mentor")]
    assert len(orchestrator.memory.turns) == 2  # memory written before scheduling

    summarizer.gate.set()
    await orchestrator.wait_for_background_tasks()


@pytest.mark.asyncio
async def test_only_one_summarizer_task_per_scope_at_a_time():
    summarizer = _StubSummarizer()
    orchestrator = _orchestrator(summarizer)

    await orchestrator.handle_text_turn("u1", _persona(), [], "one")
    await asyncio.sleep(0)
    await orchestrator.handle_text_turn("u1", _persona(), [], "two")  # first still running
    await asyncio.sleep(0)
    assert summarizer.calls == [("u1", "mentor")]

    summarizer.gate.set()
    await orchestrator.wait_for_background_tasks()
    await orchestrator.handle_text_turn("u1", _persona(), [], "three")  # finished: runs again
    await orchestrator.wait_for_background_tasks()
    assert len(summarizer.calls) == 2


@pytest.mark.asyncio
async def test_summarizer_failure_is_logged_not_raised(caplog):
    summarizer = _StubSummarizer(fail=True)
    summarizer.gate.set()
    orchestrator = _orchestrator(summarizer)

    with caplog.at_level(logging.ERROR):
        reply = await orchestrator.handle_text_turn("u1", _persona(), [], "hi")
        await orchestrator.wait_for_background_tasks()

    assert reply.strip() == "Hello there."
    assert "RollingSummarizer failed" in caplog.text


@pytest.mark.asyncio
async def test_no_summarizer_means_no_background_tasks():
    orchestrator = _orchestrator(None)
    await orchestrator.handle_text_turn("u1", _persona(), [], "hi")
    assert orchestrator._summary_tasks == {}


# --- batches are whole exchanges (hardening after a garbled real summary) -----


def _aligned(trigger, roles):
    turns = [Turn(role=r, text=f"t{i}", timestamp=BASE + timedelta(minutes=i))
             for i, r in enumerate(roles)]
    batch = RollingSummarizer(object(), _FakeLLM(), trigger_turn_count=trigger)._whole_exchanges(turns)
    return [t.text for t in batch]


def test_batch_never_ends_with_a_question_cut_off_from_its_reply():
    """With trigger 6 the plain cut is 3 turns: user, reply, user. That left
    the third question's reply to start the *next* batch with its question
    missing, and a real summary of such a batch garbled a fact."""
    roles = ["user", "assistant"] * 4
    assert _aligned(6, roles) == ["t0", "t1", "t2", "t3"]
    assert _aligned(4, roles) == ["t0", "t1"]  # already whole: unchanged


def test_batch_extends_over_several_replies_until_the_next_user_message():
    roles = ["user", "assistant", "user", "assistant", "assistant", "user", "assistant"]
    assert _aligned(6, roles) == ["t0", "t1", "t2", "t3", "t4"]


def test_batch_falls_back_to_the_plain_cut_when_no_user_message_follows():
    roles = ["user", "assistant", "assistant", "assistant", "assistant", "assistant"]
    assert _aligned(6, roles) == ["t0", "t1", "t2"]
