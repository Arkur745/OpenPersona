"""Tests for Orchestrator.handle_voice_turn's state machine: listening ->
user speaking -> processing -> speaking response -> [interrupted |
finished] -> listening. See docs/ARCHITECTURE.md's "Voice-turn state
machine" section for the design.

All fakes here — deliberately not real backends, per the task: the state
machine's correctness doesn't depend on real inference, only on each
interface being called correctly and reacted to correctly. PromptBuilder
is the one real component (it's cheap, pure, and already tested on its
own in tests/test_prompt_builder.py) — using the real one here also
exercises the actual seam handle_voice_turn uses it through.

Mic-audio fixtures are self-describing: a "speech" frame is
FRAME_BYTES of 0x01, a "silence" frame is FRAME_BYTES of 0x00, and
_FakeVAD classifies purely by content — no separate scripted-by-call-index
list to keep in sync with how many frames a test's mic fixture happens to
produce.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest

from companion.core.interfaces import (
    LLMBackend,
    MemoryStore,
    MemoryStoreError,
    SpeechToText,
    TextToSpeech,
    Turn,
    VoiceActivityDetector,
)
from companion.core.interfaces.types import Persona
from companion.core.orchestrator import Orchestrator
from companion.core.prompt_builder import PromptBuilder

FRAME_BYTES = 4
SPEECH_FRAME = b"\x01" * FRAME_BYTES
SILENCE_FRAME = b"\x00" * FRAME_BYTES


class _FakeVAD(VoiceActivityDetector):
    frame_bytes = FRAME_BYTES

    def __init__(self) -> None:
        self.calls = 0

    def is_speech(self, audio_chunk: bytes) -> bool:
        self.calls += 1
        return audio_chunk == SPEECH_FRAME


class _FakeMemoryStore(MemoryStore):
    """query() returns every turn ever written for the queried
    (user_id, persona_id) scope, plus any `extra_results` configured up
    front — deliberately simulating a real MemoryStore that does NOT
    itself filter out turns already present in the live session's own
    recent_turns (the exact scenario Orchestrator's dedup step exists to
    handle, see docs/MEMORY_STORE.md section 4), so a dedup regression
    would actually be caught here rather than tested around."""

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

    async def list_turns(self, user_id: str, persona_id: str) -> list[Turn]:
        return []

    async def replace_turns(self, user_id, persona_id, old, new) -> bool:
        return False


class _FakeSTT(SpeechToText):
    """Always reports the same fixed transcript once any audio has been
    seen — the state machine's transcript handling doesn't depend on real
    incremental STT behavior (that's tested separately)."""

    def __init__(self, transcript: str = "hello there") -> None:
        self._transcript = transcript
        self.reset_calls = 0
        self.received_chunks: list[bytes] = []

    async def transcribe_stream(self, audio_chunk: bytes) -> str | None:
        self.received_chunks.append(audio_chunk)
        return self._transcript

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeLLM(LLMBackend):
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.received_calls: list[tuple[str, list[Turn]]] = []

    async def generate_stream(self, prompt: str, context: list[Turn]):
        self.received_calls.append((prompt, context))
        for token in self._tokens:
            yield token


class _FakeTTS(TextToSpeech):
    """Yields a few chunks per sentence with a real `await` between each
    (so the concurrent barge-in watcher gets scheduled), honoring stop()
    the same way the shared _sync_stream.py bridge does for real backends:
    stop mid-synthesis actually cuts output short, not just gets ignored."""

    def __init__(self, chunks_per_sentence: int = 3) -> None:
        self.chunks_per_sentence = chunks_per_sentence
        self.synthesized_sentences: list[str] = []
        self.stop_calls = 0
        self._stop_event = asyncio.Event()

    async def synthesize_stream(self, text_chunk: str) -> AsyncIterator[bytes]:
        self.synthesized_sentences.append(text_chunk)
        self._stop_event.clear()
        for i in range(self.chunks_per_sentence):
            if self._stop_event.is_set():
                return
            await asyncio.sleep(0)
            if self._stop_event.is_set():
                return
            yield f"{text_chunk}#{i}".encode()

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop_event.set()


async def _mic(
    frames: list[bytes], trailing_silence: int = 0, frame_interval: float = 0.0
) -> AsyncIterator[bytes]:
    """`frame_interval` paces frames with a real (tiny) sleep instead of
    `asyncio.sleep(0)` — needed whenever a test cares about the *order* of
    events relative to the concurrent barge-in watcher (e.g. "utterance 2
    starts only after utterance 1's response finishes"), since fakes have
    no real synthesis/inference latency and asyncio.sleep(0) only yields a
    scheduling turn, not real elapsed time — the watcher can otherwise
    race arbitrarily far ahead of a very short fake response."""
    for frame in frames:
        yield frame
        await asyncio.sleep(frame_interval)
    for _ in range(trailing_silence):
        yield SILENCE_FRAME
        await asyncio.sleep(frame_interval)


def _make_orchestrator(
    *,
    stt: SpeechToText | None = None,
    tts: TextToSpeech | None = None,
    llm: LLMBackend | None = None,
    vad: VoiceActivityDetector | None = None,
    memory: MemoryStore | None = None,
) -> Orchestrator:
    return Orchestrator(
        stt=stt or _FakeSTT(),
        tts=tts or _FakeTTS(),
        llm=llm or _FakeLLM(["Hi. ", "How are you? "]),
        vad=vad or _FakeVAD(),
        memory=memory or _FakeMemoryStore(),
        prompt_builder=PromptBuilder(),
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
async def test_missing_frame_bytes_fails_loudly():
    from unittest.mock import Mock

    orchestrator = _make_orchestrator(vad=Mock(spec=VoiceActivityDetector))

    with pytest.raises(RuntimeError, match="frame_bytes"):
        async for _ in orchestrator.handle_voice_turn(
            "u1", _persona(), _mic([SPEECH_FRAME])
        ):
            pass


@pytest.mark.asyncio
async def test_empty_mic_stream_ends_without_yielding_anything():
    orchestrator = _make_orchestrator()

    chunks = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), _mic([]))]

    assert chunks == []


@pytest.mark.asyncio
async def test_silence_only_never_reaches_processing():
    stt = _FakeSTT()
    llm = _FakeLLM(["should not be reached"])
    orchestrator = _make_orchestrator(stt=stt, llm=llm)
    mic = _mic([SILENCE_FRAME] * 5)  # never speaks -> never a full utterance

    chunks = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert chunks == []
    assert llm.received_calls == []


@pytest.mark.asyncio
async def test_full_turn_runs_stt_through_tts_end_to_end():
    """The actual end-to-end assertion the task asked to be explicit
    about: this exercises transcribe -> build prompt -> generate -> speak
    through Orchestrator.handle_voice_turn itself, not by calling the
    pieces separately."""
    stt = _FakeSTT(transcript="what is the weather")
    llm = _FakeLLM(["It's sunny. ", "Enjoy the day."])
    tts = _FakeTTS(chunks_per_sentence=2)
    orchestrator = _make_orchestrator(stt=stt, llm=llm, tts=tts)

    # speak, then enough trailing silence to (a) end the utterance and
    # (b) outlast the fake response before the mic stream itself ends.
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=40)

    chunks = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert stt.reset_calls >= 1
    assert len(llm.received_calls) == 1
    prompt, context = llm.received_calls[0]
    assert "what is the weather" in "".join(
        t.text for t in context
    )  # the transcript reached PromptBuilder as the new user turn
    assert tts.synthesized_sentences == ["It's sunny.", "Enjoy the day."]
    assert chunks == [
        b"It's sunny.#0",
        b"It's sunny.#1",
        b"Enjoy the day.#0",
        b"Enjoy the day.#1",
    ]
    assert tts.stop_calls == 0  # finished normally, never interrupted


@pytest.mark.asyncio
async def test_barge_in_stops_tts_and_calls_stop():
    stt = _FakeSTT()
    llm = _FakeLLM(["This is a long reply. ", "With multiple sentences. ", "Really."])
    tts = _FakeTTS(chunks_per_sentence=5)
    orchestrator = _make_orchestrator(stt=stt, llm=llm, tts=tts)

    mic = _mic(
        [
            SPEECH_FRAME,
            SILENCE_FRAME,  # end of utterance -> processing/speaking begins
            SILENCE_FRAME,
            SILENCE_FRAME,
            SPEECH_FRAME,  # barge-in
        ],
        trailing_silence=0,  # mic ends right after -> whole turn ends, no hang
    )

    chunks = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert tts.stop_calls >= 1
    # interrupted before the full 3-sentence, 5-chunks-each reply finished
    assert len(chunks) < 15


@pytest.mark.asyncio
async def test_first_utterance_in_a_fresh_session_has_no_history_or_memory_yet():
    """A session's first utterance genuinely has nothing to recall yet —
    empty recent_turns (session just started) and empty retrieved_memory
    (default _FakeMemoryStore has nothing written for this user/persona
    yet). Not stubbing anymore (see docs/MEMORY_STORE.md) — this pins
    real first-turn behavior, distinct from later turns which do carry
    history (see the dedup/history tests below)."""
    stt = _FakeSTT(transcript="test message")
    llm = _FakeLLM(["ok."])
    memory = _FakeMemoryStore()
    orchestrator = _make_orchestrator(stt=stt, llm=llm, memory=memory)
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=10)

    _ = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert len(llm.received_calls) == 1
    prompt, context = llm.received_calls[0]
    assert "Relevant memories" not in prompt
    assert len(context) == 1
    assert context[0].role == "user"
    assert context[0].text == "test message"
    # the query that ran to check for memory used the transcript, scoped
    # to this user+persona, with k = prompt_builder.max_memory_turns
    assert memory.query_calls == [("u1", "mentor", "test message", 5)]


@pytest.mark.asyncio
async def test_turns_are_written_to_memory_for_user_and_assistant():
    memory = _FakeMemoryStore()
    stt = _FakeSTT(transcript="hello there")
    llm = _FakeLLM(["Hi. ", "Good to hear from you."])
    orchestrator = _make_orchestrator(stt=stt, llm=llm, memory=memory)
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=20)

    _ = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert len(memory.written) == 2
    (uid1, pid1, user_turn), (uid2, pid2, assistant_turn) = memory.written
    assert (uid1, pid1) == ("u1", "mentor")
    assert (uid2, pid2) == ("u1", "mentor")
    assert user_turn.role == "user" and user_turn.text == "hello there"
    assert assistant_turn.role == "assistant"
    assert assistant_turn.text == "Hi. Good to hear from you."


@pytest.mark.asyncio
async def test_retrieved_memory_excludes_turns_already_in_recent_history():
    """A second utterance's memory query naturally returns the first
    utterance's own turns too (per _FakeMemoryStore's docstring) — pins
    that Orchestrator._dedup_retrieved_memory actually strips those back
    out before they reach PromptBuilder, while a genuinely separate past
    memory still comes through. Asserts the effect on the prompt sent to
    the LLM, not just that dedup runs."""
    memory = _FakeMemoryStore(
        extra_results=[Turn(role="user", text="a genuinely separate old memory")]
    )
    stt = _FakeSTT(transcript="second question")
    llm = _FakeLLM(["short reply."])
    orchestrator = _make_orchestrator(stt=stt, llm=llm, memory=memory)

    mic = _mic(
        [SPEECH_FRAME, SILENCE_FRAME]
        + [SILENCE_FRAME] * 10  # let the first response finish
        + [SPEECH_FRAME, SILENCE_FRAME],
        trailing_silence=10,
        frame_interval=0.002,
    )

    _ = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert len(llm.received_calls) == 2
    second_prompt, _second_context = llm.received_calls[1]
    assert "a genuinely separate old memory" in second_prompt
    assert "second question" not in second_prompt  # deduped: already in recent_turns
    assert "short reply." not in second_prompt  # deduped: already in recent_turns


@pytest.mark.asyncio
async def test_second_utterance_after_finishing_first_is_handled_too():
    """Confirms the [finished -> listening] loop-back actually loops:
    a second utterance after the first response completes is processed as
    its own turn, not dropped."""
    stt = _FakeSTT(transcript="second question")
    llm = _FakeLLM(["short reply."])
    tts = _FakeTTS(chunks_per_sentence=1)
    orchestrator = _make_orchestrator(stt=stt, llm=llm, tts=tts)

    mic = _mic(
        [
            SPEECH_FRAME,
            SILENCE_FRAME,  # first utterance ends
        ]
        + [SILENCE_FRAME] * 10  # let the first response finish
        + [
            SPEECH_FRAME,
            SILENCE_FRAME,  # second utterance ends
        ],
        trailing_silence=10,
        # paced so the (near-instant) fake response reliably finishes
        # before the second SPEECH_FRAME arrives — see _mic's docstring.
        frame_interval=0.002,
    )

    _ = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert len(llm.received_calls) == 2
    assert stt.reset_calls >= 2  # STT reset at the start of each listening phase


# --- Real-playback-latency barge-in fix: on_playback_interrupt /
# wait_for_playback_drained. See companion/audio/playback.py and
# orchestrator.py's handle_voice_turn docstring for the design this
# closes the gap in.


@pytest.mark.asyncio
async def test_wait_for_playback_drained_is_awaited_after_normal_completion():
    drain_calls = []

    async def fake_wait_drained():
        drain_calls.append(True)

    orchestrator = _make_orchestrator()
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=10)

    _ = [
        c
        async for c in orchestrator.handle_voice_turn(
            "u1", _persona(), mic, wait_for_playback_drained=fake_wait_drained
        )
    ]

    assert drain_calls == [True]


@pytest.mark.asyncio
async def test_wait_for_playback_drained_skipped_when_already_interrupted():
    """If barge-in already happened mid-response, there's nothing left to
    usefully wait for — asserts the fix doesn't add a pointless wait on
    the already-interrupted path."""
    drain_calls = []

    async def fake_wait_drained():
        drain_calls.append(True)

    tts = _FakeTTS(chunks_per_sentence=5)
    llm = _FakeLLM(["This is a long reply. ", "With multiple sentences. ", "Really."])
    orchestrator = _make_orchestrator(llm=llm, tts=tts)
    mic = _mic(
        [SPEECH_FRAME, SILENCE_FRAME, SILENCE_FRAME, SILENCE_FRAME, SPEECH_FRAME],
        trailing_silence=0,
    )

    _ = [
        c
        async for c in orchestrator.handle_voice_turn(
            "u1", _persona(), mic, wait_for_playback_drained=fake_wait_drained
        )
    ]

    assert drain_calls == []


@pytest.mark.asyncio
async def test_on_playback_interrupt_called_on_barge_in():
    interrupt_calls = []
    tts = _FakeTTS(chunks_per_sentence=5)
    llm = _FakeLLM(["This is a long reply. ", "With multiple sentences. ", "Really."])
    orchestrator = _make_orchestrator(llm=llm, tts=tts)
    mic = _mic(
        [SPEECH_FRAME, SILENCE_FRAME, SILENCE_FRAME, SILENCE_FRAME, SPEECH_FRAME],
        trailing_silence=0,
    )

    _ = [
        c
        async for c in orchestrator.handle_voice_turn(
            "u1", _persona(), mic, on_playback_interrupt=lambda: interrupt_calls.append(True)
        )
    ]

    assert interrupt_calls == [True]


@pytest.mark.asyncio
async def test_late_barge_in_during_playback_drain_is_still_caught():
    """The actual regression test for the fix: a barge-in that arrives
    *after* the response has fully finished synthesizing/yielding (while
    handle_voice_turn is awaiting wait_for_playback_drained, simulating
    real playback still sounding out the tail of the response) must
    still register — proving the barge-in watcher stays alive through the
    wait, not just through synthesis. Against the pre-fix code (no
    wait_for_playback_drained support), the watcher would already have
    been cancelled by the time this late frame arrives, and this
    assertion would fail."""
    interrupt_calls = []

    async def slow_wait_drained():
        # Simulates real playback still running for a bit after synthesis
        # finished — long enough for the "late" SPEECH_FRAME below to
        # arrive and be processed by the still-alive watcher.
        await asyncio.sleep(0.08)

    stt = _FakeSTT()
    llm = _FakeLLM(["short reply."])
    orchestrator = _make_orchestrator(stt=stt, llm=llm)

    mic = _mic(
        [SPEECH_FRAME, SILENCE_FRAME]  # utterance ends, response starts
        + [SILENCE_FRAME] * 2  # a little gap
        + [SPEECH_FRAME]  # "late" barge-in, arrives during the drain-wait
        + [SILENCE_FRAME] * 2,
        frame_interval=0.01,
    )

    _ = [
        c
        async for c in orchestrator.handle_voice_turn(
            "u1",
            _persona(),
            mic,
            on_playback_interrupt=lambda: interrupt_calls.append(True),
            wait_for_playback_drained=slow_wait_drained,
        )
    ]

    assert interrupt_calls == [True]


@pytest.mark.asyncio
async def test_mic_ending_naturally_after_response_does_not_hang():
    """Reproduces a real deadlock found via a real (non-fake) round trip
    with continuous, unscripted mic audio. Every fixture above either ends
    right inside the listening loop (empty/silence-only mics) or ends with
    a final real barge-in SPEECH_FRAME (so the watcher exits via the
    barge-in branch, leaving stream_ended to be read later by the next
    listening iteration). This one ends with silence-then-nothing after an
    utterance instead: the utterance-ending debounce breaks the listening
    loop while several trailing-silence events (and eventually
    stream_ended) are still queued, so it's the barge-in watcher — not the
    listening loop — that reads stream_ended.

    Before the fix, the watcher consumed that one-shot signal for itself
    (setting `interrupted` and returning), and the *next* listening
    iteration hung forever on a queue nothing would ever fill again — an
    unbounded hang, not a slow one. Wrapped in asyncio.wait_for so a
    regression fails fast with a clear timeout instead of hanging the
    whole test suite.

    Timing note: this must reliably put the watcher, not the listening
    loop, in the position of reading stream_ended — which requires
    _pump_mic to have already fully drained the mic (including
    stream_ended) into the queue *before* the response finishes, so the
    watcher's `events.get()` calls find it all already sitting there
    (asyncio.Queue.get()/put() don't actually suspend when the queue
    already has items / isn't full, so once that's true the watcher
    drains straight through to stream_ended in one uninterruptible
    synchronous stretch). A real per-mic-frame `frame_interval` would work
    against that: it introduces a real timer delay on every _pump_mic
    step while every other fake here is instant, so the (near-instant)
    fake response — and the `finally: watcher.cancel()` after it — would
    very likely complete first, cancelling the watcher before it ever
    reads anything and silently defeating the whole test (confirmed
    empirically: an earlier version of this test used frame_interval and
    passed even with the bug still present). So the mic uses the default
    frame_interval=0 (fastest possible) and, instead, the fake LLM here
    sleeps briefly before its first token — a real, generous head start
    for _pump_mic, not a race against instant fakes."""

    class _SlowLLM(LLMBackend):
        async def generate_stream(self, prompt: str, context: list[Turn]):
            await asyncio.sleep(0.05)
            yield "short reply."

    stt = _FakeSTT()
    orchestrator = _make_orchestrator(stt=stt, llm=_SlowLLM())

    mic = _mic(
        [SPEECH_FRAME, SILENCE_FRAME, SILENCE_FRAME, SILENCE_FRAME],
        trailing_silence=5,
    )

    async def _collect() -> list[bytes]:
        return [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    chunks = await asyncio.wait_for(_collect(), timeout=5.0)

    assert chunks  # the response completed normally, not cut short to nothing


# --- write() failure handling: end the turn, never end the session or
# swallow silently. See Orchestrator._write_turn_safely's own docstring
# for the full reasoning; these tests confirm it holds for the voice-turn
# state machine specifically, which has two separate write() call sites
# (user turn before the response, assistant turn after it) unlike
# handle_text_turn's two calls back-to-back.


class _FailingWriteMemoryStore(_FakeMemoryStore):
    """query() behaves exactly like _FakeMemoryStore; write() always
    raises MemoryStoreError, simulating a real persistence-layer failure
    (e.g. MemArkPersistenceError, which subclasses this)."""

    async def write(self, user_id: str, persona_id: str, turn: Turn) -> None:
        raise MemoryStoreError("simulated disk failure")


@pytest.mark.asyncio
async def test_write_failure_does_not_affect_response_audio_already_yielded(caplog):
    """The core requirement for voice mode specifically: the user has
    already heard (or would have already heard, chunk by chunk) the
    response by the time either write() call happens — a persistence
    failure must never retroactively affect what was already yielded."""
    memory = _FailingWriteMemoryStore()
    stt = _FakeSTT(transcript="what is the weather")
    llm = _FakeLLM(["It's sunny. ", "Enjoy the day."])
    tts = _FakeTTS(chunks_per_sentence=2)
    orchestrator = _make_orchestrator(stt=stt, llm=llm, tts=tts, memory=memory)
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=40)

    with caplog.at_level(logging.ERROR):
        chunks = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert chunks == [
        b"It's sunny.#0",
        b"It's sunny.#1",
        b"Enjoy the day.#0",
        b"Enjoy the day.#1",
    ]  # identical to the no-failure case — completely unaffected


@pytest.mark.asyncio
async def test_write_failure_is_logged_for_both_call_sites_not_swallowed(caplog):
    """Both the user-turn write (before the response) and the
    assistant-turn write (after it) must each be caught and logged
    individually — a failure on one must not prevent the other from at
    least being attempted and, on its own failure, also logged."""
    memory = _FailingWriteMemoryStore()
    stt = _FakeSTT(transcript="what is the weather")
    llm = _FakeLLM(["short reply."])
    orchestrator = _make_orchestrator(stt=stt, llm=llm, memory=memory)
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=40)

    with caplog.at_level(logging.ERROR):
        _ = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(error_records) == 2  # user turn + assistant turn
    for record in error_records:
        message = record.getMessage()
        assert "u1" in message
        assert "mentor" in message
    assert any(r.exc_info is not None for r in error_records)


@pytest.mark.asyncio
async def test_write_failure_does_not_end_the_session_second_utterance_still_works():
    """No retry and no ending the session: a write failure during the
    first utterance's turn must not prevent a second utterance, later in
    the same handle_voice_turn call, from being processed normally."""
    memory = _FailingWriteMemoryStore()
    stt = _FakeSTT(transcript="second question")
    llm = _FakeLLM(["short reply."])
    tts = _FakeTTS(chunks_per_sentence=1)
    orchestrator = _make_orchestrator(stt=stt, llm=llm, tts=tts, memory=memory)

    mic = _mic(
        [SPEECH_FRAME, SILENCE_FRAME]
        + [SILENCE_FRAME] * 10
        + [SPEECH_FRAME, SILENCE_FRAME],
        trailing_silence=10,
        frame_interval=0.002,
    )

    chunks = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]

    assert len(llm.received_calls) == 2  # both utterances were processed
    assert chunks  # both produced audio


# --- Agent path (personas with tools) -----------------------------------------
# Same fakes as above plus a scripted agent. The agent's stream is read by one
# background task inside _sentences_from_agent; these tests pin what the user
# hears (filler, then answer), what reaches memory (answer only), and that
# barge-in or a stalled agent never leaves the turn hanging.


class _FakeAgent:
    """Replays `steps`: AgentEvents are yielded, numbers are sleeps."""

    def __init__(self, steps) -> None:
        self.steps = steps
        self.cancelled = False
        self.finished = False

    def handles(self, persona: Persona) -> bool:
        return bool(persona.tools)

    async def run_stream(self, persona, user_id, built):
        try:
            for step in self.steps:
                if isinstance(step, (int, float)):
                    await asyncio.sleep(step)
                else:
                    yield step
            self.finished = True
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def _tool_persona() -> Persona:
    return Persona(id="planner", name="Planner", tone="Brisk.", boundaries="None.",
                   voice_id="piper:x", tools=["get_datetime"], filler_phrases=["One sec."])


def _agent_orchestrator(agent, *, llm=None, tts=None, memory=None) -> Orchestrator:
    orchestrator = _make_orchestrator(llm=llm, tts=tts, memory=memory)
    orchestrator.agent = agent
    return orchestrator


@pytest.mark.asyncio
async def test_agent_turn_speaks_filler_then_answer_and_remembers_only_the_answer():
    from companion.core.agent import TextDelta, ToolFinished, ToolStarted

    agent = _FakeAgent([ToolStarted("get_datetime"), 0.02, ToolFinished("get_datetime", True),
                        TextDelta("It's Monday. "), TextDelta("Anything else?")])
    tts = _FakeTTS()
    memory = _FakeMemoryStore()
    orchestrator = _agent_orchestrator(agent, tts=tts, memory=memory)
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=10, frame_interval=0.01)

    _ = [c async for c in orchestrator.handle_voice_turn("u1", _tool_persona(), mic)]

    assert tts.synthesized_sentences == ["One sec.", "It's Monday.", "Anything else?"]
    assert [(t.role, t.text) for _, _, t in memory.written] == [
        ("user", "hello there"), ("assistant", "It's Monday. Anything else?"),
    ]


@pytest.mark.asyncio
async def test_no_filler_when_the_agent_answers_without_tools():
    from companion.core.agent import TextDelta

    tts = _FakeTTS()
    orchestrator = _agent_orchestrator(_FakeAgent([TextDelta("Hello! Nice to see you.")]), tts=tts)
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=10, frame_interval=0.01)

    _ = [c async for c in orchestrator.handle_voice_turn("u1", _tool_persona(), mic)]

    assert tts.synthesized_sentences == ["Hello!", "Nice to see you."]


@pytest.mark.asyncio
async def test_barge_in_while_waiting_on_a_tool_cancels_the_agent():
    """The silent gap after the filler: the consumer is waiting on the
    agent, not on TTS. Barge-in there must end the turn at once and cancel
    the agent, so its result can't arrive after the user moved on."""
    from companion.core.agent import TextDelta, ToolStarted

    agent = _FakeAgent([ToolStarted("get_datetime"), 30, TextDelta("Too late.")])
    tts = _FakeTTS()
    memory = _FakeMemoryStore()
    orchestrator = _agent_orchestrator(agent, tts=tts, memory=memory)
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME] + [SILENCE_FRAME] * 5 + [SPEECH_FRAME],
               frame_interval=0.01)

    _ = await asyncio.wait_for(
        _collect(orchestrator.handle_voice_turn("u1", _tool_persona(), mic)), timeout=5
    )

    assert agent.cancelled and not agent.finished
    assert tts.synthesized_sentences == ["One sec."]
    assert [t.role for _, _, t in memory.written] == ["user"]  # filler isn't remembered


@pytest.mark.asyncio
async def test_stalled_agent_falls_back_to_the_plain_llm():
    from companion.core.agent import ToolStarted
    from companion.core.orchestrator.orchestrator import VOICE_AGENT_TIMEOUT_CAVEAT

    agent = _FakeAgent([ToolStarted("get_datetime"), 30])
    tts = _FakeTTS()
    memory = _FakeMemoryStore()
    orchestrator = _agent_orchestrator(agent, tts=tts, memory=memory,
                                       llm=_FakeLLM(["From what I know, it's Monday."]))
    orchestrator.agent_voice_timeout_s = 0.05
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=20, frame_interval=0.01)

    _ = await asyncio.wait_for(
        _collect(orchestrator.handle_voice_turn("u1", _tool_persona(), mic)), timeout=5
    )

    assert agent.cancelled
    assert tts.synthesized_sentences == ["One sec.", VOICE_AGENT_TIMEOUT_CAVEAT,
                                         "From what I know, it's Monday."]
    assert memory.written[-1][2].text == "From what I know, it's Monday."


@pytest.mark.asyncio
async def test_agent_error_propagates_like_an_llm_error():
    class _BrokenAgent(_FakeAgent):
        async def run_stream(self, persona, user_id, built):
            raise RuntimeError("model not pulled")
            yield  # pragma: no cover — makes this an async generator

    orchestrator = _agent_orchestrator(_BrokenAgent([]))
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=10, frame_interval=0.01)

    with pytest.raises(RuntimeError, match="model not pulled"):
        await asyncio.wait_for(
            _collect(orchestrator.handle_voice_turn("u1", _tool_persona(), mic)), timeout=5
        )


async def _collect(stream) -> list[bytes]:
    return [chunk async for chunk in stream]


@pytest.mark.asyncio
async def test_pending_announcements_are_spoken_first_but_not_remembered():
    tts = _FakeTTS()
    memory = _FakeMemoryStore()
    orchestrator = _make_orchestrator(tts=tts, memory=memory, llm=_FakeLLM(["Hi."]))
    pending = [["Reminder: stretch."]]
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=10, frame_interval=0.01)

    _ = [c async for c in orchestrator.handle_voice_turn(
        "u1", _persona(), mic, pending_announcements=lambda: pending.pop() if pending else [])]

    assert tts.synthesized_sentences == ["Reminder: stretch.", "Hi."]
    assert memory.written[-1][2].text == "Hi."


class _RecordingSummarizer:
    """Records calls and when they happened relative to memory writes."""

    def __init__(self, memory: _FakeMemoryStore) -> None:
        self.memory = memory
        self.calls: list[tuple[str, str, int]] = []

    async def summarize_if_due(self, user_id: str, persona_id: str) -> bool:
        self.calls.append((user_id, persona_id, len(self.memory.written)))
        return False


@pytest.mark.asyncio
async def test_voice_turn_schedules_summarizer_after_memory_writes():
    """docs/ROLLING_SUMMARIZER.md section 5: after each voice turn's writes,
    the summarizer runs as a background task."""
    memory = _FakeMemoryStore()
    summarizer = _RecordingSummarizer(memory)
    orchestrator = Orchestrator(
        stt=_FakeSTT(transcript="hello there"),
        tts=_FakeTTS(),
        llm=_FakeLLM(["Hi. "]),
        vad=_FakeVAD(),
        memory=memory,
        prompt_builder=PromptBuilder(),
        summarizer=summarizer,
    )
    mic = _mic([SPEECH_FRAME, SILENCE_FRAME], trailing_silence=20)

    _ = [c async for c in orchestrator.handle_voice_turn("u1", _persona(), mic)]
    await orchestrator.wait_for_background_tasks()

    assert summarizer.calls == [("u1", "mentor", 2)]  # both turns written first
