"""Ties STT -> VAD -> LLM -> TTS -> MemoryStore together for a
conversation.

Both `handle_text_turn` and `handle_voice_turn` are real now that
`MemoryStore` (`companion/memory/store.py`, see docs/MEMORY_STORE.md) is
implemented. `handle_text_turn` is request/response (caller owns and
passes `recent_turns` fresh each call); `handle_voice_turn` maintains its
own session-scoped `recent_turns` across the whole listening/speaking
loop — see its docstring and docs/ARCHITECTURE.md's "Voice-turn state
machine" section for the full design. `transcribe_chunk`,
`synthesize_reply`/`stop_speaking`, and `generate_reply` remain as thin,
real single-backend delegation methods, reused by `handle_voice_turn`
rather than duplicated.
"""

import asyncio
import contextlib
import logging
import re
from typing import AsyncIterator, Awaitable, Callable

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
from companion.core.prompt_builder.prompt_builder import BuiltPrompt, PromptBuilder

# First use of the stdlib logging module in this project's own code
# (elsewhere, e.g. __main__.py, surfaces errors via print() — a
# presentation-layer choice that doesn't fit here: Orchestrator is a
# backend-agnostic core layer with no idea whether it's running under
# text mode's CLI loop or voice mode's, so it shouldn't decide how a
# diagnostic gets displayed, only that one gets emitted. See
# _write_turn_safely for the one thing that currently logs here.
logger = logging.getLogger(__name__)

# Matches through a sentence-ending punctuation run and any trailing
# whitespace — e.g. "Hello there. " -> match ends after the space, leaving
# "Hello there." as the popped sentence. Deliberately simple: doesn't
# special-case abbreviations ("Mr.") or decimals ("3.14"); those get split
# early, which just means TTS gets a slightly short sentence, not a broken
# one. Real NLP sentence segmentation is a possible future upgrade, not
# needed for streaming TTS to work correctly.
_SENTENCE_END_RE = re.compile(r'[.!?]+[)"\']?(?:\s+|$)')


def _pop_complete_sentence(buffer: str) -> tuple[str | None, str]:
    """If `buffer` contains a complete sentence, return (sentence, rest);
    otherwise (None, buffer) unchanged."""
    match = _SENTENCE_END_RE.search(buffer)
    if not match:
        return None, buffer
    return buffer[: match.end()].strip(), buffer[match.end() :]


def _dedup_retrieved_memory(
    retrieved_memory: list[Turn], recent_turns: list[Turn]
) -> list[Turn]:
    """Drop any `retrieved_memory` turn that's also already present in
    `recent_turns` (by value equality — `Turn` is a frozen, structurally-
    comparable dataclass).

    Resolves the gap flagged in docs/PROMPT_BUILDER.md and designed in
    docs/MEMORY_STORE.md section 4: nothing stops `MemoryStore.query()`
    from returning a turn that's also in the live conversation's own
    recent window, since that turn was written to memory as it happened.
    Deliberately resolved here (the one place both lists are already in
    hand) rather than by adding an exclude parameter to
    `MemoryStore.query()` — see that doc for why. Relies on `write()`/
    `query()` round-tripping `Turn.timestamp` exactly (verified — see
    `companion/memory/store.py`'s docstring); if that stops holding this
    silently stops deduplicating instead of raising, which is why it's
    worth remembering if this filter ever looks like it's not working."""
    recent = set(recent_turns)
    return [turn for turn in retrieved_memory if turn not in recent]


def _trim_to_last(turns: list[Turn], max_len: int) -> None:
    """Trim `turns` in place to its last `max_len` items. `list[:-0]` is
    `list[:0]` (since `-0 == 0`), not "keep nothing" — plain
    `del turns[:-max_len]` silently keeps everything when `max_len == 0`,
    so that case needs its own branch rather than falling through."""
    if max_len <= 0:
        turns.clear()
    else:
        del turns[:-max_len]


class _FrameBuffer:
    """Accumulates arbitrary-sized byte chunks and hands back fixed-size
    frames as they become available — e.g. mic-capture chunks of whatever
    size into SileroVAD's exact 512-sample requirement. See
    docs/ARCHITECTURE.md's "VAD" section for why this lives here rather
    than inside the VAD backend itself."""

    def __init__(self, frame_bytes: int) -> None:
        self.frame_bytes = frame_bytes
        self._buffer = bytearray()

    def push(self, chunk: bytes) -> list[bytes]:
        self._buffer.extend(chunk)
        frames = []
        while len(self._buffer) >= self.frame_bytes:
            frames.append(bytes(self._buffer[: self.frame_bytes]))
            del self._buffer[: self.frame_bytes]
        return frames


class Orchestrator:
    """Backend-agnostic conversation loop.

    Holds only interface references — never a concrete backend class — so
    swapping e.g. the LLM from Ollama to a future cloud adapter requires no
    change here, only a different object passed in at construction time
    (see config/config.example.yaml for how backend selection happens).
    """

    def __init__(
        self,
        stt: SpeechToText,
        tts: TextToSpeech,
        llm: LLMBackend,
        vad: VoiceActivityDetector,
        memory: MemoryStore,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.vad = vad
        self.memory = memory
        self.prompt_builder = prompt_builder

    async def handle_text_turn(
        self, user_id: str, persona: Persona, recent_turns: list[Turn], message: str
    ) -> str:
        """Text-mode turn: no STT/VAD/TTS involved, just memory + LLM.

        Request/response, unlike `handle_voice_turn`'s whole-session loop
        — the caller owns and passes `recent_turns` fresh each call (e.g.
        a chat UI's own history), rather than this method tracking a
        session across calls. Queries `MemoryStore` for relevant past
        turns, de-duplicates against `recent_turns` (see
        `_dedup_retrieved_memory`), builds the prompt, generates the full
        reply, and persists both sides of the exchange to `MemoryStore`
        before returning — so a later call (or a future session) can
        retrieve this exchange as memory.
        """
        retrieved_memory = await self.memory.query(
            user_id, persona.id, message, k=self.prompt_builder.max_memory_turns
        )
        retrieved_memory = _dedup_retrieved_memory(retrieved_memory, recent_turns)

        built: BuiltPrompt = self.prompt_builder.build(
            persona=persona,
            recent_turns=recent_turns,
            retrieved_memory=retrieved_memory,
            user_message=message,
        )

        reply = "".join(
            [token async for token in self.generate_reply(built.prompt, built.context)]
        )

        await self._write_turn_safely(
            user_id, persona.id, Turn(role="user", text=message.strip())
        )
        if reply.strip():
            await self._write_turn_safely(
                user_id, persona.id, Turn(role="assistant", text=reply.strip())
            )

        return reply

    async def _write_turn_safely(self, user_id: str, persona_id: str, turn: Turn) -> None:
        """Writes one turn to `MemoryStore`, catching a backend's
        `MemoryStoreError` — a genuine persistence-layer failure (disk
        corruption, an unreachable server) — rather than letting it
        propagate.

        Deliberate, not a default choice made for convenience: per
        `docs/MEMARK_RETRIEVAL_ADAPTER.md` section 8, `MemArkMemoryStore
        .write()` specifically does *not* swallow a persist failure
        itself — it lets `MemArkPersistenceError` (a `MemoryStoreError`)
        propagate, precisely so this decision could be made once, here,
        deliberately, rather than silently one layer down. Evaluated
        against three options: end the turn (this one), end the session,
        or swallow with only a log line. Ending the session is
        disproportionate — `MemoryStoreError` represents one failed
        write, not proof the whole store is unusable, and the user can't
        do anything about a disk hiccup anyway. Swallowing with just a
        log line would defeat the entire point of `MemArkPersistenceError`
        propagating out of `write()` in the first place — this project's
        established convention (`AudioDeviceError`, `MemArkGraphStore`'s
        own corruption handling) is that a real failure is always
        surfaced loudly, never silently absorbed as if nothing happened.

        This method always runs *after* the response has already been
        generated (text mode) or already yielded/spoken (voice mode) —
        never gates it — so a write failure here can never affect the
        response the user already received; the only thing lost is this
        one turn's long-term memory, logged at ERROR level (not raised
        further) so it's diagnosable rather than invisible. No retry: a
        transient failure gets another chance on the *next* turn's write,
        which is enough given the alternative (blocking the user's
        conversation on a background persistence concern they have no
        way to act on) is worse.
        """
        try:
            await self.memory.write(user_id, persona_id, turn)
        except MemoryStoreError:
            logger.exception(
                "MemoryStore.write() failed — turn not saved to long-term "
                "memory (user_id=%s, persona_id=%s, role=%s, text_preview=%r)",
                user_id,
                persona_id,
                turn.role,
                turn.text[:50],
            )

    async def transcribe_chunk(self, audio_chunk: bytes) -> str | None:
        """Feed one chunk of mic audio to the configured STT backend.

        This is the STT-only slice of the voice-turn loop — delegates
        straight to `self.stt.transcribe_stream()`, whatever concrete
        backend config selected (e.g. `FasterWhisperSTT`). Superseded, for
        the actual voice-turn loop, by the VAD-integrated logic inside
        `handle_voice_turn`; kept as a standalone real method for direct
        use/testing of the STT slice alone.
        """
        return await self.stt.transcribe_stream(audio_chunk)

    async def synthesize_reply(self, text_chunk: str) -> AsyncIterator[bytes]:
        """Feed one chunk of reply text (typically one sentence) to the
        configured TTS backend. Delegates straight to
        `self.tts.synthesize_stream()`. See `handle_voice_turn` for how
        this composes with barge-in via `stop_speaking()`."""
        async for audio_chunk in self.tts.synthesize_stream(text_chunk):
            yield audio_chunk

    def stop_speaking(self) -> None:
        """Abort in-flight TTS playback immediately. Delegates to
        `self.tts.stop()` — the same call `handle_voice_turn` uses for
        barge-in; no separate interruption mechanism exists."""
        self.tts.stop()

    async def generate_reply(
        self, prompt: str, context: list[Turn]
    ) -> AsyncIterator[str]:
        """Feed an already-assembled prompt + context to the configured LLM
        backend. Delegates straight to `self.llm.generate_stream()`.
        Assembling `prompt`/`context` is `PromptBuilder`'s job — see
        `core/prompt_builder/` and `docs/PROMPT_BUILDER.md`.
        """
        async for token in self.llm.generate_stream(prompt, context):
            yield token

    async def _sentences_from_llm(
        self, prompt: str, context: list[Turn]
    ) -> AsyncIterator[str]:
        """Bridge LLM token streaming to TTS's sentence-granularity
        `synthesize_stream()` — see PiperTTS/CoquiTTS docstrings for why
        that granularity is what actually matters for streaming latency."""
        buffer = ""
        async for token in self.generate_reply(prompt, context):
            buffer += token
            while True:
                sentence, buffer = _pop_complete_sentence(buffer)
                if sentence is None:
                    break
                if sentence:
                    yield sentence
        remainder = buffer.strip()
        if remainder:
            yield remainder

    async def _pump_mic(
        self,
        mic_audio: AsyncIterator[bytes],
        frame_buffer: "_FrameBuffer",
        events: "asyncio.Queue[tuple[bool, str | None, bool | None]]",
    ) -> None:
        """The *only* thing that ever calls `mic_audio.__anext__()` for a
        given `handle_voice_turn` call, running as one long-lived
        background task for its whole duration. Every classified chunk
        (or the stream ending) is pushed onto `events` as
        `(stream_ended, transcript_or_None, is_speech_or_None)` — same
        shape `_read_and_classify` used to return directly.

        `stream_ended`'s queue item is a one-shot value with two possible
        readers across a session (the listening loop, the per-response
        barge-in watcher) that are never concurrently active but do
        alternate — whichever one reads it "spends" it. Only the
        listening loop's own `stream_ended` branch is allowed to let it
        stay spent, since that ends `handle_voice_turn` for good; the
        watcher must put it back before returning, or the *next*
        listening iteration hangs forever on a queue nothing will ever
        fill again (a real deadlock found via a real, unscripted mic
        round trip — see `watch_for_barge_in`).

        This exists because of a real bug caught in testing, not
        speculative caution: an earlier version had two separate call
        sites (`handle_voice_turn`'s listening loop and a per-response
        `watch_for_barge_in` task) each independently calling
        `mic_iter.__anext__()` on the *same* shared async generator, and
        cancelling the watcher task — needed to stop watching once a
        response finishes normally — while it was suspended mid-`await`
        inside that shared generator's `__anext__()` call silently kills
        the generator (confirmed with a minimal repro: cancelling a task
        mid-`await` on an async generator's `__anext__()` leaves it
        raising `StopAsyncIteration` forever after, even with items still
        "left"). Every subsequent read looked like the mic stream had
        ended. `asyncio.Queue.get()`, unlike an async generator's
        `__anext__()`, is safe to have a cancelled waiter on — cancelling
        one waiter doesn't corrupt the queue for whoever reads it next.
        So: one task owns the only `__anext__()` calls and is never
        cancelled except when `handle_voice_turn` itself is done with the
        whole session; everything else just consumes the queue and can be
        cancelled freely.
        """
        mic_iter = mic_audio.__aiter__()
        while True:
            try:
                chunk = await mic_iter.__anext__()
            except StopAsyncIteration:
                await events.put((True, None, None))
                return

            transcript = await self.stt.transcribe_stream(chunk)

            is_speech = None
            for frame in frame_buffer.push(chunk):
                is_speech = self.vad.is_speech(frame)

            await events.put((False, transcript, is_speech))

    async def handle_voice_turn(
        self,
        user_id: str,
        persona: Persona,
        mic_audio: AsyncIterator[bytes],
        *,
        on_playback_interrupt: Callable[[], None] | None = None,
        wait_for_playback_drained: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[bytes]:
        """Run a full voice conversation session: listen, respond, repeat,
        until `mic_audio` ends. Yields TTS playback audio chunks.

        This is the real implementation of the state machine documented in
        docs/ARCHITECTURE.md — read that first; this docstring covers what
        the code does, not why.

        Composes, per cycle: VAD (end-of-utterance + barge-in) -> STT
        (already accumulating) -> `MemoryStore.query()` (+ dedup against
        this session's own history) -> `PromptBuilder.build()` ->
        LLMBackend -> TTS -> `MemoryStore.write()`, with barge-in calling
        `self.tts.stop()` (the same method `stop_speaking()` above uses —
        no second interruption mechanism).

        **`recent_turns` is session-scoped, not persisted across calls:**
        a plain `list[Turn]` local to this call, growing as the session
        progresses and trimmed to `prompt_builder.max_recent_turns` after
        each append (older turns are still retrievable later via
        `MemoryStore.query()` — they're just no longer "recent"). A new
        call to `handle_voice_turn` (e.g. a new session) starts this list
        empty again; only `MemoryStore` persists across sessions.

        **Assistant turns are recorded as whole sentences, not partial
        audio:** each sentence pulled from `_sentences_from_llm` is
        appended to this turn's transcript as soon as it's generated,
        before TTS speaks it — so an interruption mid-sentence still
        records that whole sentence as said (the LLM did produce it),
        rather than an audio-level partial fragment. Written to memory
        once the turn's speaking phase ends, whether finished or
        interrupted, using whatever text was actually generated.

        **A gap surfaced by actually implementing this, not resolved by
        changing an abstract interface:** re-chunking `mic_audio` into
        VAD's required fixed frame size needs to know that size, but
        `VoiceActivityDetector` (the abstract interface) has no
        `frame_bytes` property — only concrete `SileroVAD` happens to
        expose one. Rather than add that to the interface on the strength
        of a single implementation, this method duck-types it via
        `getattr` and fails loudly (not silently) if it's missing. If a
        second VAD backend needs this too, promoting it to the abstract
        interface then would rest on two real implementations agreeing on
        the shape, not one.

        **`on_playback_interrupt` / `wait_for_playback_drained` — real
        playback-latency barge-in, resolved:** previously-documented
        known limitation, now closed for real now that
        `companion/audio/playback.py`'s `SpeakerPlayback` gives a real
        caller to drive this with (voice-mode CLI build-order pass — see
        docs/ARCHITECTURE.md's "Voice mode audio I/O" section). The gap:
        synthesis+yielding a response can finish well before the audio
        has actually finished
        *sounding* out of a speaker (playback has its own buffering
        latency) — watching for barge-in only until yielding stops leaves
        a window where the user could start talking during that residual
        playback tail without it registering. Both parameters are
        optional and default to `None`, in which case this method behaves
        exactly as it always has (watcher cancelled the moment yielding
        stops) — existing callers/tests are unaffected.

        If given: `on_playback_interrupt()` is called synchronously,
        alongside `self.tts.stop()`, the moment real barge-in is detected
        — the caller's chance to immediately discard whatever audio is
        already queued for playback but hasn't sounded yet (`self.tts
        .stop()` alone only stops *synthesis* from producing more audio;
        it was never wired to reach into a downstream playback buffer,
        and still isn't — that stays this new parameter's job, not a
        change to `TextToSpeech`). After a response finishes *yielding*
        normally (not already interrupted), `await
        wait_for_playback_drained()` is called before the barge-in
        watcher is cancelled — the contract is that this resolves once
        the caller has nothing left to actually play, whether that's
        because playback finished naturally or because
        `on_playback_interrupt()` just flushed it; either way the watcher
        stays alive and able to catch a late barge-in for the entire real
        playback tail, not just the synthesis+yield window. See
        `companion/audio/playback.py`'s `SpeakerPlayback` for the real
        implementation these two hooks are designed to drive, including
        the honest precision caveat on what "drained" actually means.
        """
        frame_bytes = getattr(self.vad, "frame_bytes", None)
        if not isinstance(frame_bytes, int):
            raise RuntimeError(
                "handle_voice_turn requires a VoiceActivityDetector that "
                "exposes a `frame_bytes` attribute (SileroVAD does); the "
                "configured VAD backend doesn't. See this method's "
                "docstring for why that's not part of the abstract "
                "VoiceActivityDetector interface."
            )

        frame_buffer = _FrameBuffer(frame_bytes)
        events: "asyncio.Queue[tuple[bool, str | None, bool | None]]" = asyncio.Queue()
        pump = asyncio.create_task(self._pump_mic(mic_audio, frame_buffer, events))
        recent_turns: list[Turn] = []
        max_recent = self.prompt_builder.max_recent_turns

        try:
            while True:
                # --- listening -> user speaking: accumulate until end-of-utterance ---
                self.stt.reset()
                transcript = ""
                user_has_spoken = False
                while True:
                    stream_ended, chunk_transcript, is_speech = await events.get()
                    if stream_ended:
                        return  # mic stream ended before any utterance completed
                    if chunk_transcript is not None:
                        transcript = chunk_transcript
                    if is_speech is None:
                        continue
                    if is_speech:
                        user_has_spoken = True
                    elif user_has_spoken:
                        break  # was speaking, now silent long enough -> end of utterance

                if not transcript.strip():
                    continue  # VAD triggered but STT captured nothing meaningful

                # --- processing: query memory, dedup, build the prompt ---
                retrieved_memory = await self.memory.query(
                    user_id,
                    persona.id,
                    transcript,
                    k=self.prompt_builder.max_memory_turns,
                )
                retrieved_memory = _dedup_retrieved_memory(retrieved_memory, recent_turns)

                built: BuiltPrompt = self.prompt_builder.build(
                    persona=persona,
                    recent_turns=recent_turns,
                    retrieved_memory=retrieved_memory,
                    user_message=transcript,
                )

                user_turn = Turn(role="user", text=transcript.strip())
                recent_turns.append(user_turn)
                _trim_to_last(recent_turns, max_recent)
                await self._write_turn_safely(user_id, persona.id, user_turn)

                # --- speaking response, watching for barge-in concurrently ---
                interrupted = asyncio.Event()
                response_sentences: list[str] = []

                async def watch_for_barge_in() -> None:
                    while not interrupted.is_set():
                        stream_ended, _transcript, is_speech = await events.get()
                        if stream_ended:
                            # Not a barge-in — nothing more will ever
                            # arrive on the mic, but that's not the same
                            # as the user interrupting right now. Let the
                            # in-flight response finish normally; put the
                            # signal back so the next listening iteration
                            # (which does end the session) still sees it
                            # — see _pump_mic's docstring.
                            await events.put((True, None, None))
                            return
                        if is_speech:
                            self.stop_speaking()
                            if on_playback_interrupt is not None:
                                on_playback_interrupt()
                            interrupted.set()
                            return

                watcher = asyncio.create_task(watch_for_barge_in())
                try:
                    async for sentence in self._sentences_from_llm(
                        built.prompt, built.context
                    ):
                        response_sentences.append(sentence)
                        if interrupted.is_set():
                            break
                        async for audio_chunk in self.tts.synthesize_stream(sentence):
                            if interrupted.is_set():
                                break
                            yield audio_chunk
                        if interrupted.is_set():
                            break

                    if wait_for_playback_drained is not None and not interrupted.is_set():
                        # Keep the watcher alive through the real playback
                        # tail — finishing synthesis+yielding doesn't mean
                        # the audio has actually finished sounding yet.
                        await wait_for_playback_drained()
                finally:
                    # Safe to cancel: `watcher` only ever awaits events.get()
                    # (a Queue), never mic_audio directly — see _pump_mic's
                    # docstring for why that distinction is load-bearing.
                    watcher.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await watcher

                assistant_text = " ".join(response_sentences).strip()
                if assistant_text:
                    assistant_turn = Turn(role="assistant", text=assistant_text)
                    recent_turns.append(assistant_turn)
                    _trim_to_last(recent_turns, max_recent)
                    await self._write_turn_safely(user_id, persona.id, assistant_turn)

                # -> [interrupted -> listening] or [finished -> listening]: loop.
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
