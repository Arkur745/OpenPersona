"""Orchestrator wiring tests — no optional backend dependencies needed.

Covers the STT, TTS, and LLM delegation slices
(`Orchestrator.transcribe_chunk`, `.synthesize_reply`/`.stop_speaking`,
`.generate_reply`) only. `handle_text_turn` is real now — its tests live
in tests/test_orchestrator_text_turn.py; `handle_voice_turn`'s
state-machine tests live in tests/test_orchestrator_voice_turn.py. Not
here, since both need a real (fake) MemoryStore, unlike the delegation
slices covered in this file."""

from typing import AsyncIterator
from unittest.mock import Mock

import pytest

from companion.core.interfaces import LLMBackend, SpeechToText, TextToSpeech, Turn
from companion.core.orchestrator import Orchestrator


class _RecordingSTT(SpeechToText):
    """Minimal real SpeechToText implementation for testing delegation —
    not FasterWhisperSTT, since this test shouldn't care which concrete
    backend config selected, only that the orchestrator calls *a*
    SpeechToText correctly."""

    def __init__(self) -> None:
        self.received_chunks: list[bytes] = []
        self.reset_calls = 0

    async def transcribe_stream(self, audio_chunk: bytes) -> str | None:
        self.received_chunks.append(audio_chunk)
        return f"transcribed:{audio_chunk!r}" if audio_chunk else None

    def reset(self) -> None:
        self.reset_calls += 1


class _RecordingTTS(TextToSpeech):
    """Minimal real TextToSpeech implementation for testing delegation —
    not PiperTTS/CoquiTTS, for the same reason as _RecordingSTT above."""

    def __init__(self) -> None:
        self.received_text_chunks: list[str] = []
        self.stop_calls = 0

    async def synthesize_stream(self, text_chunk: str) -> AsyncIterator[bytes]:
        self.received_text_chunks.append(text_chunk)
        yield f"audio-for:{text_chunk}".encode()

    def stop(self) -> None:
        self.stop_calls += 1


class _RecordingLLM(LLMBackend):
    """Minimal real LLMBackend implementation for testing delegation — not
    OllamaLLM, for the same reason as _RecordingSTT/_RecordingTTS above."""

    def __init__(self) -> None:
        self.received_calls: list[tuple[str, list[Turn]]] = []

    async def generate_stream(self, prompt: str, context: list[Turn]):
        self.received_calls.append((prompt, context))
        yield f"reply-to:{prompt}"


def _make_orchestrator(
    stt: SpeechToText | None = None,
    tts: TextToSpeech | None = None,
    llm: LLMBackend | None = None,
) -> Orchestrator:
    return Orchestrator(
        stt=stt or _RecordingSTT(),
        tts=tts or _RecordingTTS(),
        llm=llm or _RecordingLLM(),
        vad=Mock(),
        memory=Mock(),
        prompt_builder=Mock(),
    )


@pytest.mark.asyncio
async def test_transcribe_chunk_delegates_to_configured_stt_backend():
    stt = _RecordingSTT()
    orchestrator = _make_orchestrator(stt)

    result = await orchestrator.transcribe_chunk(b"\x01\x02")

    assert result == "transcribed:b'\\x01\\x02'"
    assert stt.received_chunks == [b"\x01\x02"]


@pytest.mark.asyncio
async def test_transcribe_chunk_passes_through_none():
    stt = _RecordingSTT()
    orchestrator = _make_orchestrator(stt)

    result = await orchestrator.transcribe_chunk(b"")

    assert result is None


@pytest.mark.asyncio
async def test_synthesize_reply_delegates_to_configured_tts_backend():
    tts = _RecordingTTS()
    orchestrator = _make_orchestrator(tts=tts)

    chunks = [c async for c in orchestrator.synthesize_reply("hello there")]

    assert chunks == [b"audio-for:hello there"]
    assert tts.received_text_chunks == ["hello there"]


def test_stop_speaking_delegates_to_configured_tts_backend():
    tts = _RecordingTTS()
    orchestrator = _make_orchestrator(tts=tts)

    orchestrator.stop_speaking()

    assert tts.stop_calls == 1


@pytest.mark.asyncio
async def test_generate_reply_delegates_to_configured_llm_backend():
    llm = _RecordingLLM()
    orchestrator = _make_orchestrator(llm=llm)
    context = [Turn(role="user", text="hi")]

    tokens = [t async for t in orchestrator.generate_reply("system prompt", context)]

    assert tokens == ["reply-to:system prompt"]
    assert llm.received_calls == [("system prompt", context)]


