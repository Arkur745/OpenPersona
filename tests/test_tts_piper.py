"""Tests for companion/backends/tts/piper_local.py.

Same two-tier pattern as tests/test_stt_whisper.py: fake-voice-injected
tests (bypass `_ensure_voice`'s lazy load by setting `._voice` directly)
cover PiperTTS's own chunking/stop/resolution logic deterministically and
offline; `test_real_piper_backend_smoke` additionally exercises the real
`piper` library end-to-end against a small downloaded voice, skipping
itself (not failing) if that's not possible in this environment.
"""

import asyncio
import time

import pytest

from companion.backends.tts.piper_local import PiperTTS


class _FakeAudioChunk:
    def __init__(self, pcm_bytes: bytes) -> None:
        self.audio_int16_bytes = pcm_bytes


class _FakeVoice:
    """Stands in for piper.PiperVoice. Returns one scripted list of chunks
    per call, ignoring the actual text — isolates PiperTTS's own streaming/
    resolution logic from piper's (real) synthesis output."""

    def __init__(self, chunks_per_call: list[list[bytes]]) -> None:
        self._chunks_per_call = chunks_per_call
        self.calls = 0
        self.received_texts: list[str] = []

    def synthesize(self, text, syn_config=None):
        self.received_texts.append(text)
        chunks = self._chunks_per_call[min(self.calls, len(self._chunks_per_call) - 1)]
        self.calls += 1
        return [_FakeAudioChunk(c) for c in chunks]


class _SlowFakeVoice:
    """Like _FakeVoice, but yields lazily with a delay between chunks, so
    a stop() called mid-consumption has real timing to interrupt."""

    def __init__(self, chunks: list[bytes], delay: float = 0.05) -> None:
        self.chunks = chunks
        self.delay = delay

    def synthesize(self, text, syn_config=None):
        for pcm in self.chunks:
            yield _FakeAudioChunk(pcm)
            time.sleep(self.delay)


@pytest.mark.asyncio
async def test_synthesize_stream_yields_backend_chunks_in_order():
    tts = PiperTTS(voice_model_path="unused")
    fake_voice = _FakeVoice([[b"chunk1", b"chunk2"]])
    tts._voice = fake_voice

    received = [c async for c in tts.synthesize_stream("hello there")]

    assert received == [b"chunk1", b"chunk2"]
    assert fake_voice.received_texts == ["hello there"]


@pytest.mark.asyncio
async def test_synthesize_stream_reuses_loaded_voice_across_calls():
    tts = PiperTTS(voice_model_path="unused")
    fake_voice = _FakeVoice([[b"a"], [b"b"]])
    tts._voice = fake_voice

    first = [c async for c in tts.synthesize_stream("one")]
    second = [c async for c in tts.synthesize_stream("two")]

    assert first == [b"a"]
    assert second == [b"b"]
    assert fake_voice.calls == 2  # no re-load between calls


@pytest.mark.asyncio
async def test_stop_halts_playback_stream():
    tts = PiperTTS(voice_model_path="unused")
    tts._voice = _SlowFakeVoice([b"a", b"b", b"c", b"d", b"e"])

    received = []
    async for chunk in tts.synthesize_stream("hello world"):
        received.append(chunk)
        tts.stop()

    assert received == [b"a"]


@pytest.mark.asyncio
async def test_stop_does_not_carry_over_to_the_next_call():
    """.stop() aborts in-flight synthesis, not future calls — matches the
    TextToSpeech interface's documented contract."""
    tts = PiperTTS(voice_model_path="unused")
    tts._voice = _FakeVoice([[b"a", b"b"]])
    tts.stop()  # called with nothing in flight

    received = [c async for c in tts.synthesize_stream("hello")]

    assert received == [b"a", b"b"]


def test_from_voice_id_resolves_model_path(tmp_path):
    tts = PiperTTS.from_voice_id("piper:en_US-lessac-medium", voices_dir=str(tmp_path))

    assert tts.voice_model_path == str(tmp_path / "en_US-lessac-medium.onnx")


def test_from_voice_id_rejects_other_backend_namespace(tmp_path):
    with pytest.raises(ValueError):
        PiperTTS.from_voice_id("coqui:something", voices_dir=str(tmp_path))


def test_real_piper_backend_smoke(tmp_path):
    """Optional real end-to-end check: downloads a small stock Piper voice
    and synthesizes a short clip. Requires the `tts-piper` extra and
    network access to Hugging Face; skipped (not failed) if either isn't
    available."""
    pytest.importorskip("piper")
    from piper.download_voices import download_voice

    try:
        download_voice("en_US-lessac-low", tmp_path)
    except Exception as exc:  # network failure, etc.
        pytest.skip(f"could not download a Piper voice in this environment: {exc}")

    async def run():
        tts = PiperTTS(voice_model_path=str(tmp_path / "en_US-lessac-low.onnx"))
        chunks = [c async for c in tts.synthesize_stream("Hello there. This is a test.")]
        return tts, chunks

    tts, chunks = asyncio.run(run())

    assert len(chunks) >= 1
    assert all(isinstance(c, bytes) and len(c) > 0 for c in chunks)
    assert tts.sample_rate == 16000  # "low" quality Piper voices are 16 kHz


def test_real_piper_sample_rate_mismatch_raises(tmp_path):
    pytest.importorskip("piper")
    from piper.download_voices import download_voice

    try:
        download_voice("en_US-lessac-low", tmp_path)
    except Exception as exc:
        pytest.skip(f"could not download a Piper voice in this environment: {exc}")

    async def run():
        tts = PiperTTS(
            voice_model_path=str(tmp_path / "en_US-lessac-low.onnx"),
            sample_rate=99999,
        )
        async for _ in tts.synthesize_stream("hi"):
            pass

    with pytest.raises(ValueError, match="sample_rate"):
        asyncio.run(run())
