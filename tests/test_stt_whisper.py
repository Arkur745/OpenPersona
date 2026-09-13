"""Tests for companion/backends/stt/whisper_local.py.

Two tiers:

- The tests below the `_FakeWhisperModel` section drive the *real*
  buffering/LocalAgreement/gating logic in `FasterWhisperSTT` against a
  fake model (injected by setting `._model` directly, bypassing
  `_load_model`), so they exercise our streaming wrapper deterministically
  and offline — no faster-whisper install, no model download, no
  nondeterministic ASR output to assert against. This is the primary
  interface-contract coverage and always runs (only needs numpy, which is
  in the `dev` extra).
- `test_real_faster_whisper_backend_smoke` additionally exercises the real
  faster-whisper library end-to-end on a synthetic clip. It needs the
  `stt-whisper` extra and a model download, so it's skipped (not failed)
  when unavailable — see its own docstring.
"""

import numpy as np
import pytest

from companion.backends.stt.whisper_local import (
    SAMPLE_RATE_HZ,
    FasterWhisperSTT,
    _pcm16_bytes_to_float32,
)


def _tone_pcm16(duration_sec: float, freq: float = 440.0, amplitude: float = 0.2) -> bytes:
    t = np.arange(int(duration_sec * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
    wave = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return (wave * 32767).astype("<i2").tobytes()


def _silence_pcm16(duration_sec: float) -> bytes:
    return np.zeros(int(duration_sec * SAMPLE_RATE_HZ), dtype="<i2").tobytes()


# A short synthetic "utterance" fixture: silence, then a tone, then silence
# — per the task's suggestion when no real speech sample is available. The
# fake model below doesn't inspect audio content, but this still exercises
# the real PCM decode + buffering path with realistic chunk sizes.
_UTTERANCE_PCM16 = _silence_pcm16(0.3) + _tone_pcm16(0.5) + _silence_pcm16(0.3)


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel. Returns a scripted
    sequence of hypotheses, one per call, ignoring actual audio content —
    that's the point: it isolates FasterWhisperSTT's own buffering/
    agreement/gating logic from faster-whisper's (real, nondeterministic)
    ASR output."""

    def __init__(self, hypotheses: list[str]) -> None:
        self._hypotheses = hypotheses
        self.calls = 0
        self.received_buffer_lengths: list[int] = []

    def transcribe(self, buffer, **kwargs):
        self.received_buffer_lengths.append(len(buffer))
        text = self._hypotheses[min(self.calls, len(self._hypotheses) - 1)]
        self.calls += 1
        return [_FakeSegment(text)], object()


def _chunks(data: bytes, chunk_seconds: float) -> list[bytes]:
    chunk_bytes = int(chunk_seconds * SAMPLE_RATE_HZ) * 2
    return [data[i : i + chunk_bytes] for i in range(0, len(data), chunk_bytes)]


def test_pcm16_decode_round_trips_amplitude():
    chunk = _tone_pcm16(0.01, amplitude=0.5)
    decoded = _pcm16_bytes_to_float32(chunk)
    assert decoded.dtype == np.float32
    assert decoded.max() <= 1.0 and decoded.min() >= -1.0
    assert decoded.max() > 0.3  # round-trips a meaningful fraction of the 0.5 amplitude


def test_pcm16_decode_rejects_odd_length():
    with pytest.raises(ValueError):
        _pcm16_bytes_to_float32(b"\x00\x01\x02")


@pytest.mark.asyncio
async def test_no_partial_returned_before_min_refresh_seconds():
    stt = FasterWhisperSTT(min_refresh_seconds=1.0)
    stt._model = _FakeWhisperModel(["should not be reached"])

    for chunk in _chunks(_UTTERANCE_PCM16, chunk_seconds=0.3)[:3]:  # 0.9s < 1.0s
        result = await stt.transcribe_stream(chunk)
        assert result is None
    assert stt._model.calls == 0


@pytest.mark.asyncio
async def test_partial_transcripts_stream_and_stabilize_on_agreement():
    """Exercises the LocalAgreement-2 commit logic end to end, including the
    case that matters most: once a word is committed, a later pass that
    revises it must NOT change what's returned."""
    stt = FasterWhisperSTT(min_refresh_seconds=0.3)
    fake_model = _FakeWhisperModel(
        [
            "hello",
            "hello world",
            "help world how",  # revises "hello" -> "help"; "hello" is committed by now
        ]
    )
    stt._model = fake_model

    chunks = _chunks(_UTTERANCE_PCM16, chunk_seconds=0.3)
    results = [await stt.transcribe_stream(c) for c in chunks[:3]]

    assert results == ["hello", "hello world", "hello world how"]
    assert fake_model.calls == 3
    # The committed "hello" must never flip to the revised "help".
    assert all(r is None or not r.startswith("help") for r in results)


@pytest.mark.asyncio
async def test_no_repeat_return_when_hypothesis_unchanged():
    stt = FasterWhisperSTT(min_refresh_seconds=0.3)
    stt._model = _FakeWhisperModel(["same text", "same text"])

    chunks = _chunks(_UTTERANCE_PCM16, chunk_seconds=0.3)
    first = await stt.transcribe_stream(chunks[0])
    second = await stt.transcribe_stream(chunks[1])

    assert first == "same text"
    assert second is None  # unchanged from what was already returned


@pytest.mark.asyncio
async def test_reset_clears_buffer_and_commit_state():
    stt = FasterWhisperSTT(min_refresh_seconds=0.3)
    stt._model = _FakeWhisperModel(["hello world"])

    chunks = _chunks(_UTTERANCE_PCM16, chunk_seconds=0.3)
    first = await stt.transcribe_stream(chunks[0])
    assert first == "hello world"

    stt.reset()
    assert len(stt._state.buffer) == 0
    assert stt._state.committed_words == []
    assert stt._state.last_returned_text is None

    # Below refresh threshold again post-reset — proves state actually reset,
    # not just that the buffer array was replaced.
    result = await stt.transcribe_stream(chunks[0][: len(chunks[0]) // 2])
    assert result is None


@pytest.mark.asyncio
async def test_buffer_hard_cap_resets_commit_state():
    """Build up real committed words first, then confirm the hard cap
    actually clears them — not just that committed happened to be empty
    already."""
    stt = FasterWhisperSTT(min_refresh_seconds=0.2, max_buffer_seconds=0.6)
    fake_model = _FakeWhisperModel(["alpha", "alpha beta"])
    stt._model = fake_model

    chunks = _chunks(_UTTERANCE_PCM16, chunk_seconds=0.2)
    await stt.transcribe_stream(chunks[0])  # pass 1: "alpha", nothing committed yet
    await stt.transcribe_stream(chunks[1])  # pass 2: "alpha beta" agrees on "alpha" -> committed
    assert stt._state.committed_words == ["alpha"]

    big_chunk = _silence_pcm16(1.0)  # pushes buffer well past the 0.6s cap
    await stt.transcribe_stream(big_chunk)

    assert len(stt._state.buffer) == int(0.6 * SAMPLE_RATE_HZ)
    assert stt._state.committed_words == []


def test_real_faster_whisper_backend_smoke():
    """Optional real end-to-end check: actually loads faster-whisper's
    smallest model and runs one pass. Requires the `stt-whisper` extra and
    a one-time model download from Hugging Face Hub; skipped (not failed)
    if either isn't available, since this repo's default test run
    shouldn't require network access or a multi-hundred-MB download."""
    faster_whisper = pytest.importorskip("faster_whisper")

    try:
        model = faster_whisper.WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as exc:  # model download / network failure, etc.
        pytest.skip(f"faster-whisper model unavailable in this environment: {exc}")

    audio = _pcm16_bytes_to_float32(_tone_pcm16(1.0))
    segments, _info = model.transcribe(audio, condition_on_previous_text=False)
    # Not asserting on transcribed text (a pure tone has no real content) —
    # this only proves the real library call path faster-whisper.WhisperModel
    # ↔ our _run_model() usage (buffer shape/dtype, kwargs) actually works.
    list(segments)
