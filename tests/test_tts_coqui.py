"""Tests for companion/backends/tts/coqui_local.py.

Same two-tier pattern as the STT and Piper test suites: fake-model-injected
tests (bypass `_ensure_model`'s lazy load by setting `._model` directly)
cover CoquiTTS's own chunking/stop/resolution/normalization logic
deterministically and offline; the `test_real_coqui_*` tests additionally
exercise the real `coqui-tts` library, skipping themselves (not failing)
if the extra isn't installed or the model can't be loaded in this
environment (some Coqui models need the espeak-ng system binary — see
coqui_local.py's module docstring).
"""

import asyncio

import numpy as np
import pytest

from companion.backends.tts.coqui_local import CoquiTTS

_REAL_MODEL_NAME = "tts_models/en/ljspeech/vits"


class _FakeSynthesizer:
    def __init__(self, sample_rate: int) -> None:
        self.output_sample_rate = sample_rate


class _FakeCoquiModel:
    """Stands in for TTS.api.TTS. Returns one scripted waveform per call,
    ignoring actual text — isolates CoquiTTS's own logic from the real
    (and here, non-streaming) synthesis output."""

    def __init__(self, wav_per_call: list, sample_rate: int = 22050) -> None:
        self.synthesizer = _FakeSynthesizer(sample_rate)
        self._wav_per_call = wav_per_call
        self.calls = 0
        self.received: list[tuple] = []

    def tts(self, text, speaker=None, language=None, speaker_wav=None):
        self.received.append((text, speaker, language, speaker_wav))
        wav = self._wav_per_call[min(self.calls, len(self._wav_per_call) - 1)]
        self.calls += 1
        return wav


def _decode_pcm16(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32767.0


def test_run_model_normalizes_list_and_ndarray_wav_types():
    tts = CoquiTTS(model_name="unused")
    original = [0.0, 0.5, -0.5, 1.0, -1.0]

    for wav_input in (original, np.array(original, dtype=np.float32)):
        fake_model = _FakeCoquiModel([wav_input])
        pcm_bytes = tts._run_model(fake_model, "hello")
        decoded = _decode_pcm16(pcm_bytes)
        assert len(decoded) == 5
        np.testing.assert_allclose(decoded, original, atol=1e-3)


@pytest.mark.asyncio
async def test_synthesize_stream_splits_output_into_chunk_frames():
    fake_model = _FakeCoquiModel([[0.0, 0.0, 0.0, 0.0, 0.0]])  # 5 samples
    tts = CoquiTTS(model_name="unused", chunk_frames=2)  # 2 samples = 4 bytes/chunk
    tts._model = fake_model

    chunks = [c async for c in tts.synthesize_stream("hello")]

    assert [len(c) for c in chunks] == [4, 4, 2]  # 10 bytes total, split 4+4+2
    assert sum(len(c) for c in chunks) == 10


@pytest.mark.asyncio
async def test_synthesize_stream_passes_speaker_and_language_to_model():
    fake_model = _FakeCoquiModel([[0.0]])
    tts = CoquiTTS(model_name="unused", speaker="alice", language="en")
    tts._model = fake_model

    _ = [c async for c in tts.synthesize_stream("hi")]

    assert fake_model.received == [("hi", "alice", "en", None)]


@pytest.mark.asyncio
async def test_synthesize_stream_reuses_loaded_model_across_calls():
    fake_model = _FakeCoquiModel([[0.0], [0.1]])
    tts = CoquiTTS(model_name="unused")
    tts._model = fake_model

    _ = [c async for c in tts.synthesize_stream("one")]
    _ = [c async for c in tts.synthesize_stream("two")]

    assert fake_model.calls == 2  # no re-load between calls


@pytest.mark.asyncio
async def test_stop_halts_playback_stream():
    fake_model = _FakeCoquiModel([[0.0] * 100])  # many samples -> many chunks
    tts = CoquiTTS(model_name="unused", chunk_frames=2)
    tts._model = fake_model

    received = []
    async for chunk in tts.synthesize_stream("hello"):
        received.append(chunk)
        tts.stop()

    assert len(received) == 1


@pytest.mark.asyncio
async def test_stop_does_not_carry_over_to_the_next_call():
    fake_model = _FakeCoquiModel([[0.0, 0.0]])
    tts = CoquiTTS(model_name="unused")
    tts.stop()  # called with nothing in flight

    tts._model = fake_model
    received = [c async for c in tts.synthesize_stream("hello")]

    assert len(received) == 1  # one chunk_frames-sized (default) chunk for 2 samples


def test_from_voice_id_resolves_speaker_name():
    tts = CoquiTTS.from_voice_id("coqui:alice", model_name="tts_models/some/model")

    assert tts.speaker == "alice"
    assert tts.model_name == "tts_models/some/model"
    assert tts.speaker_wav is None  # voice cloning not wired up via from_voice_id yet


def test_from_voice_id_default_means_single_speaker_model():
    tts = CoquiTTS.from_voice_id("coqui:default", model_name="tts_models/en/ljspeech/vits")

    assert tts.speaker is None


def test_from_voice_id_rejects_other_backend_namespace():
    with pytest.raises(ValueError):
        CoquiTTS.from_voice_id("piper:something", model_name="tts_models/some/model")


def _load_real_model_or_skip(**kwargs) -> CoquiTTS:
    tts = CoquiTTS(model_name=_REAL_MODEL_NAME, **kwargs)
    try:
        asyncio.run(tts._ensure_model())
    except Exception as exc:  # missing espeak-ng, network failure, etc.
        pytest.skip(f"Coqui model unavailable in this environment: {exc}")
    return tts


def test_real_coqui_backend_smoke():
    """Optional real end-to-end check against a small, non-XTTS stock
    model (avoids a multi-GB XTTS-v2 download). Requires the `tts-coqui`
    extra, network access, and — for this particular model — the
    espeak-ng system binary; skipped (not failed) if any of that isn't
    available."""
    pytest.importorskip("TTS")
    tts = _load_real_model_or_skip()

    async def run():
        return [c async for c in tts.synthesize_stream("Hello there. This is a test.")]

    chunks = asyncio.run(run())

    assert len(chunks) >= 1
    assert all(isinstance(c, bytes) and len(c) > 0 for c in chunks)
    assert tts.sample_rate > 0


def test_real_coqui_sample_rate_mismatch_raises():
    pytest.importorskip("TTS")
    # Probe with a correct instance first so an environment failure (e.g.
    # missing espeak-ng) skips the test instead of masquerading as the
    # ValueError assertion below passing for the wrong reason.
    _load_real_model_or_skip()

    mismatched = CoquiTTS(model_name=_REAL_MODEL_NAME, sample_rate=1)
    with pytest.raises(ValueError, match="sample_rate"):
        asyncio.run(mismatched._ensure_model())
