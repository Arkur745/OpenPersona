"""Tests for companion/backends/vad/silero_vad.py.

Debounce-state-machine tests mock at `_speech_probability` (not `_model`)
so they run fully offline without torch installed at all — torch is only
needed for the real decode+model path, exercised by the gated
`test_real_silero_backend_smoke`. See silero_vad.py's own module docstring
for why torch is deliberately not in the `dev` extra (unlike numpy/ollama,
it's genuinely heavy).
"""

import pytest

from companion.backends.vad.silero_vad import SileroVAD


def _make_vad(**overrides) -> SileroVAD:
    defaults = dict(threshold=0.5, min_speech_duration_ms=64.0, min_silence_duration_ms=64.0)
    defaults.update(overrides)
    return SileroVAD(**defaults)


def _script(vad: SileroVAD, probs: list[float]) -> None:
    """Replace _speech_probability with a scripted sequence — bypasses
    the real torch/model call entirely."""
    it = iter(probs)
    vad._speech_probability = lambda chunk: next(it)  # noqa: ARG005


def _frame(vad: SileroVAD) -> bytes:
    return b"\x00" * vad.frame_bytes


def test_frame_bytes_matches_16khz_512_sample_requirement():
    vad = _make_vad(sample_rate=16000)
    assert vad.frame_samples == 512
    assert vad.frame_bytes == 1024


def test_frame_bytes_matches_8khz_256_sample_requirement():
    vad = _make_vad(sample_rate=8000)
    assert vad.frame_samples == 256
    assert vad.frame_bytes == 512


def test_rejects_unsupported_sample_rate():
    with pytest.raises(ValueError):
        SileroVAD(sample_rate=44100)


def test_is_speech_rejects_wrong_sized_chunk():
    vad = _make_vad()
    with pytest.raises(ValueError, match="1024-byte frames"):
        vad.is_speech(b"\x00" * 100)


def test_below_threshold_never_triggers_speech():
    vad = _make_vad(min_speech_duration_ms=32.0)
    _script(vad, [0.1, 0.1, 0.1])
    for _ in range(3):
        assert vad.is_speech(_frame(vad)) is False


def test_sustained_speech_above_threshold_triggers_after_min_duration():
    vad = _make_vad(min_speech_duration_ms=64.0)  # 2 frames @ 32ms
    _script(vad, [0.9, 0.9, 0.9])
    assert vad.is_speech(_frame(vad)) is False  # 1st frame: not yet enough
    assert vad.is_speech(_frame(vad)) is True  # 2nd frame: min duration reached
    assert vad.is_speech(_frame(vad)) is True  # stays speaking


def test_brief_single_frame_dip_delays_onset_not_skips_it():
    # Onset requires CONSECUTIVE speech frames -- a dip resets the counter.
    vad = _make_vad(min_speech_duration_ms=64.0)  # 2 frames
    _script(vad, [0.9, 0.1, 0.9, 0.9])
    assert vad.is_speech(_frame(vad)) is False  # speech frame 1 of run
    assert vad.is_speech(_frame(vad)) is False  # dip resets the onset counter
    assert vad.is_speech(_frame(vad)) is False  # speech frame 1 of a new run
    assert vad.is_speech(_frame(vad)) is True  # speech frame 2 -> triggers


def test_sustained_silence_ends_speech_after_min_duration():
    vad = _make_vad(min_speech_duration_ms=32.0, min_silence_duration_ms=64.0)
    _script(vad, [0.9, 0.1, 0.1])
    assert vad.is_speech(_frame(vad)) is True  # triggers speech
    assert vad.is_speech(_frame(vad)) is True  # 1 silence frame: not yet enough
    assert vad.is_speech(_frame(vad)) is False  # 2nd silence frame: min duration reached


def test_brief_silence_dip_cancels_end_of_utterance_countdown():
    vad = _make_vad(min_speech_duration_ms=32.0, min_silence_duration_ms=64.0)
    _script(vad, [0.9, 0.1, 0.9, 0.1, 0.1])
    assert vad.is_speech(_frame(vad)) is True  # triggers speech
    assert vad.is_speech(_frame(vad)) is True  # 1 silence frame (countdown started)
    assert vad.is_speech(_frame(vad)) is True  # speech resumes -> countdown cancelled
    assert vad.is_speech(_frame(vad)) is True  # 1 silence frame (fresh countdown)
    assert vad.is_speech(_frame(vad)) is False  # 2nd consecutive silence frame -> ends


def test_reset_clears_debounce_state():
    vad = _make_vad(min_speech_duration_ms=32.0)
    _script(vad, [0.9])
    assert vad.is_speech(_frame(vad)) is True

    vad.reset()

    assert vad._speaking is False
    assert vad._speech_accum_ms == 0.0
    assert vad._silence_accum_ms is None


def test_real_silero_backend_smoke():
    """Optional real end-to-end check: loads the actual Silero model and
    classifies one silent frame. Requires the `vad-silero` extra; skipped
    (not failed) if it's not installed or the model can't be loaded."""
    pytest.importorskip("torch")
    pytest.importorskip("silero_vad")

    vad = SileroVAD()
    silence = b"\x00\x00" * vad.frame_samples
    try:
        result = vad.is_speech(silence)
    except Exception as exc:  # model download/load failure, etc.
        pytest.skip(f"Silero VAD model unavailable in this environment: {exc}")

    assert isinstance(result, bool)
