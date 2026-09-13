"""Tests for companion/voice_app.py.

Honest scope, per the task: proves the *wiring* — that run_voice_loop
passes playback.stop/playback.wait_drained into
handle_voice_turn's on_playback_interrupt/wait_for_playback_drained
correctly and plays every yielded chunk, that build_mic_capture/
build_speaker_playback enforce the sample-rate-matches-vad and
unimplemented-backend checks, and that run_voice_session cleans up
devices it opened even when a later step fails. Uses a duck-typed
recording stand-in for Orchestrator (not a real one) for the pure-wiring
test, since handle_voice_turn's own state-machine correctness is already
exhaustively covered in tests/test_orchestrator_voice_turn.py — this
file isn't re-proving that. Real device behavior (does captured/played
audio actually work) is not tested here at all — see
tests/test_audio_capture.py/test_audio_playback.py's honest-scope notes
and the session summary for what was verified by hand against real
hardware instead.
"""

import sys
import types
from typing import ClassVar

import pytest

from companion.config.loader import AppConfig, BackendConfig, TTSConfig
from companion.core.interfaces.types import Persona
from companion.voice_app import (
    VoiceSetupError,
    build_mic_capture,
    build_speaker_playback,
    run_voice_loop,
    run_voice_session,
)


def _persona() -> Persona:
    return Persona(
        id="mentor",
        name="The Old Mentor",
        tone="Calm.",
        boundaries="Not a therapist.",
        voice_id="piper:en_US-mentor-medium",
    )


def _config(**overrides) -> AppConfig:
    defaults = {
        "personas_dir": "./companion/personas",
        "stt": BackendConfig(backend="faster_whisper", options={}),
        "tts": TTSConfig(backend="piper", voices_dir="/tmp/voices", options={}),
        "llm": BackendConfig(backend="ollama", options={"model": "x"}),
        "vad": BackendConfig(backend="silero", options={"sample_rate": 16000}),
        "memory": BackendConfig(backend="chroma", options={"persist_dir": "/tmp/mem"}),
        "audio": BackendConfig(
            backend="sounddevice",
            options={"sample_rate": 16000, "input_device": None, "output_device": None},
        ),
        "mode": "voice",
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


# --- Sample-rate cross-check + backend rejection ----------------------------


def test_build_mic_capture_rejects_mismatched_sample_rate():
    config = _config(
        audio=BackendConfig(backend="sounddevice", options={"sample_rate": 44100}),
        vad=BackendConfig(backend="silero", options={"sample_rate": 16000}),
    )
    with pytest.raises(VoiceSetupError, match="must match"):
        build_mic_capture(config)


def test_build_speaker_playback_rejects_mismatched_sample_rate():
    config = _config(
        audio=BackendConfig(backend="sounddevice", options={"sample_rate": 8000}),
        vad=BackendConfig(backend="silero", options={"sample_rate": 16000}),
    )
    with pytest.raises(VoiceSetupError, match="must match"):
        build_speaker_playback(config)


def test_matches_when_vad_sample_rate_uses_its_own_default():
    # vad.options doesn't set sample_rate explicitly -> SileroVAD's own
    # 16000 default applies; audio.options.sample_rate=16000 should match.
    config = _config(
        audio=BackendConfig(backend="sounddevice", options={"sample_rate": 16000}),
        vad=BackendConfig(backend="silero", options={}),
    )
    # Should not raise past the sample-rate check; will fail later trying
    # to actually open a device (no fake sounddevice installed here) —
    # that's fine, this test only cares about the rate check itself.
    with pytest.raises(Exception) as exc_info:
        build_mic_capture(config)
    assert "must match" not in str(exc_info.value)


def test_build_mic_capture_rejects_unimplemented_audio_backend():
    config = _config(audio=BackendConfig(backend="pyaudio", options={"sample_rate": 16000}))
    with pytest.raises(VoiceSetupError, match="pyaudio"):
        build_mic_capture(config)


def test_build_speaker_playback_rejects_unimplemented_audio_backend():
    config = _config(audio=BackendConfig(backend="pyaudio", options={"sample_rate": 16000}))
    with pytest.raises(VoiceSetupError, match="pyaudio"):
        build_speaker_playback(config)


def test_missing_sample_rate_raises():
    config = _config(audio=BackendConfig(backend="sounddevice", options={}))
    with pytest.raises(VoiceSetupError, match="sample_rate"):
        build_mic_capture(config)


# --- build_mic_capture/build_speaker_playback forward config correctly -----


class _FakePortAudioError(Exception):
    pass


def _install_fake_sounddevice(monkeypatch):
    class _FakeStream:
        instances: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _FakeStream.instances.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def write(self, data):
            pass

    _FakeStream.instances = []
    module = types.ModuleType("sounddevice")
    module.PortAudioError = _FakePortAudioError
    module.RawInputStream = lambda callback=None, **kwargs: _FakeStream(
        callback=callback, **kwargs
    )
    module.RawOutputStream = lambda **kwargs: _FakeStream(**kwargs)
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return _FakeStream


@pytest.mark.asyncio
async def test_build_mic_capture_forwards_configured_device_and_rate(monkeypatch):
    fake_stream_cls = _install_fake_sounddevice(monkeypatch)
    config = _config(
        audio=BackendConfig(
            backend="sounddevice",
            options={"sample_rate": 16000, "input_device": 3, "channels": 1},
        )
    )

    mic = build_mic_capture(config)

    stream = fake_stream_cls.instances[0]
    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["device"] == 3
    mic.close()


@pytest.mark.asyncio
async def test_build_speaker_playback_forwards_configured_device_and_rate(monkeypatch):
    fake_stream_cls = _install_fake_sounddevice(monkeypatch)
    config = _config(
        audio=BackendConfig(
            backend="sounddevice",
            options={"sample_rate": 16000, "output_device": "Headphones"},
        )
    )

    playback = build_speaker_playback(config)

    stream = fake_stream_cls.instances[0]
    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["device"] == "Headphones"
    playback.close()


# --- run_voice_loop: wiring, via a duck-typed recording Orchestrator -------


class _RecordingOrchestrator:
    """Not a real Orchestrator — deliberately, since this file tests
    run_voice_loop's wiring, not handle_voice_turn's state machine
    (already covered in tests/test_orchestrator_voice_turn.py)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.call_kwargs: dict | None = None

    async def handle_voice_turn(
        self, user_id, persona, mic_audio, *, on_playback_interrupt=None,
        wait_for_playback_drained=None,
    ):
        self.call_kwargs = {
            "user_id": user_id,
            "persona": persona,
            "mic_audio": mic_audio,
            "on_playback_interrupt": on_playback_interrupt,
            "wait_for_playback_drained": wait_for_playback_drained,
        }
        for chunk in self._chunks:
            yield chunk


class _FakePlayback:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.stop_calls = 0

    async def play(self, chunk: bytes) -> None:
        self.played.append(chunk)

    def stop(self) -> None:
        self.stop_calls += 1

    async def wait_drained(self) -> None:
        return None


@pytest.mark.asyncio
async def test_run_voice_loop_plays_every_yielded_chunk():
    orchestrator = _RecordingOrchestrator([b"chunk1", b"chunk2", b"chunk3"])
    playback = _FakePlayback()
    mic = object()  # never touched directly by run_voice_loop itself

    await run_voice_loop(
        orchestrator, _persona(), "u1", mic, playback, print_fn=lambda _: None
    )

    assert playback.played == [b"chunk1", b"chunk2", b"chunk3"]


@pytest.mark.asyncio
async def test_run_voice_loop_wires_playback_hooks_into_handle_voice_turn():
    orchestrator = _RecordingOrchestrator([])
    playback = _FakePlayback()
    mic = object()

    await run_voice_loop(
        orchestrator, _persona(), "u1", mic, playback, print_fn=lambda _: None
    )

    # Bound methods aren't identity-stable across separate attribute
    # accesses (each `obj.method` access creates a new bound-method
    # wrapper) — `==` compares __self__/__func__, which is what actually
    # matters here: are these the same underlying method on the same
    # object.
    assert orchestrator.call_kwargs["on_playback_interrupt"] == playback.stop
    assert orchestrator.call_kwargs["wait_for_playback_drained"] == playback.wait_drained
    assert orchestrator.call_kwargs["mic_audio"] is mic
    assert orchestrator.call_kwargs["user_id"] == "u1"


# --- run_voice_session: cleanup guarantees -----------------------------


@pytest.mark.asyncio
async def test_run_voice_session_closes_mic_even_if_playback_construction_fails(
    monkeypatch,
):
    class _FakeStream:
        instances: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.stopped = False
            self.closed = False
            _FakeStream.instances.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    _FakeStream.instances = []
    module = types.ModuleType("sounddevice")
    module.PortAudioError = _FakePortAudioError
    module.RawInputStream = lambda callback=None, **kwargs: _FakeStream(
        callback=callback, **kwargs
    )

    def _raise_output(**kwargs):
        raise _FakePortAudioError("no output device")

    module.RawOutputStream = _raise_output
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    config = _config()
    orchestrator = _RecordingOrchestrator([])

    with pytest.raises(Exception, match="Could not open speaker output"):
        await run_voice_session(config, orchestrator, _persona(), "u1", print_fn=lambda _: None)

    mic_stream = _FakeStream.instances[0]
    assert mic_stream.stopped is True
    assert mic_stream.closed is True
