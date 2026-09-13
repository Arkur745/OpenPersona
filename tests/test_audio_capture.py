"""Tests for companion/audio/capture.py.

Honest scope, per the task: this tests MicCapture's *logic* (callback ->
asyncio.Queue bridging, the AsyncIterator protocol, error wrapping)
against a fake `sounddevice` module injected into sys.modules — not
against a real microphone. Nothing here proves captured audio is
correct/intelligible; that was verified by hand against a real device
(see the session summary). Injecting a fake module (rather than
monkeypatching attributes on the real package) also means these tests
run without the `audio-sounddevice` extra installed at all, matching
every other optional-backend test file's offline-tier pattern.
"""

import asyncio
import sys
import types
from typing import ClassVar

import pytest

from companion.audio.errors import AudioDeviceError


class _FakePortAudioError(Exception):
    pass


class _FakeRawInputStream:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.stopped = False
        self.closed = False
        _FakeRawInputStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _install_fake_sounddevice(monkeypatch, *, raise_on_open: Exception | None = None):
    _FakeRawInputStream.instances = []
    module = types.ModuleType("sounddevice")
    module.PortAudioError = _FakePortAudioError

    def _raw_input_stream(**kwargs):
        if raise_on_open is not None:
            raise raise_on_open
        return _FakeRawInputStream(**kwargs)

    module.RawInputStream = _raw_input_stream
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


@pytest.mark.asyncio
async def test_portaudio_error_wrapped_as_audio_device_error(monkeypatch):
    _install_fake_sounddevice(
        monkeypatch, raise_on_open=_FakePortAudioError("Invalid device")
    )
    from companion.audio.capture import MicCapture

    with pytest.raises(AudioDeviceError, match="Could not open microphone"):
        MicCapture(sample_rate=16000, device=9999)


@pytest.mark.asyncio
async def test_stream_constructed_with_configured_parameters(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.capture import MicCapture

    MicCapture(sample_rate=16000, device="Built-in Mic", channels=1, blocksize=800)

    stream = _FakeRawInputStream.instances[0]
    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["device"] == "Built-in Mic"
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["blocksize"] == 800
    assert stream.started is True


@pytest.mark.asyncio
async def test_callback_delivers_chunks_through_async_iteration(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.capture import MicCapture

    capture = MicCapture(sample_rate=16000)
    stream = _FakeRawInputStream.instances[0]

    stream.callback(b"\x01\x02\x03\x04", 2, None, None)
    stream.callback(b"\x05\x06\x07\x08", 2, None, None)

    first = await capture.__anext__()
    second = await capture.__anext__()

    assert first == b"\x01\x02\x03\x04"
    assert second == b"\x05\x06\x07\x08"


@pytest.mark.asyncio
async def test_supports_async_for_protocol(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.capture import MicCapture

    capture = MicCapture(sample_rate=16000)
    assert capture.__aiter__() is capture


@pytest.mark.asyncio
async def test_close_stops_and_closes_the_stream(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.capture import MicCapture

    capture = MicCapture(sample_rate=16000)
    stream = _FakeRawInputStream.instances[0]

    capture.close()

    assert stream.stopped is True
    assert stream.closed is True


@pytest.mark.asyncio
async def test_real_mic_capture_smoke():
    """Optional real end-to-end check: opens the actual default input
    device and captures ~0.5s of real audio. Requires the
    `audio-sounddevice` extra; skipped (not failed) if it's not installed
    or no input device is available (e.g. headless/CI) — same pattern as
    every other real-backend smoke test in this repo.

    This confirms real capture *runs* (a real device opens, real bytes of
    the right size/format come back) — it does NOT verify the captured
    audio is meaningful/intelligible; that was checked by hand (see the
    session summary), not by this test.
    """
    pytest.importorskip("sounddevice")

    from companion.audio.capture import MicCapture

    try:
        capture = MicCapture(sample_rate=16000)
    except Exception as exc:  # no input device available, permissions, etc.
        pytest.skip(f"Real microphone unavailable in this environment: {exc}")

    try:
        chunk = await asyncio.wait_for(capture.__anext__(), timeout=3.0)
    finally:
        capture.close()

    assert isinstance(chunk, bytes)
    assert len(chunk) > 0
    assert len(chunk) % 2 == 0  # whole number of int16 samples
