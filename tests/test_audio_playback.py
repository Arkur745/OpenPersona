"""Tests for companion/audio/playback.py.

Honest scope, per the task: this tests SpeakerPlayback's *logic* (the
queue/thread bridge, the pending-count "drained" tracking, stop()'s
discard-vs-in-flight distinction) against a fake `sounddevice` module
injected into sys.modules — not against a real speaker. Nothing here
proves played-back audio actually sounds correct; that was verified by
hand against a real device (see the session summary), including timing
a real write() call at ~1.03s for 1.0s of audio, which is what this
module's "drained" tracking design leans on. Injecting a fake module
means these tests run without the `audio-sounddevice` extra installed,
matching every other optional-backend test file's offline-tier pattern.
"""

import asyncio
import sys
import threading
import types
from typing import ClassVar

import pytest

from companion.audio.errors import AudioDeviceError


class _FakePortAudioError(Exception):
    pass


class _FakeRawOutputStream:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.written = []
        self.started = False
        self.stopped = False
        self.closed = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def write(self, data):
        self.written.append(data)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class _BlockingFakeRawOutputStream(_FakeRawOutputStream):
    """write() blocks until the test explicitly releases it — lets tests
    deterministically catch the worker thread mid-write, to distinguish
    "already in-flight, can't be stopped" from "still queued, can be
    discarded" (see SpeakerPlayback.stop()'s documented contract)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.write_started = threading.Event()
        self.release = threading.Event()

    def write(self, data):
        self.write_started.set()
        self.release.wait(timeout=2.0)
        self.written.append(data)


def _install_fake_sounddevice(
    monkeypatch, *, raise_on_open: Exception | None = None, stream_cls=_FakeRawOutputStream
):
    stream_cls.instances = []
    module = types.ModuleType("sounddevice")
    module.PortAudioError = _FakePortAudioError

    def _raw_output_stream(**kwargs):
        if raise_on_open is not None:
            raise raise_on_open
        return stream_cls(**kwargs)

    module.RawOutputStream = _raw_output_stream
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


@pytest.mark.asyncio
async def test_portaudio_error_wrapped_as_audio_device_error(monkeypatch):
    _install_fake_sounddevice(
        monkeypatch, raise_on_open=_FakePortAudioError("Invalid sample rate")
    )
    from companion.audio.playback import SpeakerPlayback

    with pytest.raises(AudioDeviceError, match="Could not open speaker output"):
        SpeakerPlayback(sample_rate=999999)


@pytest.mark.asyncio
async def test_stream_constructed_with_configured_parameters(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.playback import SpeakerPlayback

    playback = SpeakerPlayback(sample_rate=16000, device="Headphones", channels=1)
    playback.close()

    stream = _FakeRawOutputStream.instances[0]
    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["device"] == "Headphones"
    assert stream.kwargs["dtype"] == "int16"
    assert stream.started is True


@pytest.mark.asyncio
async def test_play_writes_chunk_and_becomes_drained(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.playback import SpeakerPlayback

    playback = SpeakerPlayback(sample_rate=16000)
    await playback.play(b"\x01\x02" * 100)

    await asyncio.wait_for(playback.wait_drained(), timeout=2.0)

    stream = _FakeRawOutputStream.instances[0]
    assert stream.written == [b"\x01\x02" * 100]
    playback.close()


@pytest.mark.asyncio
async def test_starts_drained_with_nothing_queued(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.playback import SpeakerPlayback

    playback = SpeakerPlayback(sample_rate=16000)
    # Should resolve immediately — nothing was ever queued.
    await asyncio.wait_for(playback.wait_drained(), timeout=0.5)
    playback.close()


@pytest.mark.asyncio
async def test_stop_discards_queued_chunks_but_not_one_already_in_flight(monkeypatch):
    """Pins the documented contract exactly: stop() can discard chunks
    still sitting in the queue, but not one the worker thread has already
    started writing (write() isn't cancellable mid-call)."""
    _install_fake_sounddevice(monkeypatch, stream_cls=_BlockingFakeRawOutputStream)
    from companion.audio.playback import SpeakerPlayback

    playback = SpeakerPlayback(sample_rate=16000)
    stream = _BlockingFakeRawOutputStream.instances[0]

    await playback.play(b"chunk1")
    # Wait until the worker thread is actually blocked inside write(chunk1).
    await asyncio.to_thread(stream.write_started.wait, 2.0)

    await playback.play(b"chunk2")
    await playback.play(b"chunk3")  # both still sitting in the queue

    playback.stop()  # should discard chunk2/chunk3

    stream.release.set()  # let the in-flight write(chunk1) complete
    await asyncio.wait_for(playback.wait_drained(), timeout=2.0)

    assert stream.written == [b"chunk1"]
    playback.close()


@pytest.mark.asyncio
async def test_stop_with_nothing_queued_stays_drained(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.playback import SpeakerPlayback

    playback = SpeakerPlayback(sample_rate=16000)
    playback.stop()  # should not raise or hang

    await asyncio.wait_for(playback.wait_drained(), timeout=0.5)
    playback.close()


@pytest.mark.asyncio
async def test_close_stops_worker_thread_and_stream(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    from companion.audio.playback import SpeakerPlayback

    playback = SpeakerPlayback(sample_rate=16000)
    stream = _FakeRawOutputStream.instances[0]

    playback.close()

    assert stream.stopped is True
    assert stream.closed is True


@pytest.mark.asyncio
async def test_real_speaker_playback_smoke():
    """Optional real end-to-end check: opens the actual default output
    device and plays ~0.3s of silence through it. Requires the
    `audio-sounddevice` extra; skipped (not failed) if it's not installed
    or no output device is available (e.g. headless/CI) — same pattern
    as every other real-backend smoke test in this repo.

    This confirms real playback *runs* (a real device opens, write()
    completes without error) — it does NOT verify anything audible came
    out correctly; that was checked by hand with a real tone and timed
    write() latency (see the session summary), not by this test.
    """
    pytest.importorskip("sounddevice")

    from companion.audio.playback import SpeakerPlayback

    try:
        playback = SpeakerPlayback(sample_rate=16000)
    except Exception as exc:  # no output device available, permissions, etc.
        pytest.skip(f"Real speaker output unavailable in this environment: {exc}")

    try:
        silence = b"\x00\x00" * 4800  # 0.3s @ 16kHz mono int16
        await playback.play(silence)
        await asyncio.wait_for(playback.wait_drained(), timeout=3.0)
    finally:
        playback.close()
    assert not playback._worker.is_alive()
