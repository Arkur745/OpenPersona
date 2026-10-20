"""Real speaker playback: consumes `handle_voice_turn`'s yielded PCM16LE
audio chunks and plays them through the configured output device.

## Why this is more than "write bytes to a stream"

`handle_voice_turn`'s barge-in watcher previously stopped watching as
soon as it finished *yielding* a response's audio chunks — a known,
previously-documented limitation (see docs/ARCHITECTURE.md's "Voice-turn
state machine" history): synthesis+yielding can outrun real-time
playback, so there was a window after yielding ended where audio was
still actually sounding but barge-in detection had already stopped. This
class tracks the gap between "handed a chunk" and "that chunk has
actually finished playing" so `handle_voice_turn`'s two new optional
hooks (`on_playback_interrupt`/`wait_for_playback_drained`) can close it
for real — see `companion/core/orchestrator/orchestrator.py`'s docstring
for the mechanism this class is designed to drive.

## Blocking write(), not a raw PortAudio callback, and why

sounddevice's callback-streaming API (used for capture — a mic just
*arrives* on its own schedule) would require producing exactly the right
amount of data on demand from a realtime thread, awkward to bridge
cleanly from an async producer that has no visibility into PortAudio's
internal timing. The blocking `RawOutputStream.write()` API is push-based
instead: it accepts a chunk and blocks until there's room for it in the
output buffer — verified directly against a real output device (timing a
1.0s chunk's `write()` call at ~1.03s wall-clock) — so it naturally paces
close to real-time on its own. A background thread does the blocking
writes, bridging it onto asyncio the same *spirit* as
`backends/tts/_sync_stream.py` bridges TTS's blocking synthesis calls,
though this is the inverse shape (async chunks arrive here and get
pushed to a thread that blocks on device I/O, rather than a blocking
generator being pulled from a thread into async code) — not reused
directly for that reason.

## What "drained" means, precisely (an honest limitation)

A `write()` call returning means the chunk has been accepted into
PortAudio's internal output buffer — not that it has *finished sounding*.
This class tracks "pending" as chunks handed to the write-thread but not
yet `write()`-returned, and reports "drained" once that count hits zero.
That leaves up to one output-buffer's worth of latency (bounded by the
stream's own buffering, typically tens of milliseconds) between "reported
drained" and "actually silent" — not literally zero, but a small, bounded
residual, nowhere near the multi-hundred-millisecond-to-seconds gap this
mechanism exists to close (the gap between finishing *synthesis* of a
whole multi-sentence response and finishing *playing* it).
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING

from companion.audio.errors import AudioDeviceError

if TYPE_CHECKING:
    import sounddevice as sd


logger = logging.getLogger(__name__)

# Extra time wait_drained() allows beyond the queued audio's own length.
DRAIN_MARGIN_S = 2.0

class SpeakerPlayback:
    """Consumes audio chunks (`await play(chunk)`), plays them through the
    configured output device on a background thread, and exposes the two
    hooks `handle_voice_turn` needs: `stop()` — discard whatever's queued
    but not yet handed to the device (called synchronously, from the
    barge-in watcher) — and `wait_drained()` — async-wait until nothing
    is pending (called before `handle_voice_turn` resumes listening). See
    module docstring for the "drained" precision caveat.
    """

    def __init__(
        self,
        sample_rate: int,
        device: int | str | None = None,
        channels: int = 1,
    ) -> None:
        import sounddevice as sd

        self.sample_rate = sample_rate
        self._bytes_per_second = sample_rate * 2 * channels  # int16 PCM
        self._loop = asyncio.get_running_loop()
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._pending = 0
        # Seconds of audio queued or being written; bounds wait_drained().
        self._queued_seconds = 0.0
        self._lock = threading.Lock()
        self._drained = asyncio.Event()
        self._drained.set()

        try:
            self._stream: sd.RawOutputStream = sd.RawOutputStream(
                samplerate=sample_rate, channels=channels, dtype="int16", device=device
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(
                f"Could not open speaker output (device={device!r}, "
                f"sample_rate={sample_rate}): {exc}. Run `python -m "
                "sounddevice` to list available output devices, or check "
                "audio.options.output_device in your config."
            ) from exc

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._stream.write(item)
            finally:
                self._mark_one_done(len(item))

    def _mark_one_done(self, nbytes: int = 0) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            self._queued_seconds = max(0.0, self._queued_seconds - nbytes / self._bytes_per_second)
            now_drained = self._pending == 0
        if now_drained:
            self._loop.call_soon_threadsafe(self._drained.set)

    async def play(self, chunk: bytes) -> None:
        with self._lock:
            self._pending += 1
            self._queued_seconds += len(chunk) / self._bytes_per_second
            self._drained.clear()
        self._queue.put(chunk)

    def stop(self) -> None:
        """Discard whatever's queued but not yet handed to the output
        device — called on barge-in. Cannot interrupt a single write()
        already in progress on the worker thread (PortAudio's blocking
        write isn't cancellable mid-call); that's at most one chunk's
        worth of residual audio, not the whole response."""
        now_drained = False
        with self._lock:
            try:
                while True:
                    discarded = self._queue.get_nowait()
                    self._pending = max(0, self._pending - 1)
                    if discarded:
                        self._queued_seconds = max(
                            0.0, self._queued_seconds - len(discarded) / self._bytes_per_second
                        )
            except queue.Empty:
                pass
            now_drained = self._pending == 0
        if now_drained:
            self._loop.call_soon_threadsafe(self._drained.set)

    async def wait_drained(self) -> None:
        """Wait until everything queued has been handed to the device, but
        never longer than that audio should take plus `DRAIN_MARGIN_S`.

        `RawOutputStream.write()` blocks, and if the output device stalls
        (seen on Windows; e.g. a Bluetooth headset switching profile when
        the mic opens) it may never return. An unbounded wait then froze
        the whole voice session: everything after went silent and it
        stopped listening. Bounded, a stall is logged and the conversation
        carries on listening."""
        with self._lock:
            budget = self._queued_seconds + DRAIN_MARGIN_S
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=budget)
        except TimeoutError:
            logger.warning(
                "speaker playback didn't finish within %.1fs (%.1fs of audio "
                "still queued); the output device may have stalled. Continuing "
                "to listen. Check audio.options.output_device, or try wired "
                "headphones.",
                budget, self._queued_seconds,
            )

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        self._stream.stop()
        self._stream.close()
