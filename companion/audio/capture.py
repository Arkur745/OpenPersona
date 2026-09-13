"""Real microphone capture -> the `AsyncIterator[bytes]`
`handle_voice_turn`'s `_pump_mic` already expects: mono 16kHz PCM16LE,
the same format contract `FasterWhisperSTT` and `SileroVAD` have assumed
since the STT/VAD passes (neither backend resamples — see
whisper_local.py's and silero_vad.py's module docstrings).

`MicCapture` doesn't need to know `SileroVAD`'s exact frame-size
requirement (512 samples @ 16kHz) — the orchestrator's own `_FrameBuffer`
already re-chunks whatever size this class happens to emit into VAD's
exact requirement, exactly as it already does for any
`AsyncIterator[bytes]` mic source. This class just needs to emit *some*
reasonably-sized chunks of the right format, promptly.

sounddevice's callback-based streaming (not its blocking `read()` API) is
used here so audio arrives as soon as PortAudio has it, without a
polling loop. The callback runs on PortAudio's own realtime thread, so it
does the minimum possible work — copy the buffer, hand it to the asyncio
loop via `call_soon_threadsafe` — and nothing else; anything heavier
belongs downstream (STT/VAD), not here.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from companion.audio.errors import AudioDeviceError

if TYPE_CHECKING:
    import sounddevice as sd

# 1600 samples @ 16kHz = 100ms per callback — small enough for low
# end-to-end latency, large enough not to spam the event loop with a
# call_soon_threadsafe on every few milliseconds of audio.
_DEFAULT_BLOCKSIZE = 1600


class MicCapture:
    """`AsyncIterator[bytes]` over a live microphone. Pass an instance of
    this directly as `handle_voice_turn`'s `mic_audio` argument.

    Runs until `close()` is called — there's no natural "end of stream"
    for a live mic, unlike a test fixture's finite frame list.
    """

    def __init__(
        self,
        sample_rate: int,
        device: int | str | None = None,
        channels: int = 1,
        blocksize: int = _DEFAULT_BLOCKSIZE,
    ) -> None:
        import sounddevice as sd

        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        try:
            self._stream: sd.RawInputStream = sd.RawInputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                device=device,
                blocksize=blocksize,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise AudioDeviceError(
                f"Could not open microphone (device={device!r}, "
                f"sample_rate={sample_rate}): {exc}. Run `python -m "
                "sounddevice` to list available input devices, or check "
                "audio.options.input_device in your config."
            ) from exc

    def _callback(self, indata, frames: int, time_info, status) -> None:
        # Realtime thread — copy and hand off immediately, nothing else.
        # `status` (e.g. input overflow) is deliberately not inspected
        # here to keep this callback minimal; a dropped/overflowed frame
        # just means a gap in the stream, not a crash.
        chunk = bytes(indata)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

    def __aiter__(self) -> MicCapture:
        return self

    async def __anext__(self) -> bytes:
        return await self._queue.get()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
