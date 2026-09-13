"""Local VAD backend: Silero VAD.

## A hard constraint, not a preference: fixed-size frames only

Verified directly against the real model (not just docs): Silero VAD's
model is a TorchScript module compiled for an exact input length —
`model(x, sample_rate)` raises a TorchScript runtime error for any `x`
that isn't exactly 512 samples at 16 kHz (256 at 8 kHz). This isn't a
recommended/configurable window — `get_speech_timestamps`' own
`window_size_samples` parameter is marked `!!! DEPRECATED, DOES NOTHING
!!!` in Silero's own docstring; the reference implementation hardcodes
512/256 internally regardless of what's passed.

That's a real, worth-flagging divergence from `SpeechToText`, which
explicitly accepts "chunk size decided by the audio capture layer" (it
buffers internally). `VoiceActivityDetector.is_speech(chunk) -> bool` has
no such buffering escape hatch — every call must return a real answer,
and there's no clean way to buffer arbitrary input into exact frames
*inside* `is_speech()` without either delaying the returned bool (stale
answer for a partial frame) or silently resampling/re-chunking behind the
caller's back, which is exactly what was asked not to do here.

Resolution (not an abstract-interface change): `SileroVAD.is_speech()`
validates its input is exactly `frame_bytes` (a documented, discoverable
property) and raises `ValueError` otherwise — same "fail loudly, not
silently" posture used elsewhere in this codebase. The caller (here,
`Orchestrator`'s voice-turn loop — see that module) owns re-chunking
whatever size the mic-capture layer produces into exact VAD frames; it
already needs to buffer independently for STT's own re-transcription
cadence, so this isn't new complexity, just a second, differently-sized
buffer. 512 samples (32ms) at 16 kHz is also a perfectly fine input size
for `FasterWhisperSTT` (which accepts anything), so standardizing
mic-capture chunking on that value satisfies both backends — see
docs/ARCHITECTURE.md's "VAD" section.

## Debouncing: closer to Silero's own VADIterator than a naive threshold

A raw `speech_prob >= threshold` per 32ms frame flickers — real speech
has brief sub-threshold dips (between syllables, etc.) that would
otherwise register as spurious end-of-utterance. Silero's own streaming
helper (`VADIterator`) handles this with asymmetric hysteresis: entering
"speech" requires sustained speech frames, but once in a speech region, a
single frame dipping back below threshold does NOT immediately start
counting toward silence — only *sustained* silence does, and that
countdown cancels entirely (not just pauses) if speech resumes before it
completes. `VADIterator` itself wasn't reused because its API returns
start/end event dicts (`{"start": ...}` / `{"end": ...}` / `None`), which
doesn't fit `is_speech() -> bool`'s contract — but the same debounce
*logic* is reimplemented here against `min_speech_duration_ms` /
`min_silence_duration_ms`, verified against `VADIterator`'s source rather
than guessed at, so `is_speech()`'s returned bool has the same "don't
flicker on a brief dip" behavior Silero's reference implementation has.

Install: pip install "companion[vad-silero]".

`torch` (needed to decode audio and run the model) is imported lazily,
inside methods, not at module level — same reason as `faster_whisper`/
`piper`/`ollama`/`TTS` being deferred in the other backends: importing
this module (e.g. to test frame-size validation or the debounce state
machine against a fake model) shouldn't require torch installed at all.
Unlike `numpy` (STT/TTS) or `ollama` (LLM), torch is genuinely heavy, so
it's *not* in the `dev` extra — offline tests mock at `_speech_probability`
instead of `_model`, so the real decode path (which does need torch) is
only exercised by the real-model smoke test, gated on `vad-silero`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from companion.core.interfaces.vad import VoiceActivityDetector

if TYPE_CHECKING:
    import torch

_MAX_INT16 = 32768.0


def _pcm16_bytes_to_float32(chunk: bytes) -> "torch.Tensor":
    """Decode mono 16-bit signed little-endian PCM bytes to the float32
    [-1, 1] tensor Silero's model expects (same convention as
    torchaudio.load(), which is what Silero's own read_audio() uses).

    `bytearray(chunk)` (not `chunk` directly) — torch.frombuffer on a plain
    `bytes` object works, but warns on every single call that the buffer
    isn't writable (it doesn't matter here; nothing writes back into it),
    and this runs once per 32ms frame, so that warning would spam any real
    session's logs.
    """
    import torch

    samples = torch.frombuffer(bytearray(chunk), dtype=torch.int16)
    return samples.float() / _MAX_INT16


class SileroVAD(VoiceActivityDetector):
    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_speech_duration_ms: float = 250.0,
        min_silence_duration_ms: float = 100.0,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"Silero VAD supports sample_rate 8000 or 16000, got {sample_rate!r}"
            )
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # Hard model constraint (see module docstring) — not configurable.
        self.frame_samples = 512 if sample_rate == 16000 else 256
        self.frame_bytes = self.frame_samples * 2  # int16
        self._frame_duration_ms = self.frame_samples / sample_rate * 1000.0

        self._model = None  # loaded lazily — see _ensure_model

        self._speaking = False
        self._speech_accum_ms = 0.0
        self._silence_accum_ms: float | None = None  # None = no pending silence countdown

    def _ensure_model(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    def _speech_probability(self, audio_chunk: bytes) -> float:
        import torch

        model = self._ensure_model()
        samples = _pcm16_bytes_to_float32(audio_chunk)
        with torch.no_grad():
            return model(samples, self.sample_rate).item()

    def is_speech(self, audio_chunk: bytes) -> bool:
        if len(audio_chunk) != self.frame_bytes:
            raise ValueError(
                f"SileroVAD requires exactly {self.frame_bytes}-byte frames "
                f"({self.frame_samples} samples @ {self.sample_rate} Hz = "
                f"{self._frame_duration_ms:.0f}ms) — got {len(audio_chunk)} bytes. "
                "This is a hard constraint of Silero's model, not a tunable "
                "option — see this module's docstring."
            )

        prob = self._speech_probability(audio_chunk)
        frame_is_speech = prob >= self.threshold

        if frame_is_speech:
            # Speech observed: cancel any pending end-of-utterance
            # countdown outright (a brief dip recovering is not silence).
            self._silence_accum_ms = None
            if not self._speaking:
                self._speech_accum_ms += self._frame_duration_ms
                if self._speech_accum_ms >= self.min_speech_duration_ms:
                    self._speaking = True
                    self._speech_accum_ms = 0.0
        else:
            self._speech_accum_ms = 0.0  # onset needs consecutive speech frames
            if self._speaking:
                self._silence_accum_ms = (
                    self._frame_duration_ms
                    if self._silence_accum_ms is None
                    else self._silence_accum_ms + self._frame_duration_ms
                )
                if self._silence_accum_ms >= self.min_silence_duration_ms:
                    self._speaking = False
                    self._silence_accum_ms = None

        return self._speaking

    def reset(self) -> None:
        """Clear debounce state and the model's own short-term context.

        Not part of the `VoiceActivityDetector` interface (which has no
        `reset()` — unlike `SpeechToText`, nothing in this pass's state
        machine needs to reset VAD between utterances within one
        continuous session). Provided for hygiene: starting an unrelated
        audio stream, or test isolation.
        """
        self._speaking = False
        self._speech_accum_ms = 0.0
        self._silence_accum_ms = None
        if self._model is not None:
            self._model.reset_states()
