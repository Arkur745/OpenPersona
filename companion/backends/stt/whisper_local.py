"""Local STT backend: faster-whisper (GPU-friendly primary).

CPU-only deployments should configure `backends.stt.whisper_cpp_local`
instead (not yet implemented — same interface, whisper.cpp under the hood)
via config/config.example.yaml.

## Why this isn't a for-loop around batch `model.transcribe()` calls

faster-whisper (like every local Whisper variant) has no incremental
decoding API — there is no way to hand it "one more chunk" and get an
update cheaper than re-running attention over the buffered audio. Every
practical streaming wrapper (whisper_streaming's LocalAgreement policy,
RealtimeSTT) works around this the same way, and that's the pattern this
module adapts rather than reinventing:

1. Buffer incoming audio; only re-run the model once at least
   `min_refresh_seconds` of *new* audio has arrived (re-transcribing on
   every tiny chunk is wasted compute for no latency benefit).
2. Each re-transcription produces a full hypothesis for the whole buffer.
   Naively returning that every time is exactly the "flicker" problem
   these wrappers exist to solve: word choices near the end of the buffer
   are the least reliable and change between passes. This module borrows
   whisper_streaming's *LocalAgreement-2* policy: a word is only
   considered "committed" once it agrees between two consecutive
   transcription passes. Committed text is stable; the tail beyond it is
   tentative and still included in what's returned, just not guaranteed
   to survive the next pass.
3. `condition_on_previous_text=False` on every pass — with a
   repeatedly-extended buffer, letting Whisper condition on its own prior
   output is a known source of repetition/hallucination loops in these
   wrappers; re-deciding from the raw audio each time avoids that.

## What's deliberately not implemented here

whisper_streaming additionally trims the audio buffer at the audio
timestamp of the last committed word (via `word_timestamps=True`) so the
re-transcribed window doesn't grow unboundedly for a long utterance. That
requires precise word/audio alignment bookkeeping across trims; getting it
subtly wrong silently corrupts the commit logic. Instead, `max_buffer_seconds`
is a simple hard cap: past that, the oldest audio is dropped and the
commit/hypothesis state resets (equivalent to a fresh utterance starting
mid-buffer). For a companion app's turn-taking conversational utterances —
seconds, not minutes, before VAD end-of-utterance hands off to the LLM —
this is a reasonable bound, not a smarter aligned trim. Revisit if VAD
tuning (build-order step 6) allows unusually long utterances in practice.

Install: pip install "companion[stt-whisper]" (faster-whisper + numpy; a
CUDA build of ctranslate2 if you want GPU support, CPU works out of the box).
"""

import asyncio
from dataclasses import dataclass, field

import numpy as np

from companion.core.interfaces.speech_to_text import SpeechToText

# The whole local audio pipeline (VAD, STT) standardizes on mono 16 kHz
# 16-bit signed little-endian PCM — the format both Silero VAD and Whisper
# expect natively, so no per-backend resampling/conversion is needed
# upstream. Whatever produces `audio_chunk` bytes (mic capture, once
# implemented — build-order Section 3.1) must emit this format.
SAMPLE_RATE_HZ = 16_000
_BYTES_PER_SAMPLE = 2  # int16


def _pcm16_bytes_to_float32(chunk: bytes) -> np.ndarray:
    """Decode mono 16 kHz PCM16LE bytes to the float32 [-1, 1] array
    faster-whisper expects."""
    if len(chunk) % _BYTES_PER_SAMPLE != 0:
        raise ValueError(
            f"audio_chunk length ({len(chunk)} bytes) is not a whole number "
            "of 16-bit samples — expected mono PCM16LE."
        )
    samples = np.frombuffer(chunk, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def _resolve_device(device: str) -> str:
    """Expand config's `device: auto` into a concrete ctranslate2 device
    string. faster-whisper/ctranslate2 don't accept "auto" themselves."""
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        # No ctranslate2 CUDA support built, or detection failed for any
        # other reason — CPU always works, so fail safe rather than error.
        return "cpu"


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class _UtteranceState:
    buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    new_audio_sec: float = 0.0
    committed_words: list[str] = field(default_factory=list)
    previous_tail_words: list[str] = field(default_factory=list)
    last_returned_text: str | None = None


class FasterWhisperSTT(SpeechToText):
    """Model-size / device / latency trade-offs (see also
    docs/HARDWARE_REQUIREMENTS.md and docs/ARCHITECTURE.md):

    - `model_size`: larger models are more accurate, especially on accents,
      background noise, and domain vocabulary, but slower per pass. For a
      buffered-re-transcription streamer, model latency directly sets the
      floor on how fresh partial transcripts can be — "large-v3" on CPU
      will feel noticeably laggier than "small". Defaults to "small" as a
      middle ground; "tiny"/"base" trade accuracy for snappier partials on
      modest hardware, "medium"/"large-v3" want a GPU to stay responsive.
    - `device`: "auto" picks CUDA if ctranslate2 reports a usable GPU, else
      CPU. Pin explicitly to "cpu" to force CPU even on a GPU machine
      (e.g. to leave VRAM for the LLM backend).
    - `min_refresh_seconds`: lower = fresher partials, more compute spent
      re-transcribing the same audio repeatedly. 1.0s is a reasonable
      default; drop it for snappier UI feedback on fast hardware.
    - `max_buffer_seconds`: safety cap on a single utterance's buffer
      growth — see the module docstring for why this isn't a smarter
      word-aligned trim yet.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = None,
        beam_size: int = 5,
        min_refresh_seconds: float = 1.0,
        max_buffer_seconds: float = 30.0,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.min_refresh_seconds = min_refresh_seconds
        self.max_buffer_seconds = max_buffer_seconds

        self._model = None  # loaded lazily — see _ensure_model
        self._state = _UtteranceState()

    async def _ensure_model(self):
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            self.model_size,
            device=_resolve_device(self.device),
            compute_type=self.compute_type,
        )

    def _run_model(self, model, buffer: np.ndarray) -> str:
        segments, _info = model.transcribe(
            buffer,
            language=self.language,
            beam_size=self.beam_size,
            condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()

    async def transcribe_stream(self, audio_chunk: bytes) -> str | None:
        s = self._state
        s.buffer = np.concatenate([s.buffer, _pcm16_bytes_to_float32(audio_chunk)])
        s.new_audio_sec += len(audio_chunk) / _BYTES_PER_SAMPLE / SAMPLE_RATE_HZ

        max_samples = int(self.max_buffer_seconds * SAMPLE_RATE_HZ)
        if len(s.buffer) > max_samples:
            # Safety valve, not a smart trim — see module docstring. Drop
            # the oldest audio and start the commit/hypothesis state over,
            # since word alignment across the dropped span is unknown.
            s.buffer = s.buffer[-max_samples:]
            s.committed_words = []
            s.previous_tail_words = []

        if s.new_audio_sec < self.min_refresh_seconds:
            return None
        s.new_audio_sec = 0.0

        model = await self._ensure_model()
        hypothesis = await asyncio.to_thread(self._run_model, model, s.buffer)
        hypothesis_words = hypothesis.split()

        # LocalAgreement-2: only the words beyond what's already committed
        # are in play; a word there is promoted to committed once it
        # agrees with where the *previous* pass's tail put it.
        tail_now = hypothesis_words[len(s.committed_words):]
        agreed = _common_prefix_len(s.previous_tail_words, tail_now)
        if agreed:
            s.committed_words.extend(tail_now[:agreed])
            tail_now = hypothesis_words[len(s.committed_words):]
        s.previous_tail_words = tail_now

        # Built from committed_words (frozen at commit time) + the current
        # tentative tail, NOT from raw hypothesis_words — this is what
        # makes commitment actually stabilize the output: if this pass's
        # hypothesis revises a word inside the already-committed prefix
        # (Whisper changing its mind about earlier audio as more context
        # arrives), we keep the previously agreed-upon word rather than
        # flip-flopping on already-shown text.
        full_text = " ".join(s.committed_words + tail_now)
        if full_text and full_text != s.last_returned_text:
            s.last_returned_text = full_text
            return full_text
        return None

    def reset(self) -> None:
        self._state = _UtteranceState()
