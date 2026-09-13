"""Local TTS backend: Coqui TTS / XTTS-v2 (optional higher-fidelity path).

Higher quality than Piper, and (via XTTS-v2 specifically) supports voice
cloning from a short reference clip — see "Voice cloning" below for why
that's not wired up in this pass even though the plumbing is ready for it.

Uses `coqui-tts` (https://github.com/idiap/coqui-ai-TTS), not the original
`TTS` PyPI package — Coqui the company shut down in Jan 2024 and
`coqui-ai/TTS` has had no release since 0.22 (Nov 2023). `coqui-tts` is an
actively maintained community fork, API-compatible (still imports as
`TTS`). See THIRD_PARTY_LICENSES.md — this matters more than a routine
dependency swap: XTTS-v2's *model weights* carry Coqui's own non-commercial
license (CPML) regardless of which Python package loads them.

If `from TTS.api import TTS` raises `ImportError: cannot import name
'isin_mps_friendly'`, that's not a bug here — coqui-tts 0.27.5 declares
`transformers>=4.57` with no upper bound, and unconstrained-latest
`transformers` (5.x as of writing) removed that name. `pyproject.toml`'s
`tts-coqui` extra pins `transformers<5` specifically to avoid this; if
you're seeing it anyway, something re-resolved transformers outside that
pin.

## Streaming: sentence-granularity, not sub-sentence

`TTS.api.TTS` (the high-level class used here) has no incremental/streaming
synthesis API — `.tts()` is one blocking call that returns the complete
waveform for whatever text it's given, internally splitting multi-sentence
input into sentences and *concatenating* the result before returning (see
its `split_sentences` param). There's a lower-level streaming API
(`Xtts.inference_stream()`) but it lives on the raw XTTS model class,
bypasses `TTS.api.TTS` entirely, and only exists for XTTS — not a general
Coqui-model streaming API. Using it would mean two different code paths
for XTTS vs. every other Coqui model this backend could otherwise load,
which is worse than the alternative below.

So "streaming" here means the same thing it does for the interface
contract as a whole: `synthesize_stream()` is called once per `text_chunk`
as the orchestrator produces one (typically one sentence, same as Piper —
see `piper_local.py`), so synthesis starts on sentence N without waiting
for the LLM to finish generating sentence N+1. That's the actual latency
win the interface exists for. What Coqui does *not* give us for free that
Piper does is multiple `AudioChunk`s out of a single multi-sentence
`text_chunk` — a multi-sentence `text_chunk` here comes back as one
already-concatenated waveform, which this backend then slices into
fixed-size byte chunks (`chunk_frames`) purely so a downstream audio
player still receives a stream of reasonably sized pieces instead of one
large blob — that slicing doesn't reduce this call's synthesis latency,
it only smooths out how the already-computed audio is handed off.

This doesn't require any change to the abstract `TextToSpeech` interface —
worth stating explicitly since that was the flag condition for this pass.

## Voice cloning (not implemented, shape is ready for it)

`speaker_wav` is already a constructor parameter (unused by
`from_voice_id` in this pass) — the natural extension is
`CoquiTTS.from_voice_id` treating `coqui:<label>` as a reference clip
filename (`<voices_dir>/<label>.wav`) instead of a built-in speaker name,
and passing it as `speaker_wav=` instead of `speaker=`. That needs no
change to `TextToSpeech`, `Persona`, or personas/schema.md — `voice_id` is
already an opaque backend-namespaced string with no assumption baked in
about what's on the other side of the colon. What it *does* need before
being wired up for real: a consent-tracking story for whose voice is being
cloned (see personas/schema.md's & docs/PERSONA_AUTHORING.md's existing
note on this) — that's a product/policy decision, not a technical one,
which is why this pass stops at "the shape supports it."

Install: pip install "companion[tts-coqui]". Some stock Coqui models
additionally require the `espeak-ng` system binary (not a Python package —
install via your OS package manager) for phonemization; XTTS-v2 does not.
"""

import asyncio
import threading
from typing import AsyncIterator

import numpy as np

from companion.backends.tts._sync_stream import iter_blocking_generator
from companion.core.interfaces.text_to_speech import TextToSpeech

_DEFAULT_CHUNK_FRAMES = 4096  # ~0.19s at 22050 Hz — see chunk_frames below


class CoquiTTS(TextToSpeech):
    def __init__(
        self,
        model_name: str,
        speaker: str | None = None,
        speaker_wav: str | None = None,
        language: str | None = None,
        use_gpu: bool = False,
        sample_rate: int | None = None,
        chunk_frames: int = _DEFAULT_CHUNK_FRAMES,
    ) -> None:
        self.model_name = model_name
        self.speaker = speaker
        self.speaker_wav = speaker_wav  # reference audio for voice cloning, if set
        self.language = language
        self.use_gpu = use_gpu
        # Validated against the loaded model's actual rate on first use,
        # same pattern as PiperTTS — see _load_model.
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames

        self._model = None  # loaded lazily — see _ensure_model
        self._stop_event = threading.Event()

    @classmethod
    def from_voice_id(cls, voice_id: str, model_name: str, **kwargs) -> "CoquiTTS":
        """Resolve a persona's `voice_id` (e.g. "coqui:speaker-42") to a
        built-in speaker name on `model_name`. Raises ValueError if
        `voice_id` isn't namespaced for this backend.

        This does not do voice-cloning resolution (`speaker_wav`) — see
        the module docstring's "Voice cloning" section for why, and for
        the shape this would take if/when that's wired up.
        """
        backend, sep, speaker_name = voice_id.partition(":")
        if not sep or backend != "coqui":
            raise ValueError(
                f"voice_id {voice_id!r} is not a coqui: voice "
                '(expected "coqui:<speaker-name>")'
            )
        return cls(model_name=model_name, speaker=speaker_name, **kwargs)

    async def _ensure_model(self):
        if self._model is None:
            self._model = await asyncio.to_thread(self._load_model)
        return self._model

    def _load_model(self):
        from TTS.api import TTS as CoquiTTSApi

        model = CoquiTTSApi(
            model_name=self.model_name, progress_bar=False, gpu=self.use_gpu
        )
        actual_rate = model.synthesizer.output_sample_rate
        if self.sample_rate is not None and self.sample_rate != actual_rate:
            raise ValueError(
                f"configured sample_rate={self.sample_rate} doesn't match "
                f"the loaded model's native rate={actual_rate} "
                f"({self.model_name}) — drop the override and let it be "
                "inferred, or fix the mismatch."
            )
        self.sample_rate = actual_rate
        return model

    def _run_model(self, model, text_chunk: str) -> bytes:
        wav = model.tts(
            text=text_chunk,
            speaker=self.speaker,
            language=self.language,
            speaker_wav=self.speaker_wav,
        )
        # .tts() return type varies by model (list[float] / np.ndarray /
        # torch.Tensor have all been observed across Coqui model types) —
        # normalize rather than assume.
        samples = np.asarray(wav, dtype=np.float32)
        pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
        return pcm16.tobytes()

    async def synthesize_stream(self, text_chunk: str) -> AsyncIterator[bytes]:
        self._stop_event.clear()
        model = await self._ensure_model()

        def _generate():
            pcm_bytes = self._run_model(model, text_chunk)
            frame_bytes = self.chunk_frames * 2  # 16-bit samples
            for start in range(0, len(pcm_bytes), frame_bytes):
                yield pcm_bytes[start : start + frame_bytes]

        async for pcm_bytes in iter_blocking_generator(_generate, self._stop_event):
            yield pcm_bytes

    def stop(self) -> None:
        self._stop_event.set()
