"""Local TTS backend: Piper (default — low-latency, CPU-friendly).

## Streaming

`piper.PiperVoice.synthesize(text)` is already a generator that yields one
`AudioChunk` per *sentence* — Piper phonemizes the whole input up front,
then runs ONNX inference (the actual blocking work) sentence-by-sentence,
yielding each sentence's audio as soon as it's ready rather than waiting
for the whole input. Since the orchestrator already calls
`synthesize_stream()` once per sentence as the LLM streams (see
core/orchestrator/orchestrator.py), a `text_chunk` here is typically one
sentence and Piper yields exactly one `AudioChunk` for it — but nothing
here assumes that; a multi-sentence `text_chunk` streams multiple chunks
correctly for free.

That generator is synchronous and CPU-bound (ONNX Runtime inference), so
it's bridged onto a background thread via `_sync_stream.iter_blocking_generator`
rather than run directly on the event loop — see that module for why, and
for how `stop()` actually interrupts an in-progress synthesis (not just
gets ignored) via the shared `threading.Event`.

## Persona voice binding

`voice_id` in a persona config (see personas/schema.md) is
`piper:<voice-name>`, where `<voice-name>` follows Piper's own convention
(`<lang>_<region>-<name>-<quality>`, e.g. `en_US-lessac-medium` — see
https://github.com/rhasspy/piper/blob/master/VOICES.md for real stock
voices; this repo's starter personas use placeholder names that need a
matching .onnx file to actually exist). `PiperTTS.from_voice_id()` resolves
that into a concrete `<voices_dir>/<voice-name>.onnx` path — this is where
persona-to-voice resolution actually happens; the orchestrator/backend
itself never hardcodes a voice.

Install: pip install "companion[tts-piper]".
"""

import asyncio
import os
import threading
from typing import AsyncIterator

from companion.backends.tts._sync_stream import iter_blocking_generator
from companion.core.interfaces.text_to_speech import TextToSpeech


class PiperTTS(TextToSpeech):
    def __init__(
        self,
        voice_model_path: str,
        config_path: str | None = None,
        use_cuda: bool = False,
        speaker_id: int | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self.voice_model_path = voice_model_path
        self.config_path = config_path
        self.use_cuda = use_cuda
        self.speaker_id = speaker_id
        # If given, validated against the loaded voice's actual rate on
        # first use rather than trusted blindly — see _load_voice.
        self.sample_rate = sample_rate

        self._voice = None  # loaded lazily — see _ensure_voice
        self._stop_event = threading.Event()

    @classmethod
    def from_voice_id(cls, voice_id: str, voices_dir: str, **kwargs) -> "PiperTTS":
        """Resolve a persona's `voice_id` (e.g. "piper:en_US-lessac-medium")
        against a directory of voice models. Raises ValueError if `voice_id`
        isn't namespaced for this backend."""
        backend, sep, voice_name = voice_id.partition(":")
        if not sep or backend != "piper":
            raise ValueError(
                f"voice_id {voice_id!r} is not a piper: voice "
                '(expected "piper:<voice-name>")'
            )
        model_path = os.path.join(voices_dir, f"{voice_name}.onnx")
        return cls(voice_model_path=model_path, **kwargs)

    async def _ensure_voice(self):
        if self._voice is None:
            self._voice = await asyncio.to_thread(self._load_voice)
        return self._voice

    def _load_voice(self):
        from piper import PiperVoice

        voice = PiperVoice.load(
            self.voice_model_path,
            config_path=self.config_path,
            use_cuda=self.use_cuda,
        )
        actual_rate = voice.config.sample_rate
        if self.sample_rate is not None and self.sample_rate != actual_rate:
            raise ValueError(
                f"configured sample_rate={self.sample_rate} doesn't match "
                f"the loaded voice's native rate={actual_rate} "
                f"({self.voice_model_path}) — drop the override and let it "
                "be inferred, or fix the mismatch."
            )
        self.sample_rate = actual_rate
        return voice

    async def synthesize_stream(self, text_chunk: str) -> AsyncIterator[bytes]:
        self._stop_event.clear()
        voice = await self._ensure_voice()

        syn_config = None
        if self.speaker_id is not None:
            from piper import SynthesisConfig

            syn_config = SynthesisConfig(speaker_id=self.speaker_id)

        def _generate():
            for audio_chunk in voice.synthesize(text_chunk, syn_config=syn_config):
                yield audio_chunk.audio_int16_bytes

        async for pcm_bytes in iter_blocking_generator(_generate, self._stop_event):
            yield pcm_bytes

    def stop(self) -> None:
        self._stop_event.set()
