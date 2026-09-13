"""TextToSpeech contract. See backends/tts/ for implementations."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class TextToSpeech(ABC):
    """Streaming text-to-speech.

    The orchestrator feeds text incrementally (typically sentence-by-sentence
    as an LLMBackend streams tokens) so playback can start before the full
    response has been generated.
    """

    @abstractmethod
    async def synthesize_stream(self, text_chunk: str) -> AsyncIterator[bytes]:
        """Feed one chunk of text (e.g. a completed sentence).

        Yields raw audio chunks as they're synthesized, in the sample
        format/rate the implementation documents.
        """
        raise NotImplementedError
        yield b""  # pragma: no cover — makes this an async generator for type checkers

    @abstractmethod
    def stop(self) -> None:
        """Abort any in-flight synthesis immediately.

        Called on barge-in: the user started speaking while a response was
        still being synthesized/played.
        """
        raise NotImplementedError
