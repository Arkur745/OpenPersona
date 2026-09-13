"""SpeechToText contract. See backends/stt/ for implementations."""

from abc import ABC, abstractmethod


class SpeechToText(ABC):
    """Streaming speech-to-text.

    The orchestrator feeds raw PCM audio one chunk at a time (chunk size is
    decided by the audio capture layer, not this interface). Implementations
    own their internal buffering and decide when enough audio has
    accumulated to emit an updated partial transcript.
    """

    @abstractmethod
    async def transcribe_stream(self, audio_chunk: bytes) -> str | None:
        """Feed one chunk of raw audio.

        Returns the current best partial transcript for the in-progress
        utterance, or None if no new text is available yet.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear buffered audio and transcript state.

        Called at the start of a new utterance (e.g. after VAD detects
        end-of-speech and the transcript has been consumed).
        """
        raise NotImplementedError
