"""VoiceActivityDetector contract. See backends/vad/ for implementations."""

from abc import ABC, abstractmethod


class VoiceActivityDetector(ABC):
    """Per-chunk speech/silence classification, used by the orchestrator for
    turn-taking (end-of-utterance detection) and barge-in (interrupting TTS
    playback when the user starts speaking again)."""

    @abstractmethod
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Classify one chunk of raw audio as speech or silence.

        Implementations are expected to be cheap enough to call on every
        audio chunk in real time (no batching).
        """
        raise NotImplementedError
