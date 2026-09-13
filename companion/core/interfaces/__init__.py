"""Abstract backend contracts.

Nothing outside `backends/` and `memory/` should import a concrete backend
directly — orchestrator, prompt_builder, and UI code should only ever
depend on these interfaces, with the concrete implementation chosen at
runtime by config (see config/config.example.yaml).
"""

from companion.core.interfaces.llm_backend import LLMBackend
from companion.core.interfaces.memory_store import MemoryStore, MemoryStoreError
from companion.core.interfaces.speech_to_text import SpeechToText
from companion.core.interfaces.text_to_speech import TextToSpeech
from companion.core.interfaces.types import Role, Turn
from companion.core.interfaces.vad import VoiceActivityDetector

__all__ = [
    "LLMBackend",
    "MemoryStore",
    "MemoryStoreError",
    "Role",
    "SpeechToText",
    "TextToSpeech",
    "Turn",
    "VoiceActivityDetector",
]
