"""`LLMBackend`: the streaming text-generation contract.

One method, `generate_stream(prompt, context)`, yielding text fragments.
Implementation: `companion/backends/llm/ollama_local.py` (`OllamaLLM`).
Personas with tools don't call this directly; they go through the agent
runner in `companion/backends/agent/`, which talks to Ollama itself.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from companion.core.interfaces.types import Turn


class LLMBackend(ABC):
    """Streaming text generation.

    `prompt` is the fully-assembled system prompt for this turn: persona
    plus any retrieved memories, rendered as text (see core/prompt_builder/).
    `context` is the recent conversation window, oldest first, ending with
    the user's new message. Both come straight from `PromptBuilder.build()`.
    """

    @abstractmethod
    async def generate_stream(
        self, prompt: str, context: list[Turn]
    ) -> AsyncIterator[str]:
        """Yield response text fragments (tokens or small groups of tokens)
        as they're generated."""
        raise NotImplementedError
        yield ""  # pragma: no cover — makes this an async generator for type checkers
