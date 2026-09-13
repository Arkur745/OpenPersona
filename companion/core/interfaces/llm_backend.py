"""LLMBackend contract. See backends/llm/ for implementations."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from companion.core.interfaces.types import Turn


class LLMBackend(ABC):
    """Streaming text generation.

    `prompt` is the fully-assembled system+persona prompt for this turn
    (see core/prompt_builder/), `context` is the recent conversation window
    plus any retrieved memories, already merged and ordered by PromptBuilder.
    """

    @abstractmethod
    async def generate_stream(
        self, prompt: str, context: list[Turn]
    ) -> AsyncIterator[str]:
        """Yield response text fragments (tokens or small groups of tokens)
        as they're generated."""
        raise NotImplementedError
        yield ""  # pragma: no cover — makes this an async generator for type checkers
