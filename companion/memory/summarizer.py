"""Rolling summarization of older conversation history.

Runs as a background job (e.g. triggered after N turns or on a schedule),
never inline in the request path — a chat turn must not block on
summarization latency. Keeps per-(user, persona) context size/token cost
bounded over long-running relationships with a persona.
"""

from companion.core.interfaces.llm_backend import LLMBackend
from companion.core.interfaces.memory_store import MemoryStore


class RollingSummarizer:
    def __init__(
        self,
        memory: MemoryStore,
        llm: LLMBackend,
        trigger_turn_count: int = 50,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.trigger_turn_count = trigger_turn_count

    async def summarize_if_due(self, user_id: str, persona_id: str) -> bool:
        """Check whether this user+persona's unsummarized history has grown
        past `trigger_turn_count`, and if so, replace the oldest turns with
        a single summary turn. Returns True if a summarization ran.
        """
        raise NotImplementedError
