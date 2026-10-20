"""Rolling summarization of older conversation history.

Runs as a background job, never inline in the request path — a chat turn
must not block on summarization latency. Keeps per-(user, persona) memory
useful over long-running relationships with a persona. See
docs/ROLLING_SUMMARIZER.md for the full design.

How it works: once a scope holds `trigger_turn_count` raw (`user`/
`assistant`) turns, the oldest half of them are summarized by the LLM and
replaced, through `MemoryStore.replace_turns()`, with one `role="system"`
summary turn timestamped at the newest turn it replaced. Summary turns are
excluded from the count and never re-summarized (section 3 of the note).

The summarization prompt lives here, not in `PromptBuilder`: that module
assembles persona prompts (persona + memory + recent window), and this
prompt has none of those.

Who triggers it: `Orchestrator` (CLI text and voice modes) schedules
`summarize_if_due` as a background task after each turn's memory writes;
the Streamlit UI calls it itself after a reply has rendered (section 5).
"""

from __future__ import annotations

import logging

from companion.core.interfaces.llm_backend import LLMBackend
from companion.core.interfaces.memory_store import MemoryStore
from companion.core.interfaces.types import Turn

logger = logging.getLogger(__name__)

SUMMARY_ROLE = "system"

_SUMMARY_PROMPT = (
    "You maintain the long-term memory of an AI companion. You will be given "
    "a dated transcript of an older part of a conversation between a user and "
    "the companion. Write a compact summary of it in the third person "
    "(\"The user ...\", \"The companion ...\"). Keep every concrete fact about "
    "the user: names, preferences, relationships, plans, dates, commitments, "
    "and anything the user asked to be remembered. Drop greetings and small "
    "talk. Use plain sentences, no headings or lists, at most 150 words. "
    "Output only the summary."
)


def _render_transcript(turns: list[Turn]) -> str:
    return "\n".join(
        f"[{turn.timestamp.strftime('%Y-%m-%d %H:%M')}] {turn.role}: {turn.text}"
        for turn in turns
    )


class RollingSummarizer:
    def __init__(
        self,
        memory: MemoryStore,
        llm: LLMBackend,
        trigger_turn_count: int = 50,
    ) -> None:
        if trigger_turn_count < 2:
            raise ValueError(
                f"trigger_turn_count must be at least 2 (got {trigger_turn_count}) "
                f"— the oldest half of that many turns is what gets summarized."
            )
        self.memory = memory
        self.llm = llm
        self.trigger_turn_count = trigger_turn_count

    @property
    def batch_size(self) -> int:
        return self.trigger_turn_count // 2

    async def summarize_if_due(self, user_id: str, persona_id: str) -> bool:
        """Check whether this user+persona's unsummarized history has grown
        to `trigger_turn_count`, and if so, replace the oldest
        `trigger_turn_count // 2` raw turns with a single summary turn.
        Returns True if a summarization ran and was stored.

        Returns False (changing nothing) when nothing is due, when the LLM
        produced an empty summary, or when the turns changed underneath it
        (e.g. a memory wipe while the LLM was running — see
        `MemoryStore.replace_turns`).
        """
        turns = await self.memory.list_turns(user_id, persona_id)
        raw = [turn for turn in turns if turn.role != SUMMARY_ROLE]
        if len(raw) < self.trigger_turn_count:
            return False

        batch = raw[: self.batch_size]
        summary_text = await self._summarize(batch)
        if not summary_text:
            logger.warning(
                "RollingSummarizer: LLM returned an empty summary; nothing "
                "replaced (user_id=%s, persona_id=%s)",
                user_id,
                persona_id,
            )
            return False

        first = batch[0].timestamp.strftime("%Y-%m-%d")
        last = batch[-1].timestamp.strftime("%Y-%m-%d")
        span = first if first == last else f"{first} to {last}"
        summary = Turn(
            role=SUMMARY_ROLE,
            text=f"Summary of earlier conversation ({span}): {summary_text}",
            timestamp=batch[-1].timestamp,
        )
        replaced = await self.memory.replace_turns(user_id, persona_id, batch, summary)
        if not replaced:
            logger.info(
                "RollingSummarizer: turns changed while summarizing; summary "
                "discarded (user_id=%s, persona_id=%s)",
                user_id,
                persona_id,
            )
        return replaced

    async def _summarize(self, batch: list[Turn]) -> str:
        context = [Turn(role="user", text=_render_transcript(batch))]
        parts = [token async for token in self.llm.generate_stream(_SUMMARY_PROMPT, context)]
        return "".join(parts).strip()
