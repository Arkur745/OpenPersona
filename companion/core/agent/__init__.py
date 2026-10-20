"""The agent contract `Orchestrator` depends on for personas with tools.

`AgentRunner` is a `typing.Protocol`, not a sixth abstract interface next to
STT/TTS/LLM/VAD/MemoryStore: there is one implementation
(`companion/backends/agent/langgraph_ollama.py`), and docs/AGENTIC_TOOLS.md
argues against adding abstract surface on the strength of one backend.
`Orchestrator` only sees this protocol and the event types below, never
LangGraph.

A runner owns building each persona's tools (it knows the tool data
directory, whether network tools are allowed, and the memory store), so
`Orchestrator` stays unaware of individual tools.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from companion.core.interfaces.types import Persona
    from companion.core.prompt_builder import BuiltPrompt


@dataclass(frozen=True)
class TextDelta:
    """A piece of the final answer, in order."""

    text: str


@dataclass(frozen=True)
class ToolStarted:
    name: str


@dataclass(frozen=True)
class ToolFinished:
    name: str
    ok: bool
    # When `ok` is False: the error the tool returned, shown to the user by
    # the CLI and Streamlit. A model can claim success after a failed tool
    # call; surfacing the error keeps that visible.
    detail: str = ""


AgentEvent = TextDelta | ToolStarted | ToolFinished

# Spoken in voice mode while a tool runs, for personas without their own
# `filler_phrases`. Fixed text, not LLM-generated: generating it would add
# the very latency it exists to cover.
DEFAULT_FILLER_PHRASES = ("One moment.", "Let me check.", "Just a second.")


@runtime_checkable
class AgentRunner(Protocol):
    def handles(self, persona: Persona) -> bool:
        """True if this persona has at least one tool enabled under the
        current config. False means the plain LLM path is used."""
        ...

    def run_stream(
        self, persona: Persona, user_id: str, built: BuiltPrompt
    ) -> AsyncIterator[AgentEvent]:
        """Runs the tool-calling loop for one turn. `built` is exactly what
        `PromptBuilder.build()` produced, so persona, memory and recent
        turns reach the agent the same way they reach the plain LLM."""
        ...
