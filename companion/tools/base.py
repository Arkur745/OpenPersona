"""Tool types shared by every tool and by the agent runner.

A `Tool` is a plain async function plus a name and description. Parameters
carry their descriptions via `typing.Annotated[type, "description"]`, which
LangChain reads when it builds the schema the model sees. Keeping tools as
plain functions means this package has no LangChain/LangGraph dependency:
tools are testable on their own, and the agent backend
(`companion/backends/agent/`) is the only place that wraps them.

Every tool is bound to one `ToolContext` (who is talking to which persona)
when it is built, the same way memory is scoped per (user, persona), so a
model can never read or write another user's data through a tool argument.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from companion.core.interfaces import MemoryStore


def _local_now() -> datetime:
    return datetime.now().astimezone()


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    persona_id: str
    memory: MemoryStore
    data_dir: Path
    # Injectable so tests can pin "now"; always timezone-aware local time.
    now: Callable[[], datetime] = field(default=_local_now)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    # Async, keyword arguments only; returns the text the model sees.
    func: Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class ToolSpec:
    """One registry entry: a persona's `tools:` name maps to a group of
    `Tool`s built per context (e.g. `notes` -> add_note, search_notes)."""

    name: str
    summary: str
    build: Callable[[ToolContext], list[Tool]]
    requires_network: bool = False


class ToolInputError(ValueError):
    """Bad arguments from the model. The runner returns the message to the
    model as the tool result so it can correct itself, rather than failing
    the turn."""
