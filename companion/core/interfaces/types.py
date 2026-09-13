"""Shared value types used across interface contracts.

Kept dependency-free (stdlib only) so that backends/ implementations never
need to import each other's packages just to satisfy a type hint.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class Turn:
    """One message in a conversation, as stored/retrieved by a MemoryStore
    or passed to an LLMBackend as context."""

    role: Role
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class Persona:
    """In-memory representation of a persona config file (personas/*.yaml).

    Loading/validating YAML into this shape is the persona loader's job
    (not implemented yet — see personas/schema.md); this type is the
    contract the loader must produce and everything else consumes.
    """

    id: str
    name: str
    tone: str
    boundaries: str
    voice_id: str
    example_dialogue: list[str] = field(default_factory=list)
