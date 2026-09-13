"""MemoryStore contract. See memory/store.py for the embedded-vector-DB
implementation (Chroma/LanceDB)."""

from abc import ABC, abstractmethod

from companion.core.interfaces.types import Turn


class MemoryStoreError(RuntimeError):
    """A `MemoryStore` backend's `write()`/`query()` raises this (or a
    subclass of it) to signal a genuine persistence-layer failure —
    disk corruption, an unreachable server — as distinct from a bug.

    Not every backend needs one: a backend with nothing that can fail
    this way (e.g. a pure in-memory store) has no reason to raise it.
    But any *caller* that wants to react to a persistence failure
    (`Orchestrator`, in particular) must catch this shared type, never a
    specific backend's own exception class — `Orchestrator` holds only
    interface references, never a concrete backend type, and that has to
    extend to the exceptions it catches too, or the same coupling this
    project has consistently avoided elsewhere sneaks back in through
    error handling instead of construction. See `companion.memory
    .memark_store.MemArkPersistenceError`, which subclasses this.
    """


class MemoryStore(ABC):
    """Per-user, per-persona long-term memory.

    Memory is scoped by (user_id, persona_id) — the same user's history with
    one persona must never leak into a conversation with another persona.
    """

    @abstractmethod
    async def query(
        self, user_id: str, persona_id: str, query: str, k: int = 5
    ) -> list[Turn]:
        """Return up to `k` turns most relevant to `query` for this
        user+persona, ordered most-relevant first."""
        raise NotImplementedError

    @abstractmethod
    async def write(self, user_id: str, persona_id: str, turn: Turn) -> None:
        """Persist one turn to long-term memory for this user+persona."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, user_id: str, persona_id: str | None = None) -> None:
        """Erase memory for a user.

        If `persona_id` is given, wipe only that persona's memory; if
        omitted, wipe everything stored for the user across all personas.
        This backs the user-facing memory-deletion requirement — it must be
        a real, complete erasure, not a soft/tombstone delete.
        """
        raise NotImplementedError
