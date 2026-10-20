"""MemoryStore implementation backed by an embedded vector DB — no external
server required for the local-first path.

Chroma is the only backend actually implemented — see
docs/MEMORY_STORE.md for why (picked over LanceDB, plus the ONNX MiniLM
default embedding it ships with — both verified there against the real
library, not assumed). `backend="lancedb"` is named but not implemented,
same pattern as STT's `whisper_cpp`.

Install: pip install "companion[memory-chroma]".

Scoping: one Chroma collection per `(user_id, persona_id)` pair, named
`mem-{sha256(user_id)[:16]}-{persona_id}` — collection names are
charset/length restricted (`[a-zA-Z0-9._-]`, 3-512 chars, alnum start/end
— verified directly), so `user_id` (not guaranteed to fit that, e.g. an
email contains `@`) gets hashed; `persona_id` is already constrained by
the persona loader. One collection per scope, rather than one shared
collection filtered by metadata, gives a *structural* leak guarantee —
matters given schema.md's explicit "must never leak into a conversation
with a different persona" requirement — see docs/MEMORY_STORE.md
section 7.

All chromadb calls are synchronous/blocking (local file I/O + CPU
embedding inference) — bridged onto a thread via `asyncio.to_thread` per
call, since each is one-shot, not a stream; no need for the queue-based
bridge `_sync_stream.py` provides for TTS's genuinely streaming case.

Turn round-tripping: `write()` stores `turn.timestamp.isoformat()` as
string metadata and `query()` parses it back with
`datetime.fromisoformat()` — verified exact (no precision loss) for
timezone-aware datetimes. This isn't incidental: the orchestrator's
dedup-against-`recent_turns` filter (docs/MEMORY_STORE.md section 4)
relies on a `Turn` read back from storage comparing equal (via `Turn`'s
generated `__eq__`, all three fields) to the same turn still sitting in
`recent_turns` — that only holds if `timestamp` round-trips exactly.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from companion.core.interfaces.memory_store import MemoryStore
from companion.core.interfaces.types import Turn

if TYPE_CHECKING:
    import chromadb

_COLLECTION_PREFIX = "mem-"


def _collection_name(user_id: str, persona_id: str) -> str:
    user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    return f"{_COLLECTION_PREFIX}{user_hash}-{persona_id}"


def _turn_from_chroma(document: str, metadata: dict) -> Turn:
    return Turn(
        role=metadata["role"],
        text=document,
        timestamp=datetime.fromisoformat(metadata["timestamp"]),
    )


def _turn_metadata(turn: Turn) -> dict[str, str]:
    return {"role": turn.role, "timestamp": turn.timestamp.isoformat()}


def _user_collection_prefix(user_id: str) -> str:
    user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    return f"{_COLLECTION_PREFIX}{user_hash}-"


class VectorMemoryStore(MemoryStore):
    def __init__(
        self,
        backend: str = "chroma",
        persist_dir: str = "./data/memory",
        embedding_function: Any = None,
    ) -> None:
        if backend != "chroma":
            raise NotImplementedError(
                f"memory backend {backend!r} is not implemented yet — only "
                f"'chroma' is (see docs/MEMORY_STORE.md for why LanceDB is "
                f"named-but-deferred, same pattern as STT's whisper_cpp)."
            )
        self.backend = backend
        self.persist_dir = persist_dir
        # None = let Chroma use its own default embedding function
        # (ONNX MiniLM-L6, local, no torch — see docs/MEMORY_STORE.md
        # section 2). Overridable here, not hardcoded, so swapping models
        # later is a config-level change.
        self._embedding_function = embedding_function
        # Client/collection handles are created lazily on first use, not
        # here — keep construction cheap and side-effect-free.
        self._client: chromadb.ClientAPI | None = None

    def _ensure_client(self) -> chromadb.ClientAPI:
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_collection(self, user_id: str, persona_id: str):
        client = self._ensure_client()
        kwargs: dict[str, Any] = {}
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        return client.get_or_create_collection(
            _collection_name(user_id, persona_id), **kwargs
        )

    async def query(
        self, user_id: str, persona_id: str, query: str, k: int = 5
    ) -> list[Turn]:
        def _query() -> list[Turn]:
            collection = self._get_collection(user_id, persona_id)
            # Verified directly: Chroma handles an empty collection and
            # n_results > count() itself (empty/short results, not an
            # error) — no need to special-case either here.
            results = collection.query(query_texts=[query], n_results=k)
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            # Chroma's own query() already returns results most-similar
            # first (ascending distance) — verified directly against a
            # real collection (see docs/MEMORY_STORE.md section 3) — so
            # this just maps straight through in the order given, no
            # re-sorting.
            return [
                _turn_from_chroma(document, metadata)
                for document, metadata in zip(documents, metadatas)
            ]

        return await asyncio.to_thread(_query)

    async def write(self, user_id: str, persona_id: str, turn: Turn) -> None:
        def _write() -> None:
            collection = self._get_collection(user_id, persona_id)
            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[turn.text],
                metadatas=[_turn_metadata(turn)],
            )

        await asyncio.to_thread(_write)

    async def list_turns(self, user_id: str, persona_id: str) -> list[Turn]:
        def _list() -> list[Turn]:
            collection = self._get_collection(user_id, persona_id)
            results = collection.get(include=["documents", "metadatas"])
            turns = [
                _turn_from_chroma(document, metadata)
                for document, metadata in zip(results["documents"], results["metadatas"])
            ]
            # get() returns insertion order, which isn't a timestamp
            # guarantee — sort explicitly. sorted() is stable, so equal
            # timestamps keep insertion order.
            return sorted(turns, key=lambda turn: turn.timestamp)

        return await asyncio.to_thread(_list)

    async def replace_turns(
        self, user_id: str, persona_id: str, old: Sequence[Turn], new: Turn
    ) -> bool:
        # One thread function for the whole check-add-delete, so cancelling
        # the awaiting task (e.g. asyncio.run tearing down at session end)
        # can't stop it halfway — see docs/ROLLING_SUMMARIZER.md section 5.
        def _replace() -> bool:
            collection = self._get_collection(user_id, persona_id)
            results = collection.get(include=["documents", "metadatas"])
            ids_by_turn: dict[Turn, list[str]] = {}
            for turn_id, document, metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                ids_by_turn.setdefault(_turn_from_chroma(document, metadata), []).append(
                    turn_id
                )
            delete_ids: list[str] = []
            for turn in old:
                candidates = ids_by_turn.get(turn)
                if not candidates:
                    return False
                delete_ids.append(candidates.pop())
            # Chroma has no multi-call transaction: add before delete, so a
            # crash in between leaves redundant memory, never lost memory.
            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[new.text],
                metadatas=[_turn_metadata(new)],
            )
            if delete_ids:
                collection.delete(ids=delete_ids)
            return True

        return await asyncio.to_thread(_replace)

    async def delete(self, user_id: str, persona_id: str | None = None) -> None:
        def _delete() -> None:
            client = self._ensure_client()
            existing = {c.name for c in client.list_collections()}
            if persona_id is not None:
                name = _collection_name(user_id, persona_id)
                if name in existing:
                    client.delete_collection(name)
                return
            prefix = _user_collection_prefix(user_id)
            for name in existing:
                if name.startswith(prefix):
                    client.delete_collection(name)

        await asyncio.to_thread(_delete)
