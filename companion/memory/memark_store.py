"""Local SQLite persistence for a MemArk-style graph, per
docs/MEMARK_PERSISTENCE.md — the P0 prerequisite that design note (and
docs/MEMARK_COMPANION_EXTENSIONS.md before it) identified: MemEngine's
own `GraphMemoryManager` has no local-disk persistence at all (confirmed
directly against its source: no save/dump/export method anywhere on it;
the only reconstruction path, `engine/persistence/graph_loader.py`'s
`GraphLoader.load_from_neo4j`, is hardwired to a real Neo4j server). This
module is what makes a MemArk graph survive a process restart without
depending on one.

**Scope, deliberately narrow**: this persists `MemArkNode`/`MemArkEdge`
records to and from SQLite. It is *not* a `MemoryStore`-conforming
backend — it doesn't implement `query()`/`write()` over `Turn` objects,
because that requires the graph-retrieval semantics (concept-seeding,
activation propagation, ranking into `list[Turn]`)
`docs/MEMARK_COMPANION_EXTENSIONS.md` proposed but explicitly left
unapproved and unimplemented. Building that adapter is separate, later
work; this module is the storage layer underneath it, usable on its own.

**This project does not depend on the `memark` package.** None of the
three prior MemArk design notes proposed adding it to this project's own
`pyproject.toml` — they analyzed its repo as an external reference, never
as a dependency to adopt. Importing its real `MemoryNode`/`MemoryEdge`
here would mean pulling in its full transitive dependency chain (torch
via sentence-transformers, neo4j, qdrant-client, streamlit, ...) for a
pass whose approved scope was persistence format/timing/scoping, not a
dependency-adoption decision. `MemArkNode`/`MemArkEdge` below are local
dataclasses that mirror `MemoryNode`'s/`MemoryEdge`'s field shape
exactly, as documented (and verified against the real source) in
`docs/MEMARK_PERSISTENCE.md` section 1. Swapping these for real imports
later, if `memark` ever becomes a real dependency, is a mechanical
change — nothing about the schema, write path, or reconstruction logic
depends on which class the data started out as.

Field-mapping is modeled on `GraphLoader._reconstruct_node`/
`_reconstruct_edge` (read directly before writing this, not
approximated) for the general approach: timestamp round-tripping via
ISO-8601 strings, metadata as JSON, hierarchical-edge replay to rebuild
`parent_id`/`child_ids`. **One deliberate deviation, not an oversight**:
`GraphLoader._reconstruct_node` never restores `salience_score` or
`raw_conversation_refs` at all — both silently reset to `MemoryNode`'s
dataclass defaults on every Neo4j reload, a real gap in `GraphLoader`
itself, not a design choice worth copying.
`docs/MEMARK_PERSISTENCE.md` section 1 explicitly requires salience/
activation state to survive a restart, so this module persists and
restores both correctly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from companion.core.interfaces.memory_store import MemoryStoreError

_DB_PREFIX = "memark-"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    embedding TEXT,
    timestamp_created TEXT NOT NULL,
    timestamp_updated TEXT NOT NULL,
    importance_score REAL NOT NULL,
    salience_score REAL NOT NULL,
    confidence_score REAL NOT NULL,
    retrieval_count INTEGER NOT NULL,
    last_accessed TEXT NOT NULL,
    raw_conversation_refs TEXT NOT NULL,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight REAL NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    retrieval_count INTEGER NOT NULL,
    decay_rate REAL NOT NULL
);
"""


class MemArkPersistenceError(MemoryStoreError):
    """The persisted SQLite file for a MemArk graph couldn't be read as
    one: an unopenable/unreadable database, a query that fails against
    it, or a row that doesn't deserialize into `MemArkNode`/`MemArkEdge`.
    Always names the exact file path and the underlying failure — never
    silently treated as "nothing was ever stored", since that would look
    to a user like their companion had forgotten everything with no
    indication why. Same convention as `companion.audio.errors
    .AudioDeviceError`: callers should never see a raw `sqlite3.Error`
    traceback instead of this.

    Subclasses `MemoryStoreError` (not bare `RuntimeError`) so a caller
    like `Orchestrator` can catch persistence failures generically,
    without importing anything MemArk-specific — see that shared type's
    own docstring for why this matters."""


@dataclass
class MemArkNode:
    """Mirrors MemEngine's own `MemoryNode` field-for-field — see module
    docstring for why this is a local mirror, not an import."""

    id: str
    node_type: str
    summary: str
    embedding: list[float] | None
    timestamp_created: datetime
    timestamp_updated: datetime
    importance_score: float
    salience_score: float
    confidence_score: float
    retrieval_count: int
    last_accessed: datetime
    parent_id: str | None
    child_ids: list[str]
    raw_conversation_refs: list[str]
    metadata: dict[str, Any]


@dataclass
class MemArkEdge:
    """Mirrors MemEngine's own `MemoryEdge` field-for-field."""

    id: str
    source_node: str
    target_node: str
    relation_type: str
    weight: float
    confidence: float
    created_at: datetime
    last_accessed: datetime
    retrieval_count: int
    decay_rate: float


@dataclass
class LoadedGraph:
    """What `MemArkGraphStore.load()` returns — `parent_id`/`child_ids`
    on every node are already reconstructed from hierarchical edges."""

    nodes: dict[str, MemArkNode]
    edges: dict[str, MemArkEdge]


class SupportsAddEmbedding(Protocol):
    """Structural type for whatever VectorSearch-shaped object
    `populate_vector_search()` replays embeddings into — deliberately not
    importing MemArk's own `VectorSearch` (see module docstring)."""

    def add_embedding(
        self,
        node_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


def _db_filename(user_id: str, persona_id: str) -> str:
    """Same hashing scheme as `companion/memory/store.py`'s
    `_collection_name` (`sha256(user_id)[:16]` + `persona_id`) — mirrored
    exactly per `docs/MEMARK_PERSISTENCE.md` section 5, not reinvented.
    Different prefix because this names a SQLite file, not a Chroma
    collection, not because the scheme differs."""
    user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    return f"{_DB_PREFIX}{user_hash}-{persona_id}.db"


def _user_db_prefix(user_id: str) -> str:
    """Same scheme as `companion/memory/store.py`'s
    `_user_collection_prefix` — the `sha256(user_id)[:16]` portion shared
    by every persona's file for this user, without the trailing
    `{persona_id}.db`. Only `user_id` is hashed (see `_db_filename`), so
    a filename matching this prefix has its real `persona_id` sitting
    right after it, recoverable by a plain string slice — used by
    `MemArkGraphStore.delete(user_id, persona_id=None)` to find every
    scope belonging to one user without needing a separate index."""
    user_hash = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    return f"{_DB_PREFIX}{user_hash}-"


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class MemArkGraphStore:
    """Persists `MemArkNode`/`MemArkEdge` records to one SQLite file per
    `(user_id, persona_id)` — see `docs/MEMARK_PERSISTENCE.md` for the
    full design (format choice, write timing, crash safety, scoping).

    Not a `MemoryStore`-conforming backend — see module docstring.
    """

    def __init__(self, persist_dir: str = "./data/memark") -> None:
        self.persist_dir = Path(persist_dir)
        # One connection per (user_id, persona_id), created lazily and
        # reused for the store's whole lifetime — check_same_thread=False
        # because save_node/save_edge run inside asyncio.to_thread (a
        # different worker thread per call), the same pattern
        # VectorMemoryStore already uses for its Chroma client (see
        # companion/memory/store.py's _ensure_client). Safe ONLY because
        # this store, like VectorMemoryStore, never has more than one
        # write in flight at a time for a given scope — each
        # `await asyncio.to_thread(...)` call completes before the next
        # one starts. If concurrent-session handling (more than one
        # write in flight at once for the same scope) is ever added,
        # this assumption needs revisiting alongside it, not silently.
        self._connections: dict[tuple[str, str], sqlite3.Connection] = {}
        self._connections_lock = threading.Lock()

    # --- connection management -----------------------------------------

    def _db_path(self, user_id: str, persona_id: str) -> Path:
        return self.persist_dir / _db_filename(user_id, persona_id)

    def _ensure_connection(self, user_id: str, persona_id: str) -> sqlite3.Connection:
        key = (user_id, persona_id)
        with self._connections_lock:
            conn = self._connections.get(key)
            if conn is not None:
                return conn
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            path = self._db_path(user_id, persona_id)
            try:
                conn = sqlite3.connect(str(path), check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            except sqlite3.Error as exc:
                raise MemArkPersistenceError(
                    f"MemArk persistence file {path} could not be opened as "
                    f"a SQLite database: {exc}. This looks like real "
                    f"corruption, not a first run (a first run creates a "
                    f"fresh file successfully) — investigate or restore "
                    f"from backup before continuing; starting fresh would "
                    f"silently discard whatever memory was stored there."
                ) from exc
            self._connections[key] = conn
            return conn

    # --- writes: async, to_thread-wrapped -------------------------------

    async def save_node(self, user_id: str, persona_id: str, node: MemArkNode) -> None:
        await asyncio.to_thread(self._save_node_sync, user_id, persona_id, node)

    def _save_node_sync(self, user_id: str, persona_id: str, node: MemArkNode) -> None:
        conn = self._ensure_connection(user_id, persona_id)
        conn.execute(
            """
            INSERT OR REPLACE INTO nodes (
                id, node_type, summary, embedding,
                timestamp_created, timestamp_updated,
                importance_score, salience_score, confidence_score,
                retrieval_count, last_accessed,
                raw_conversation_refs, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node.id,
                node.node_type,
                node.summary,
                json.dumps(node.embedding) if node.embedding is not None else None,
                _iso(node.timestamp_created),
                _iso(node.timestamp_updated),
                node.importance_score,
                node.salience_score,
                node.confidence_score,
                node.retrieval_count,
                _iso(node.last_accessed),
                json.dumps(node.raw_conversation_refs),
                json.dumps(node.metadata),
            ),
        )
        conn.commit()

    async def save_edge(self, user_id: str, persona_id: str, edge: MemArkEdge) -> None:
        await asyncio.to_thread(self._save_edge_sync, user_id, persona_id, edge)

    def _save_edge_sync(self, user_id: str, persona_id: str, edge: MemArkEdge) -> None:
        conn = self._ensure_connection(user_id, persona_id)
        conn.execute(
            """
            INSERT OR REPLACE INTO edges (
                id, source_node, target_node, relation_type,
                weight, confidence, created_at, last_accessed,
                retrieval_count, decay_rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge.id,
                edge.source_node,
                edge.target_node,
                edge.relation_type,
                edge.weight,
                edge.confidence,
                _iso(edge.created_at),
                _iso(edge.last_accessed),
                edge.retrieval_count,
                edge.decay_rate,
            ),
        )
        conn.commit()

    # --- load: sync, startup-time only -----------------------------------
    # Per docs/MEMARK_PERSISTENCE.md section 4: this is called from
    # app.py's build_memory-equivalent flow before asyncio.run(...) ever
    # starts the voice loop — no event loop exists yet at that point, so
    # wrapping this in asyncio.to_thread would add nothing but
    # complexity. Deliberately NOT async.

    def load(self, user_id: str, persona_id: str) -> LoadedGraph:
        conn = self._ensure_connection(user_id, persona_id)
        try:
            nodes = self._load_nodes(conn)
            edges = self._load_edges(conn)
        except (
            sqlite3.Error,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            path = self._db_path(user_id, persona_id)
            raise MemArkPersistenceError(
                f"MemArk persistence file {path} exists but a row failed "
                f"to load: {exc}. Not treating this as an empty graph — an "
                f"empty graph and a corrupted one must not look the same "
                f"to a user who might have real memory stored there."
            ) from exc

        self._reconstruct_hierarchy(nodes, edges)
        return LoadedGraph(nodes=nodes, edges=edges)

    def _load_nodes(self, conn: sqlite3.Connection) -> dict[str, MemArkNode]:
        rows = conn.execute(
            "SELECT id, node_type, summary, embedding, timestamp_created, "
            "timestamp_updated, importance_score, salience_score, "
            "confidence_score, retrieval_count, last_accessed, "
            "raw_conversation_refs, metadata FROM nodes"
        ).fetchall()
        nodes: dict[str, MemArkNode] = {}
        for row in rows:
            (
                node_id,
                node_type,
                summary,
                embedding_json,
                timestamp_created,
                timestamp_updated,
                importance_score,
                salience_score,
                confidence_score,
                retrieval_count,
                last_accessed,
                raw_refs_json,
                metadata_json,
            ) = row
            nodes[node_id] = MemArkNode(
                id=node_id,
                node_type=node_type,
                summary=summary,
                embedding=(
                    json.loads(embedding_json) if embedding_json is not None else None
                ),
                timestamp_created=_parse_iso(timestamp_created),
                timestamp_updated=_parse_iso(timestamp_updated),
                importance_score=float(importance_score),
                salience_score=float(salience_score),
                confidence_score=float(confidence_score),
                retrieval_count=int(retrieval_count),
                last_accessed=_parse_iso(last_accessed),
                parent_id=None,  # reconstructed from hierarchical edges below
                child_ids=[],  # reconstructed from hierarchical edges below
                raw_conversation_refs=json.loads(raw_refs_json),
                metadata=json.loads(metadata_json),
            )
        return nodes

    def _load_edges(self, conn: sqlite3.Connection) -> dict[str, MemArkEdge]:
        rows = conn.execute(
            "SELECT id, source_node, target_node, relation_type, weight, "
            "confidence, created_at, last_accessed, retrieval_count, "
            "decay_rate FROM edges"
        ).fetchall()
        edges: dict[str, MemArkEdge] = {}
        for row in rows:
            (
                edge_id,
                source_node,
                target_node,
                relation_type,
                weight,
                confidence,
                created_at,
                last_accessed,
                retrieval_count,
                decay_rate,
            ) = row
            edges[edge_id] = MemArkEdge(
                id=edge_id,
                source_node=source_node,
                target_node=target_node,
                relation_type=relation_type,
                weight=float(weight),
                confidence=float(confidence),
                created_at=_parse_iso(created_at),
                last_accessed=_parse_iso(last_accessed),
                retrieval_count=int(retrieval_count),
                decay_rate=float(decay_rate),
            )
        return edges

    def _reconstruct_hierarchy(
        self, nodes: dict[str, MemArkNode], edges: dict[str, MemArkEdge]
    ) -> None:
        """Rebuilds parent_id/child_ids purely from hierarchical-typed
        edges — matches `GraphLoader.load_from_neo4j`'s own
        reconstruction exactly (`relation_type == "hierarchical"`
        specifically, not `"hierarchical_goal"`; both ends must exist),
        read directly from `engine/persistence/graph_loader.py` before
        writing this, not approximated."""
        for edge in edges.values():
            if edge.relation_type != "hierarchical":
                continue
            parent = nodes.get(edge.source_node)
            child = nodes.get(edge.target_node)
            if parent is None or child is None:
                continue
            child.parent_id = edge.source_node
            if edge.target_node not in parent.child_ids:
                parent.child_ids.append(edge.target_node)

    # --- delete: sync, matches MemoryStore.delete()'s own contract -----
    # Not a MemoryStore method itself (see class docstring), but shaped
    # to match one exactly (persona_id=None means "every scope for this
    # user") since MemArkMemoryStore.delete() delegates straight to this.

    def delete(self, user_id: str, persona_id: str | None = None) -> None:
        """Removes the persisted SQLite file(s) for one scope
        (`persona_id` given) or every scope this `user_id` has
        (`persona_id=None`) — closing any cached connection first (a live
        connection holds an open file handle, which blocks deletion on
        Windows and is simply wrong to leave dangling on any platform)
        and removing WAL/SHM sidecar files `PRAGMA journal_mode=WAL`
        creates alongside the main `.db` file, not just that file itself.
        Silently no-ops on a scope/user with nothing persisted, same
        tolerance `VectorMemoryStore.delete()` has for Chroma."""
        if persona_id is not None:
            self._delete_scope_files(user_id, persona_id)
            return
        if not self.persist_dir.exists():
            return
        prefix = _user_db_prefix(user_id)
        for path in sorted(self.persist_dir.glob(f"{prefix}*.db")):
            # Only user_id is hashed into the filename (see
            # _db_filename/_user_db_prefix) — the real persona_id sits
            # right after this prefix, recoverable by a plain slice, no
            # separate index needed.
            found_persona_id = path.stem[len(prefix) :]
            self._delete_scope_files(user_id, found_persona_id)

    def _delete_scope_files(self, user_id: str, persona_id: str) -> None:
        key = (user_id, persona_id)
        with self._connections_lock:
            conn = self._connections.pop(key, None)
        if conn is not None:
            conn.close()
        path = self._db_path(user_id, persona_id)
        for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            candidate.unlink(missing_ok=True)


def populate_vector_search(
    nodes: dict[str, MemArkNode],
    vector_search: SupportsAddEmbedding,
    indexable_node_types: tuple[str, ...] = ("concept", "summary", "episodic"),
) -> None:
    """Rebuilds a VectorSearch/FAISS index from persisted node
    embeddings — no separate vector-index file, no re-embedding, per
    `docs/MEMARK_PERSISTENCE.md` section 1. Mirrors `GraphLoader
    .load_from_neo4j`'s own reconstruction loop (same
    `indexable_node_types` default, same `add_embedding()` call shape)."""
    for node in nodes.values():
        if node.node_type in indexable_node_types and node.embedding is not None:
            vector_search.add_embedding(
                node_id=node.id,
                embedding=node.embedding,
                metadata={
                    "node_type": node.node_type,
                    "salience": node.salience_score,
                    "is_dormant": node.metadata.get("is_dormant", False),
                },
            )
