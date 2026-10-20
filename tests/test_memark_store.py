"""Tests for companion/memory/memark_store.py — local SQLite persistence
for a MemArk-style graph. See docs/MEMARK_PERSISTENCE.md for the design
this implements.

No optional-dependency gating needed anywhere here: sqlite3 is Python's
standard library, and MemArkNode/MemArkEdge are local dataclasses (this
project doesn't depend on the real `memark` package — see
memark_store.py's module docstring for why).
"""

import asyncio
import time
from datetime import UTC, datetime

import pytest

from companion.memory.memark_store import (
    MemArkEdge,
    MemArkGraphStore,
    MemArkNode,
    MemArkPersistenceError,
    _db_filename,
    _user_db_prefix,
    populate_vector_search,
)


def _node(
    node_id: str = "n1",
    node_type: str = "episodic",
    summary: str = "a memory",
    embedding: list[float] | None = None,
    salience_score: float = 0.5,
    metadata: dict | None = None,
    raw_conversation_refs: list[str] | None = None,
) -> MemArkNode:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return MemArkNode(
        id=node_id,
        node_type=node_type,
        summary=summary,
        embedding=embedding,
        timestamp_created=now,
        timestamp_updated=now,
        importance_score=0.7,
        salience_score=salience_score,
        confidence_score=0.9,
        retrieval_count=3,
        last_accessed=now,
        parent_id=None,
        child_ids=[],
        raw_conversation_refs=raw_conversation_refs or ["turn-1", "turn-2"],
        metadata=metadata if metadata is not None else {"role": "user"},
    )


def _edge(
    edge_id: str = "e1",
    source_node: str = "n1",
    target_node: str = "n2",
    relation_type: str = "hierarchical",
) -> MemArkEdge:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    return MemArkEdge(
        id=edge_id,
        source_node=source_node,
        target_node=target_node,
        relation_type=relation_type,
        weight=0.8,
        confidence=0.95,
        created_at=now,
        last_accessed=now,
        retrieval_count=2,
        decay_rate=0.03,
    )


# --- naming: pure logic, mirrors companion/memory/store.py ----------------


def test_db_filename_uses_same_hash_scheme_as_chroma_store():
    from companion.memory.store import _collection_name

    # Same sha256(user_id)[:16] truncation, different prefix (a filename,
    # not a Chroma collection name) — per docs/MEMARK_PERSISTENCE.md
    # section 5, the scheme is mirrored exactly, not reinvented.
    chroma_name = _collection_name("someone@example.com", "mentor")
    memark_name = _db_filename("someone@example.com", "mentor")
    chroma_hash = chroma_name.removeprefix("mem-").removesuffix("-mentor")
    assert chroma_hash in memark_name
    assert memark_name == f"memark-{chroma_hash}-mentor.db"


# --- schema creation + startup cases ---------------------------------------


def test_load_on_missing_file_creates_schema_and_returns_empty(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))

    loaded = store.load("u1", "p1")

    assert loaded.nodes == {}
    assert loaded.edges == {}
    assert store._db_path("u1", "p1").exists()


def test_load_on_empty_existing_db_returns_empty_not_an_error(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    store.load("u1", "p1")  # creates the file with schema, no rows

    reloaded = MemArkGraphStore(persist_dir=str(tmp_path))
    loaded = reloaded.load("u1", "p1")

    assert loaded.nodes == {}
    assert loaded.edges == {}


def test_load_on_corrupted_file_raises_and_names_the_path(tmp_path):
    path = tmp_path / _db_filename("u1", "p1")
    path.write_bytes(b"this is not a sqlite database, just garbage bytes")

    store = MemArkGraphStore(persist_dir=str(tmp_path))

    with pytest.raises(MemArkPersistenceError) as exc_info:
        store.load("u1", "p1")

    assert str(path) in str(exc_info.value)


def test_load_on_row_that_fails_to_deserialize_raises_not_silently_empty(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    conn = store._ensure_connection("u1", "p1")
    # Insert a row with invalid JSON in the metadata column directly,
    # bypassing save_node -- simulates real corruption of a single field
    # within an otherwise-openable database.
    conn.execute(
        "INSERT INTO nodes (id, node_type, summary, embedding, "
        "timestamp_created, timestamp_updated, importance_score, "
        "salience_score, confidence_score, retrieval_count, "
        "last_accessed, raw_conversation_refs, metadata) VALUES "
        "('bad', 'episodic', 'x', NULL, '2026-01-01T00:00:00', "
        "'2026-01-01T00:00:00', 0.5, 0.5, 0.5, 0, "
        "'2026-01-01T00:00:00', '[]', 'NOT VALID JSON')"
    )
    conn.commit()

    fresh_store = MemArkGraphStore(persist_dir=str(tmp_path))
    with pytest.raises(MemArkPersistenceError):
        fresh_store.load("u1", "p1")


# --- node round-trip, including the JSON-encoded exception fields ---------


@pytest.mark.asyncio
async def test_node_round_trips_including_embedding_and_metadata(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    node = _node(
        node_id="n1",
        embedding=[0.1, 0.2, 0.3],
        metadata={"role": "user", "is_dormant": False, "nested": {"a": 1}},
        raw_conversation_refs=["turn-abc", "turn-def"],
    )

    await store.save_node("u1", "p1", node)
    loaded = store.load("u1", "p1")

    round_tripped = loaded.nodes["n1"]
    assert round_tripped.embedding == [0.1, 0.2, 0.3]
    assert round_tripped.metadata == {
        "role": "user",
        "is_dormant": False,
        "nested": {"a": 1},
    }
    assert round_tripped.raw_conversation_refs == ["turn-abc", "turn-def"]
    assert round_tripped.summary == node.summary
    assert round_tripped.salience_score == node.salience_score
    assert round_tripped.importance_score == node.importance_score
    assert round_tripped.confidence_score == node.confidence_score
    assert round_tripped.retrieval_count == node.retrieval_count
    assert round_tripped.timestamp_created == node.timestamp_created
    assert round_tripped.timestamp_updated == node.timestamp_updated
    assert round_tripped.last_accessed == node.last_accessed


@pytest.mark.asyncio
async def test_node_with_no_embedding_round_trips_as_none(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    node = _node(node_id="n1", embedding=None)

    await store.save_node("u1", "p1", node)
    loaded = store.load("u1", "p1")

    assert loaded.nodes["n1"].embedding is None


@pytest.mark.asyncio
async def test_saving_a_node_twice_replaces_it_not_duplicates(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "p1", _node(node_id="n1", salience_score=0.5))
    await store.save_node("u1", "p1", _node(node_id="n1", salience_score=0.1))

    loaded = store.load("u1", "p1")

    assert len(loaded.nodes) == 1
    assert loaded.nodes["n1"].salience_score == 0.1


# --- edge round-trip ---------------------------------------------------


@pytest.mark.asyncio
async def test_edge_round_trips_all_fields(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    edge = _edge(edge_id="e1", source_node="a", target_node="b")

    await store.save_edge("u1", "p1", edge)
    loaded = store.load("u1", "p1")

    round_tripped = loaded.edges["e1"]
    assert round_tripped.source_node == "a"
    assert round_tripped.target_node == "b"
    assert round_tripped.relation_type == edge.relation_type
    assert round_tripped.weight == edge.weight
    assert round_tripped.confidence == edge.confidence
    assert round_tripped.retrieval_count == edge.retrieval_count
    assert round_tripped.decay_rate == edge.decay_rate
    assert round_tripped.created_at == edge.created_at
    assert round_tripped.last_accessed == edge.last_accessed


# --- hierarchy reconstruction: matches GraphLoader's exact scope -----------


@pytest.mark.asyncio
async def test_hierarchical_edge_reconstructs_parent_and_child_ids(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "p1", _node(node_id="parent"))
    await store.save_node("u1", "p1", _node(node_id="child"))
    await store.save_edge(
        "u1",
        "p1",
        _edge(edge_id="e1", source_node="parent", target_node="child", relation_type="hierarchical"),
    )

    loaded = store.load("u1", "p1")

    assert loaded.nodes["child"].parent_id == "parent"
    assert loaded.nodes["parent"].child_ids == ["child"]


@pytest.mark.asyncio
async def test_hierarchical_goal_edge_does_not_populate_parent_child(tmp_path):
    """Regression test for matching GraphLoader's exact (narrower) scope:
    only relation_type == "hierarchical" triggers reconstruction, not
    "hierarchical_goal" (a distinct RelationType in MemEngine) -- proving
    this wasn't approximated as "anything hierarchy-like"."""
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "p1", _node(node_id="parent"))
    await store.save_node("u1", "p1", _node(node_id="child"))
    await store.save_edge(
        "u1",
        "p1",
        _edge(edge_id="e1", source_node="parent", target_node="child", relation_type="hierarchical_goal"),
    )

    loaded = store.load("u1", "p1")

    assert loaded.nodes["child"].parent_id is None
    assert loaded.nodes["parent"].child_ids == []


@pytest.mark.asyncio
async def test_hierarchical_edge_to_missing_node_is_skipped_not_an_error(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "p1", _node(node_id="parent"))
    # "child" is never saved -- a dangling edge, matching GraphLoader's
    # own "both ends must exist" guard.
    await store.save_edge(
        "u1",
        "p1",
        _edge(edge_id="e1", source_node="parent", target_node="child", relation_type="hierarchical"),
    )

    loaded = store.load("u1", "p1")  # must not raise

    assert loaded.nodes["parent"].child_ids == []


# --- scoping: separate files per (user_id, persona_id) ---------------------


@pytest.mark.asyncio
async def test_different_scopes_are_stored_in_separate_files(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "mentor", _node(node_id="n1", summary="mentor memory"))
    await store.save_node("u1", "friend", _node(node_id="n1", summary="friend memory"))

    mentor_loaded = store.load("u1", "mentor")
    friend_loaded = store.load("u1", "friend")

    assert mentor_loaded.nodes["n1"].summary == "mentor memory"
    assert friend_loaded.nodes["n1"].summary == "friend memory"
    assert store._db_path("u1", "mentor") != store._db_path("u1", "friend")


# --- delete(): what MemArkMemoryStore.delete() actually delegates to -----


def test_user_db_prefix_matches_that_users_own_filenames():
    # Same relationship companion/memory/store.py's
    # _user_collection_prefix has to _collection_name.
    name = _db_filename("u1", "mentor")
    assert name.startswith(_user_db_prefix("u1"))
    assert not name.startswith(_user_db_prefix("someone-else"))


@pytest.mark.asyncio
async def test_delete_one_scope_removes_only_that_files(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "mentor", _node(node_id="n1"))
    await store.save_node("u1", "friend", _node(node_id="n1"))

    store.delete("u1", "mentor")

    assert not store._db_path("u1", "mentor").exists()
    assert store._db_path("u1", "friend").exists()
    # A fresh load() on the deleted scope starts clean (a new file), not
    # an error — same "first run creates a fresh file successfully"
    # behavior as any other never-touched scope.
    assert store.load("u1", "mentor").nodes == {}


@pytest.mark.asyncio
async def test_delete_without_persona_removes_every_scope_for_that_user_only(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "mentor", _node(node_id="n1"))
    await store.save_node("u1", "friend", _node(node_id="n1"))
    await store.save_node("u2", "mentor", _node(node_id="n1"))

    store.delete("u1")

    assert not store._db_path("u1", "mentor").exists()
    assert not store._db_path("u1", "friend").exists()
    assert store._db_path("u2", "mentor").exists()
    assert "n1" in store.load("u2", "mentor").nodes


def test_delete_on_a_user_with_nothing_persisted_is_a_harmless_noop(tmp_path):
    store = MemArkGraphStore(persist_dir=str(tmp_path))

    store.delete("never-written-to")  # should not raise
    store.delete("never-written-to", "mentor")  # should not raise


@pytest.mark.asyncio
async def test_delete_closes_the_cached_connection_before_removing_the_file(tmp_path):
    """Windows-specific correctness requirement, not just tidiness: an
    open sqlite3 connection holds a file handle that blocks deletion on
    that platform. save_node above already forced a connection into
    store._connections for this scope — delete() must close it first,
    or the unlink itself would fail/silently leave the file behind on
    Windows even though it works by accident on POSIX."""
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "mentor", _node(node_id="n1"))
    assert ("u1", "mentor") in store._connections

    store.delete("u1", "mentor")

    assert ("u1", "mentor") not in store._connections
    assert not store._db_path("u1", "mentor").exists()


@pytest.mark.asyncio
async def test_delete_removes_wal_and_shm_sidecar_files_too(tmp_path):
    """PRAGMA journal_mode=WAL (set in _ensure_connection) leaves real
    -wal/-shm sidecar files next to the main .db file as soon as
    anything is written and not yet checkpointed — confirmed directly
    (a single save_node() call already produces both, no need to fake
    either), not assumed. Deleting only the main file would leave these
    behind."""
    store = MemArkGraphStore(persist_dir=str(tmp_path))
    await store.save_node("u1", "mentor", _node(node_id="n1"))
    db_path = store._db_path("u1", "mentor")
    wal_path = db_path.with_name(db_path.name + "-wal")
    shm_path = db_path.with_name(db_path.name + "-shm")
    assert wal_path.exists()  # sanity: WAL mode really did leave these,
    assert shm_path.exists()  # this isn't testing a fabricated scenario

    store.delete("u1", "mentor")

    assert not db_path.exists()
    assert not wal_path.exists()
    assert not shm_path.exists()


# --- populate_vector_search: FAISS rebuild via add_embedding replay -------


def test_populate_vector_search_replays_embeddings_for_indexable_types():
    calls = []

    class _FakeVectorSearch:
        def add_embedding(self, node_id, embedding, metadata=None):
            calls.append((node_id, embedding, metadata))

    nodes = {
        "n1": _node(node_id="n1", node_type="episodic", embedding=[0.1, 0.2]),
        "n2": _node(node_id="n2", node_type="goal", embedding=[0.3, 0.4]),  # not indexable
        "n3": _node(node_id="n3", node_type="concept", embedding=None),  # no embedding
    }

    populate_vector_search(nodes, _FakeVectorSearch())

    assert len(calls) == 1
    node_id, embedding, metadata = calls[0]
    assert node_id == "n1"
    assert embedding == [0.1, 0.2]
    assert metadata["node_type"] == "episodic"


# --- concurrency safety: the one non-negotiable requirement ---------------


@pytest.mark.asyncio
async def test_save_node_does_not_block_the_event_loop(tmp_path, monkeypatch):
    """Empirical, not just structural: injects a real blocking sleep into
    the synchronous write path and races it against an independent
    asyncio.sleep of the same duration via asyncio.gather. If save_node
    genuinely runs its blocking work off the event loop's thread (via
    asyncio.to_thread), the two 0.2s operations overlap and total elapsed
    time is close to 0.2s. If it ever stopped doing that, the blocking
    sleep would freeze the event loop's own thread, so the concurrent
    asyncio.sleep(0.2) couldn't overlap with it at all -- total elapsed
    would be close to their sum, ~0.4s.

    (An earlier version of this test counted ticks from a concurrent
    ticker task instead of measuring overlap -- that design was
    vacuously green even with to_thread removed, because a ticker that
    starts late from being starved still eventually accumulates the same
    tick count once it finally runs; only comparing total wall-clock time
    actually detects the delay. Confirmed by deliberately bypassing
    to_thread and re-running both versions before trusting either.)

    This is exactly the class of bug that caused the mic-ended deadlock
    (orchestrator.py's handle_voice_turn) two passes before this one --
    a real-time-timing interaction that looks correct until it's actually
    exercised concurrently."""
    store = MemArkGraphStore(persist_dir=str(tmp_path))

    real_save_node_sync = MemArkGraphStore.save_node_sync

    def slow_save_node_sync(self, user_id, persona_id, node):
        time.sleep(0.2)  # simulated slow disk I/O -- a real blocking call
        return real_save_node_sync(self, user_id, persona_id, node)

    monkeypatch.setattr(MemArkGraphStore, "save_node_sync", slow_save_node_sync)

    async def other_work() -> None:
        await asyncio.sleep(0.2)

    start = time.monotonic()
    await asyncio.gather(store.save_node("u1", "p1", _node()), other_work())
    elapsed = time.monotonic() - start

    assert elapsed < 0.35  # ~0.2s if concurrent; ~0.4s if the write blocked the loop


@pytest.mark.asyncio
async def test_save_edge_does_not_block_the_event_loop(tmp_path, monkeypatch):
    """Same proof as test_save_node_does_not_block_the_event_loop, for
    save_edge -- both write paths need this independently confirmed, not
    assumed from one passing."""
    store = MemArkGraphStore(persist_dir=str(tmp_path))

    real_save_edge_sync = MemArkGraphStore._save_edge_sync

    def slow_save_edge_sync(self, user_id, persona_id, edge):
        time.sleep(0.2)
        return real_save_edge_sync(self, user_id, persona_id, edge)

    monkeypatch.setattr(MemArkGraphStore, "_save_edge_sync", slow_save_edge_sync)

    async def other_work() -> None:
        await asyncio.sleep(0.2)

    start = time.monotonic()
    await asyncio.gather(store.save_edge("u1", "p1", _edge()), other_work())
    elapsed = time.monotonic() - start

    assert elapsed < 0.35
