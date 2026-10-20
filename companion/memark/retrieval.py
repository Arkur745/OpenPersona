"""The MemArk retrieval adapter — makes `MemArkGraphStore`
(`companion/memory/memark_store.py`, persistence, already implemented
and tested) into a real `MemoryStore`, per
`docs/MEMARK_RETRIEVAL_ADAPTER.md` sections 1-8 (approved design,
implemented here as specified — this module is not a design pass).

**Not yet wired into `app.py`/`orchestrator.py`** — deliberately, per
that note's section 7: what a live turn should do when
`MemArkPersistenceError` propagates out of `write()` (end the turn? end
the session? catch-and-continue?) is an orchestrator-layer decision this
module correctly doesn't make.

Two things found while implementing the approved design that didn't
survive contact with the real API, resolved and flagged rather than
silently worked around (both confirmed with the user before writing this
file):

1. **Where the real MemEngine classes come from.** The design note
   assumed `GraphMemoryManager`/`ActivationEngine`/`FaissVectorStore`
   are simply importable — none of the five prior MemArk notes actually
   decided how. Tracing every needed file's own imports (module-level
   and function-level) showed the real cost of what THIS adapter needs
   is `networkx` + `faiss-cpu` (`numpy` already a `dev`-extra dependency)
   — not the whole `memark` package's torch/neo4j/qdrant-client/
   streamlit chain, none of which anything here calls. Resolved by
   vendoring (not pip-installing) the specific files into
   `companion/memark/_vendor/` — see that directory's `PROVENANCE.md`
   for the exact commit, file list, and reasoning.
2. **Hydration can't literally "replay `create_node`/`create_edge`
   calls."** Both always self-generate a fresh random id
   (`self._new_id()`) with no override parameter — confirmed directly
   against `engine/core/graph_manager.py`. Replaying persisted nodes
   through `create_node()` would give every node a new id on every
   restart, orphaning every persisted edge's `source_node`/`target_node`
   reference — and replaying `create_edge()` with the *original*
   persisted ids would raise `ValueError` immediately (`create_edge`
   requires both ends to already be in `.nodes`). `_hydrate_scope()`
   below bypasses `create_node`/`create_edge` for this one purpose only,
   constructing real `MemoryNode`/`MemoryEdge` objects with the original
   persisted `id` preserved and inserting them the same way those
   methods do internally (`.nodes[id] = node`, `.graph.add_node(id,
   **asdict(node))`). Every *live* write during a session still goes
   through the real, unmodified `create_node()`/`create_edge()` — this
   bypass exists only for reconstructing a previously-persisted graph.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from companion.memark._engine_path import ensure_engine_importable

ensure_engine_importable()

from engine.core.graph_manager import GraphMemoryManager
from engine.core.memory_edge import MemoryEdge
from engine.core.memory_node import MemoryNode
from engine.retrieval.activation_engine import ActivationEngine
from engine.retrieval.faiss_store import FaissVectorStore
from engine.storage.memory_repository import MemoryRepository

from companion.core.interfaces.memory_store import MemoryStore
from companion.core.interfaces.types import Turn
from companion.memory.memark_store import (
    MemArkGraphStore,
    MemArkNode,
    populate_vector_search,
)

_CONCEPT_NODE_TYPE = "concept"
_EPISODIC_NODE_TYPE = "episodic"


@dataclass
class _Scope:
    """Everything this adapter holds in memory for one (user_id,
    persona_id) — hydrated once, lazily, on first use."""

    manager: GraphMemoryManager
    repository: MemoryRepository
    activation_engine: ActivationEngine
    vector_search: FaissVectorStore


def _default_embedding_function() -> Any:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


class MemArkMemoryStore(MemoryStore):
    """`MemoryStore` implementation backed by MemArk's graph-retrieval
    engine (vendored, see module docstring) plus `MemArkGraphStore` for
    local SQLite persistence. See `docs/MEMARK_RETRIEVAL_ADAPTER.md` for
    the full design this implements.

    `embedding_function` defaults to Chroma's own standalone
    `DefaultEmbeddingFunction` (verified callable with no Chroma client
    or collection needed — see `docs/MEMARK_RETRIEVAL_ADAPTER.md`
    section 1) rather than MemArk's own `EmbeddingManager`, specifically
    to avoid the torch dependency that would pull in. Overridable for
    tests or a different embedding source, same pattern as
    `VectorMemoryStore`.
    """

    def __init__(
        self,
        persist_dir: str = "./data/memark",
        embedding_function: Any = None,
    ) -> None:
        self._graph_store = MemArkGraphStore(persist_dir=persist_dir)
        # None = lazily construct Chroma's default the first time an
        # embedding is actually needed, not at construction (keeps
        # constructing this store cheap and side-effect-free, matching
        # VectorMemoryStore's own _ensure_client laziness).
        self._embedding_function = embedding_function
        self._embedding_dim: int | None = None
        self._scopes: dict[tuple[str, str], _Scope] = {}
        # Serializes write()/replace_turns()/delete() across threads. A
        # background RollingSummarizer makes replace_turns() a second
        # writer that can run alongside a turn's write(), and both the
        # in-memory graph and MemArkGraphStore's shared SQLite connection
        # assume one writer at a time. A threading.Lock (not asyncio.Lock)
        # because each body runs in an asyncio.to_thread worker, and
        # Streamlit drives successive calls from different event loops.
        # See docs/ROLLING_SUMMARIZER.md section 6.2.
        self._lock = threading.Lock()

    # --- embedding --------------------------------------------------

    def _ensure_embedding_function(self) -> Any:
        if self._embedding_function is None:
            self._embedding_function = _default_embedding_function()
        return self._embedding_function

    def _embed(self, text: str) -> list[float]:
        # float(...) matters, not just list(...): chromadb's
        # DefaultEmbeddingFunction returns numpy.float32 elements, which
        # MemArkGraphStore's json.dumps(node.embedding) can't serialize
        # (confirmed by actually running this, not assumed — raises
        # TypeError: Object of type float32 is not JSON serializable
        # otherwise).
        ef = self._ensure_embedding_function()
        return [float(x) for x in ef([text])[0]]

    def _embedding_dimension(self) -> int:
        # Derived from the actual configured embedding function, not a
        # hardcoded 384 -- found by actually running this with a
        # non-default (fake, test-only) embedding function: FaissVectorStore
        # asserts its index dimension matches every vector added to it, so
        # a hardcoded constant only correct for chromadb's own default
        # would crash (AssertionError deep in faiss) the moment a
        # differently-sized embedding function is configured, e.g. in
        # tests. Computed once per store instance and cached.
        if self._embedding_dim is None:
            self._embedding_dim = len(self._embed("dimension probe"))
        return self._embedding_dim

    # --- scope hydration: sync, lazy, per (user_id, persona_id) -----
    # Per docs/MEMARK_PERSISTENCE.md section 4 / docs/
    # MEMARK_RETRIEVAL_ADAPTER.md section 3: MemArkGraphStore.load() is
    # sync and only ever called here, from inside a call already
    # running via asyncio.to_thread (write()/query()'s own wrapping) —
    # never on the event loop's own thread, so this being sync is
    # exactly as safe as the design note reasoned, just lazy-per-scope
    # rather than eager-for-all-scopes (MemoryStore's interface has no
    # way to enumerate scopes in advance, so eager isn't actually
    # possible — the first touch of a given (user_id, persona_id) is
    # the earliest this could ever happen).

    def _ensure_scope(self, user_id: str, persona_id: str) -> _Scope:
        key = (user_id, persona_id)
        scope = self._scopes.get(key)
        if scope is not None:
            return scope
        scope = self._hydrate_scope(user_id, persona_id)
        self._scopes[key] = scope
        return scope

    def _hydrate_scope(self, user_id: str, persona_id: str) -> _Scope:
        loaded = self._graph_store.load(user_id, persona_id)

        manager = GraphMemoryManager()  # no neo4j, no vector_search passed
        # in -- this adapter owns FAISS separately (see module docstring
        # on why VectorSearch's own wrapper, which pulls in qdrant, is
        # skipped entirely).

        # See module docstring point 2: bypasses create_node/create_edge
        # deliberately, preserving the exact persisted ids so edges keep
        # resolving after a restart.
        for node_id, node in loaded.nodes.items():
            memory_node = MemoryNode(
                id=node_id,
                node_type=node.node_type,
                summary=node.summary,
                embedding=node.embedding,
                timestamp_created=node.timestamp_created,
                timestamp_updated=node.timestamp_updated,
                importance_score=node.importance_score,
                salience_score=node.salience_score,
                confidence_score=node.confidence_score,
                retrieval_count=node.retrieval_count,
                last_accessed=node.last_accessed,
                parent_id=node.parent_id,
                child_ids=list(node.child_ids),
                raw_conversation_refs=list(node.raw_conversation_refs),
                metadata=dict(node.metadata),
            )
            manager.nodes[node_id] = memory_node
            manager.graph.add_node(node_id, **asdict(memory_node))

        for edge_id, edge in loaded.edges.items():
            memory_edge = MemoryEdge(
                id=edge_id,
                source_node=edge.source_node,
                target_node=edge.target_node,
                relation_type=edge.relation_type,
                weight=edge.weight,
                confidence=edge.confidence,
                created_at=edge.created_at,
                last_accessed=edge.last_accessed,
                retrieval_count=edge.retrieval_count,
                decay_rate=edge.decay_rate,
            )
            manager.edges[edge_id] = memory_edge
            manager.graph.add_edge(
                edge.source_node, edge.target_node, **asdict(memory_edge)
            )

        vector_search = FaissVectorStore(embedding_dim=self._embedding_dimension())
        populate_vector_search(loaded.nodes, vector_search)  # replay from
        # persisted embeddings -- no re-embedding, per
        # docs/MEMARK_PERSISTENCE.md section 1.

        repository = MemoryRepository(manager)
        activation_engine = ActivationEngine(repository)

        return _Scope(
            manager=manager,
            repository=repository,
            activation_engine=activation_engine,
            vector_search=vector_search,
        )

    # --- write --------------------------------------------------------
    # One asyncio.to_thread call: embed + in-memory mutate + persist, all
    # under self._lock (docs/ROLLING_SUMMARIZER.md section 6.2). An earlier
    # pass used two awaits (to_thread mutate, then the async save_node())
    # to avoid calling MemArkGraphStore's then-private sync save; the
    # summarizer pass made that sync save public (save_node_sync) because
    # the lock has to cover both steps. Ordering is unchanged: mutate
    # always completes before persist starts.

    async def write(self, user_id: str, persona_id: str, turn: Turn) -> None:
        await asyncio.to_thread(self._write_sync, user_id, persona_id, turn)

    def _write_sync(self, user_id: str, persona_id: str, turn: Turn) -> None:
        with self._lock:
            _node_id, persist_node = self._embed_and_mutate(user_id, persona_id, turn)
            # If this raises, the in-memory mutation above already stands --
            # deliberate, per docs/MEMARK_RETRIEVAL_ADAPTER.md section 8:
            # GraphMemoryManager has no supported node-removal API (confirmed
            # by auditing its full method list), and MemoryRepository
            # .remove_node() -- which looks like one -- only removes the node
            # from the networkx graph, not from .nodes, so using it would
            # leave a WORSE inconsistency than not calling it at all. Not
            # caught here; propagates to the caller unswallowed.
            self._graph_store.save_node_sync(user_id, persona_id, persist_node)

    def _embed_and_mutate(
        self, user_id: str, persona_id: str, turn: Turn
    ) -> tuple[str, MemArkNode]:
        scope = self._ensure_scope(user_id, persona_id)
        embedding = self._embed(turn.text)

        # create_node() has no embedding or timestamp parameters --
        # confirmed directly against its signature
        # (engine/core/graph_manager.py) -- so both are set via direct
        # attribute mutation immediately after, matching MemArk's own
        # example pattern for embedding (examples/retrieval_demo.py) and
        # extended to timestamp for this project's own dedup requirement
        # (docs/MEMARK_RETRIEVAL_ADAPTER.md section 5: _dedup_retrieved_
        # memory compares full Turn equality including timestamp, so a
        # freshly auto-generated node.timestamp_created would silently
        # break dedup against recent_turns).
        node_id = scope.manager.create_node(
            node_type=_EPISODIC_NODE_TYPE,
            summary=turn.text,
            metadata={"role": turn.role},
        )
        node = scope.manager.get_node(node_id)
        node.embedding = embedding
        node.timestamp_created = turn.timestamp
        node.timestamp_updated = turn.timestamp

        scope.vector_search.add_embedding(
            node_id=node_id,
            embedding=embedding,
            metadata={
                "node_type": _EPISODIC_NODE_TYPE,
                "salience": node.salience_score,
                "is_dormant": False,
            },
        )

        persist_node = MemArkNode(
            id=node_id,
            node_type=node.node_type,
            summary=node.summary,
            embedding=node.embedding,
            timestamp_created=node.timestamp_created,
            timestamp_updated=node.timestamp_updated,
            importance_score=node.importance_score,
            salience_score=node.salience_score,
            confidence_score=node.confidence_score,
            retrieval_count=node.retrieval_count,
            last_accessed=node.last_accessed,
            parent_id=node.parent_id,
            child_ids=list(node.child_ids),
            raw_conversation_refs=list(node.raw_conversation_refs),
            metadata=dict(node.metadata),
        )
        return node_id, persist_node

    # --- query ----------------------------------------------------------
    # One asyncio.to_thread call for the whole method -- unlike write(),
    # query() never touches MemArkGraphStore/SQLite at all (see design
    # note section 3), so there's no analogous friction; it needs
    # to_thread purely for the CPU-bound work (embedding inference,
    # concept-seed scan, activation propagation, fallback scan).

    async def query(
        self, user_id: str, persona_id: str, query: str, k: int = 5
    ) -> list[Turn]:
        return await asyncio.to_thread(self._query_sync, user_id, persona_id, query, k)

    def _query_sync(
        self, user_id: str, persona_id: str, query: str, k: int
    ) -> list[Turn]:
        with self._lock:
            return self._query_locked(user_id, persona_id, query, k)

    def _query_locked(
        self, user_id: str, persona_id: str, query: str, k: int
    ) -> list[Turn]:
        scope = self._ensure_scope(user_id, persona_id)
        query_embedding = self._embed(query)

        graph_scores = self._graph_retrieval_scores(scope, query_embedding)
        fallback_scores = self._fallback_scores(scope.manager, query)
        merged = self._merge_scores(graph_scores, fallback_scores)

        ranked_ids = sorted(merged, key=lambda nid: merged[nid], reverse=True)[:k]

        turns: list[Turn] = []
        for node_id in ranked_ids:
            node = scope.manager.get_node(node_id)
            if node is None:
                continue
            turns.append(self._node_turn(node))
        return turns

    def _graph_retrieval_scores(
        self, scope: _Scope, query_embedding: list[float]
    ) -> dict[str, float]:
        seed_id = self._find_concept_seed(scope.manager, query_embedding)
        if seed_id is None:
            # No concept exists yet (or none have embeddings) -- a new/
            # empty scope, or one this pass's write() alone has ever
            # populated (write() only ever creates "episodic" nodes; see
            # module docstring -- concept/hierarchy structure is
            # explicitly deferred to docs/MEMARK_COMPANION_EXTENSIONS.md
            # sections 2b/2c). Not an error -- an empty graph-retrieval
            # half, per docs/MEMARK_RETRIEVAL_ADAPTER.md section 1.
            return {}
        seed_node = scope.manager.get_node(seed_id)
        result = scope.activation_engine.propagate_activation(start_node=seed_node)
        return dict(result["nodes"])

    def _find_concept_seed(
        self, manager: GraphMemoryManager, query_embedding: list[float]
    ) -> str | None:
        """Reimplements the benchmark evaluators'
        `_find_concept_seed_node` logic as this adapter's own small,
        independent function -- confirmed private to
        `HotpotQAEvaluator`/`WikiMultihopEvaluator`, not part of MemArk's
        public API, so it can't be imported or called (see
        `docs/MEMARK_RETRIEVAL_ADAPTER.md` section 1). Same cosine-
        similarity-over-concept-nodes logic, same formula."""
        import numpy as np

        q = np.array(query_embedding)
        best_id: str | None = None
        best_sim = -1.0
        for node_id, node in manager.nodes.items():
            if node.node_type != _CONCEPT_NODE_TYPE or not node.embedding:
                continue
            n = np.array(node.embedding)
            denom = float(np.linalg.norm(q) * np.linalg.norm(n)) + 1e-9
            sim = float(np.dot(q, n) / denom)
            if sim > best_sim:
                best_sim = sim
                best_id = node_id
        return best_id

    def _fallback_scores(
        self, manager: GraphMemoryManager, query_text: str
    ) -> dict[str, float]:
        """Recency + keyword-overlap scan over episodic (turn-derived)
        nodes only -- the recall floor half of hybrid retrieval, per
        docs/MEMARK_COMPANION_EXTENSIONS.md section 2 / docs/
        MEMARK_RETRIEVAL_ADAPTER.md section 2. Reads the same in-memory
        `manager.nodes` the graph half reads -- no separate index."""
        query_words = {w for w in query_text.lower().split() if w}
        now = datetime.now(UTC)
        scores: dict[str, float] = {}
        for node_id, node in manager.nodes.items():
            if node.node_type != _EPISODIC_NODE_TYPE:
                continue
            summary_words = {w for w in node.summary.lower().split() if w}
            overlap = len(query_words & summary_words)
            # node.timestamp_created is aware for every write()-created
            # node (turn.timestamp always is), but MemoryNode's own
            # dataclass default (datetime.utcnow(), for a node created
            # directly via create_node() without the write()-path
            # override) is naive -- found by actually running this
            # against a hand-built test fixture, not assumed. Treat a
            # naive timestamp as already being UTC rather than letting
            # the subtraction raise.
            created = node.timestamp_created
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            age_seconds = max((now - created).total_seconds(), 0.0)
            recency = 1.0 / (1.0 + age_seconds)
            scores[node_id] = overlap + recency
        return scores

    def _merge_scores(
        self, graph_scores: dict[str, float], fallback_scores: dict[str, float]
    ) -> dict[str, float]:
        """Min-max normalize each half independently, then take the max
        per node id across both halves -- a recall floor (either signal
        being confident is enough), not a weighted sum, per
        docs/MEMARK_RETRIEVAL_ADAPTER.md section 2's explicit reasoning."""
        normalized_graph = self._min_max_normalize(graph_scores)
        normalized_fallback = self._min_max_normalize(fallback_scores)
        merged: dict[str, float] = {}
        for node_id in set(normalized_graph) | set(normalized_fallback):
            merged[node_id] = max(
                normalized_graph.get(node_id, 0.0),
                normalized_fallback.get(node_id, 0.0),
            )
        return merged

    def _min_max_normalize(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo = min(scores.values())
        hi = max(scores.values())
        if hi == lo:
            # Every candidate tied -- no variance to normalize against.
            # Treated as all fully relevant (1.0), not all zero: a tie at
            # the top of one half's own ranking shouldn't lose to the
            # other half purely because this half had no way to
            # differentiate among genuinely-found candidates.
            return {node_id: 1.0 for node_id in scores}
        return {node_id: (score - lo) / (hi - lo) for node_id, score in scores.items()}

    # --- list / replace (RollingSummarizer) --------------------------
    # See docs/ROLLING_SUMMARIZER.md section 6.2. Only episodic nodes are
    # turns; a summary turn is an episodic node with role "system", not
    # MemArk's own "summary" node type, because _fallback_scores scans only
    # episodic nodes and a "summary" node would never be retrieved.

    @staticmethod
    def _node_turn(node: Any) -> Turn:
        return Turn(
            role=node.metadata.get("role", "user"),
            text=node.summary,
            timestamp=node.timestamp_created,
        )

    async def list_turns(self, user_id: str, persona_id: str) -> list[Turn]:
        return await asyncio.to_thread(self._list_turns_sync, user_id, persona_id)

    def _list_turns_sync(self, user_id: str, persona_id: str) -> list[Turn]:
        with self._lock:
            scope = self._ensure_scope(user_id, persona_id)
            turns = [
                self._node_turn(node)
                for node in scope.manager.nodes.values()
                if node.node_type == _EPISODIC_NODE_TYPE
            ]
        return sorted(turns, key=lambda turn: turn.timestamp)

    async def replace_turns(
        self, user_id: str, persona_id: str, old: Sequence[Turn], new: Turn
    ) -> bool:
        return await asyncio.to_thread(
            self._replace_turns_sync, user_id, persona_id, list(old), new
        )

    def _replace_turns_sync(
        self, user_id: str, persona_id: str, old: list[Turn], new: Turn
    ) -> bool:
        with self._lock:
            scope = self._ensure_scope(user_id, persona_id)
            ids_by_turn: dict[Turn, list[str]] = {}
            for node_id, node in scope.manager.nodes.items():
                if node.node_type == _EPISODIC_NODE_TYPE:
                    ids_by_turn.setdefault(self._node_turn(node), []).append(node_id)
            delete_ids: list[str] = []
            for turn in old:
                candidates = ids_by_turn.get(turn)
                if not candidates:
                    return False
                delete_ids.append(candidates.pop())

            embedding = self._embed(new.text)
            new_node = MemArkNode(
                id=str(uuid.uuid4()),
                node_type=_EPISODIC_NODE_TYPE,
                summary=new.text,
                embedding=embedding,
                timestamp_created=new.timestamp,
                timestamp_updated=new.timestamp,
                importance_score=0.5,
                salience_score=0.5,  # MemoryNode's own defaults
                confidence_score=1.0,
                retrieval_count=0,
                last_accessed=new.timestamp,
                parent_id=None,
                child_ids=[],
                raw_conversation_refs=[],
                metadata={"role": new.role},
            )
            self._graph_store.replace_nodes_sync(user_id, persona_id, delete_ids, new_node)
            # Same reasoning as delete() below: GraphMemoryManager has no
            # supported single-node removal, so drop the whole scope and let
            # it re-hydrate from SQLite (now the truth) on next use.
            self._scopes.pop((user_id, persona_id), None)
            return True

    # --- delete -----------------------------------------------------
    # Out of scope for docs/MEMARK_RETRIEVAL_ADAPTER.md's original
    # query()/write()-only pass; implemented in a later pass, matching
    # VectorMemoryStore.delete()'s own contract exactly (persona_id=None
    # means "every scope for this user"). Two things need clearing, not
    # just one: the persisted SQLite file(s) (MemArkGraphStore.delete(),
    # which already exists for this — checked its surface before adding
    # anything new there) and this adapter's own in-memory `_scopes`
    # cache (the live GraphMemoryManager/FaissVectorStore/etc. a deleted
    # scope must not go on serving queries against). Dropping the whole
    # `_Scope` object is deliberate, not a partial in-place clear —
    # GraphMemoryManager has no supported single-node removal API either
    # (see docs/MEMARK_RETRIEVAL_ADAPTER.md section 8), so discarding the
    # entire object and letting a fresh one lazily rehydrate on next use
    # avoids that unsupported-surgery problem entirely rather than
    # working around it.

    async def delete(self, user_id: str, persona_id: str | None = None) -> None:
        await asyncio.to_thread(self._delete_sync, user_id, persona_id)

    def _delete_sync(self, user_id: str, persona_id: str | None) -> None:
        with self._lock:
            self._graph_store.delete(user_id, persona_id)
            if persona_id is not None:
                self._scopes.pop((user_id, persona_id), None)
                return
            for key in [k for k in self._scopes if k[0] == user_id]:
                del self._scopes[key]
