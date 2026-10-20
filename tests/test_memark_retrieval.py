"""Tests for companion/memark/retrieval.py (MemArkMemoryStore). See
docs/MEMARK_RETRIEVAL_ADAPTER.md for the design this implements.

Most tests use _FakeEmbeddingFunction (deterministic, instant, but with
real semantic structure — shared words produce higher cosine similarity,
unlike random vectors) rather than chromadb's real DefaultEmbeddingFunction,
matching this project's established fast-offline-tests-plus-one-real-smoke-
test pattern. test_real_embedding_function_smoke is the one exception,
gated behind pytest.importorskip("chromadb"), confirming the actual
default wiring works end to end — it's what caught a real bug during
implementation (DefaultEmbeddingFunction returns numpy.float32 elements,
which MemArkGraphStore's json.dumps() can't serialize; fixed in
retrieval.py's _embed).

companion.memark.retrieval imports the vendored engine.core.graph_manager
at module load time, which needs networkx (and, transitively through
other vendored files this module also imports, faiss) — importorskip
these *before* that import happens, module-level, or importing this test
file at all raises ModuleNotFoundError instead of skipping cleanly when
the `memory-memark` extra isn't installed, breaking plain `pytest`
collection for the whole project, not just this file.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

pytest.importorskip("networkx")
pytest.importorskip("faiss")

from companion.core.interfaces.types import Turn
from companion.memark.retrieval import MemArkMemoryStore
from companion.memory.memark_store import MemArkGraphStore


class _FakeEmbeddingFunction:
    """Encodes text as a normalized bag-of-words indicator over a small
    fixed vocabulary — not random vectors: texts sharing words produce
    higher cosine similarity, texts sharing none produce ~0, so tests
    relying on similarity behaving meaningfully aren't vacuous."""

    _VOCAB: ClassVar[list[str]] = [
        "peanuts", "allergic", "allergy", "sister", "maya", "marathon",
        "june", "birthday", "mother", "call", "remind", "capital",
        "france", "weather", "sunny", "dog", "cat", "hello", "friend",
        "food", "safety",
    ]

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        words = set(text.lower().replace("?", "").replace(".", "").split())
        vec = [1.0 if w in words else 0.0 for w in self._VOCAB]
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0:
            return vec
        return [v / norm for v in vec]


def _store(tmp_path) -> MemArkMemoryStore:
    return MemArkMemoryStore(
        persist_dir=str(tmp_path), embedding_function=_FakeEmbeddingFunction()
    )


# --- basic round trip + empty scope ----------------------------------


@pytest.mark.asyncio
async def test_write_then_query_returns_the_turn(tmp_path):
    store = _store(tmp_path)
    await store.write("u1", "p1", Turn(role="user", text="I am allergic to peanuts"))

    results = await store.query("u1", "p1", "allergic peanuts", k=5)

    assert len(results) == 1
    assert results[0].role == "user"
    assert results[0].text == "I am allergic to peanuts"


@pytest.mark.asyncio
async def test_query_on_empty_new_scope_returns_empty_list_not_an_error(tmp_path):
    store = _store(tmp_path)

    results = await store.query("nobody", "nothing", "anything at all", k=5)

    assert results == []


# --- dedup equality (docs/MEMARK_RETRIEVAL_ADAPTER.md section 5) -----


@pytest.mark.asyncio
async def test_query_result_is_dedup_equal_to_the_original_turn(tmp_path):
    """Proves _dedup_retrieved_memory (orchestrator.py) would actually
    catch this: it compares full Turn equality (role, text, AND
    timestamp — Turn is frozen/structurally comparable). Without the
    node.timestamp_created = turn.timestamp override in
    _embed_and_mutate, this would fail (node.timestamp_created would be
    whatever create_node() auto-generated instead)."""
    store = _store(tmp_path)
    original = Turn(role="user", text="I am allergic to peanuts")
    await store.write("u1", "p1", original)

    results = await store.query("u1", "p1", "allergic peanuts", k=5)

    assert len(results) == 1
    assert results[0] == original  # full dataclass equality, not just text


# --- write() failure handling (design note section 8) -----------------


@pytest.mark.asyncio
async def test_write_failure_propagates_and_in_memory_mutation_stands(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(MemArkGraphStore, "save_node_sync", _raise)

    turn = Turn(role="user", text="I am allergic to peanuts")
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        await store.write("u1", "p1", turn)

    # Per section 8: no rollback attempted. The in-memory graph already
    # reflects the write even though persistence failed.
    scope = store._scopes[("u1", "p1")]
    summaries = [n.summary for n in scope.manager.nodes.values()]
    assert "I am allergic to peanuts" in summaries

    # And it's genuinely queryable within this same session, even though
    # it never reached disk.
    results = await store.query("u1", "p1", "allergic peanuts", k=5)
    assert any(t.text == "I am allergic to peanuts" for t in results)


# --- hydration / restart persistence -----------------------------------


@pytest.mark.asyncio
async def test_data_survives_a_fresh_store_instance_same_persist_dir(tmp_path):
    """The actual restart scenario: a brand-new MemArkMemoryStore
    (nothing shared with the first except the directory) must see what
    the first one wrote — this is the whole point of
    docs/MEMARK_PERSISTENCE.md existing."""
    first = _store(tmp_path)
    await first.write("u1", "p1", Turn(role="user", text="I am allergic to peanuts"))

    second = _store(tmp_path)
    results = await second.query("u1", "p1", "allergic peanuts", k=5)

    assert len(results) == 1
    assert results[0].text == "I am allergic to peanuts"


# --- delete(): matches VectorMemoryStore.delete()'s own contract -------
# Same test names/structure as tests/test_memory_store.py's delete()
# tests, for consistency across the two MemoryStore backends.


@pytest.mark.asyncio
async def test_delete_single_persona_removes_only_that_persona(tmp_path):
    store = _store(tmp_path)
    await store.write("u1", "mentor", Turn(role="user", text="mentor memory"))
    await store.write("u1", "friend", Turn(role="user", text="friend memory"))

    await store.delete("u1", "mentor")

    # Checked before any query() call below, deliberately: query() on a
    # deleted scope lazily re-hydrates it (same "get_or_create" laziness
    # VectorMemoryStore/Chroma has), which recreates an *empty* file —
    # real, expected behavior, but it would make a same-path
    # exists()-after-query() check pass for the wrong reason.
    assert not store._graph_store._db_path("u1", "mentor").exists()
    assert store._graph_store._db_path("u1", "friend").exists()

    assert await store.query("u1", "mentor", "memory", k=10) == []
    remaining = await store.query("u1", "friend", "memory", k=10)
    assert [t.text for t in remaining] == ["friend memory"]


@pytest.mark.asyncio
async def test_delete_without_persona_wipes_every_persona_for_that_user_only(tmp_path):
    store = _store(tmp_path)
    await store.write("u1", "mentor", Turn(role="user", text="u1 mentor memory"))
    await store.write("u1", "friend", Turn(role="user", text="u1 friend memory"))
    await store.write("u2", "mentor", Turn(role="user", text="u2 mentor memory"))

    await store.delete("u1")

    # Same ordering note as the single-persona test above — check file
    # absence before any query() re-hydrates (and recreates) a scope.
    assert not store._graph_store._db_path("u1", "mentor").exists()
    assert not store._graph_store._db_path("u1", "friend").exists()
    assert store._graph_store._db_path("u2", "mentor").exists()

    assert await store.query("u1", "mentor", "memory", k=10) == []
    assert await store.query("u1", "friend", "memory", k=10) == []
    remaining = await store.query("u2", "mentor", "memory", k=10)
    assert [t.text for t in remaining] == ["u2 mentor memory"]


@pytest.mark.asyncio
async def test_delete_on_a_user_with_no_memories_is_a_harmless_noop(tmp_path):
    store = _store(tmp_path)

    await store.delete("never-written-to")  # should not raise
    await store.delete("never-written-to", "mentor")  # should not raise


@pytest.mark.asyncio
async def test_delete_clears_in_memory_scope_not_just_the_file(tmp_path):
    """The regression this test exists to catch: if delete() only
    removed the persisted SQLite file and left the live GraphMemoryManager
    sitting in `_scopes`, a query() against the *same* store instance
    right after delete() would still serve results from that now-stale
    in-memory graph, even though the file backing it is gone — a subtler,
    worse version of "looks present but silently evaporates on restart"
    (docs/MEMARK_PERSISTENCE.md's own corruption-handling concern), just
    inverted: looks absent on restart, but not actually gone yet in the
    still-running process. Deliberately does NOT construct a fresh store
    for the post-delete query, unlike every other test in this file."""
    store = _store(tmp_path)
    await store.write("u1", "mentor", Turn(role="user", text="mentor memory"))
    assert (
        await store.query("u1", "mentor", "memory", k=10)
    )  # sanity: it was there before delete

    await store.delete("u1", "mentor")

    assert await store.query("u1", "mentor", "memory", k=10) == []


# --- merge rule: max, not weighted sum (docs/MEMARK_RETRIEVAL_ADAPTER.md
# section 2's explicit design choice) -----------------------------------


def test_merge_takes_max_per_half_not_a_weighted_sum(tmp_path):
    """Constructs a disagreement case, worked out by hand:

    g_top: graph=1.0 (normalized), absent from fallback.
    f_top: absent from graph, fallback=1.0 (normalized).
    generalist: moderately present in BOTH halves (graph raw 0.6 of 3,
        fallback raw 0.6 of 3 -- normalizes to 0.6 and 0.625
        respectively, given the raw score sets below).

    Under this store's actual max-merge: g_top and f_top both hit the
    ceiling (1.0, each the undisputed best of their own half) while
    generalist tops out at max(0.6, 0.625) = 0.625 -- ranked below both
    specialists, not above them.

    Under a hypothetical 50/50 weighted SUM instead: g_top = 0.5,
    f_top = 0.5, generalist = 0.5*0.6 + 0.5*0.625 = 0.6125 -- generalist
    would OUTRANK both specialists. The two rules disagree on the winner,
    not just the margin -- confirming the design note's reasoning:
    "one signal being confident should be sufficient, not diluted by the
    other signal's uncertainty about the same node" is a real, load-
    bearing choice, not a cosmetic one.
    """
    store = _store(tmp_path)

    graph_scores = {"g_top": 1.0, "generalist": 0.6, "g_floor": 0.0}
    fallback_scores = {"f_top": 0.9, "generalist": 0.6, "f_floor": 0.1}

    merged = store._merge_scores(graph_scores, fallback_scores)

    assert merged["g_top"] == 1.0
    assert merged["f_top"] == 1.0
    assert merged["generalist"] == pytest.approx(0.625)
    # The actual disagreement: generalist loses under max...
    assert merged["generalist"] < merged["g_top"]
    assert merged["generalist"] < merged["f_top"]
    # ...but would have won under a naive 50/50 weighted sum.
    weighted_sum_generalist = 0.5 * 0.6 + 0.5 * 0.625
    weighted_sum_g_top = 0.5 * 1.0 + 0.5 * 0.0
    assert weighted_sum_generalist > weighted_sum_g_top


def test_merge_normalize_handles_empty_and_tied_halves():
    instance = MemArkMemoryStore(persist_dir="unused", embedding_function=_FakeEmbeddingFunction())

    assert instance._min_max_normalize({}) == {}
    # All-tied raw scores: no variance to normalize against -- treated as
    # all fully relevant (1.0), not all zero (see _min_max_normalize's
    # own comment for the reasoning).
    assert instance._min_max_normalize({"a": 0.5, "b": 0.5}) == {"a": 1.0, "b": 1.0}


# --- ordering: descending by merged score --------------------------------


@pytest.mark.asyncio
async def test_results_ordered_by_merged_score_descending(tmp_path):
    store = _store(tmp_path)
    # Distinct keyword overlap against the query gives distinct,
    # predictable fallback scores (the only signal here -- no concept
    # nodes exist yet, so the graph half is empty for every query in
    # this test, per docs/MEMARK_RETRIEVAL_ADAPTER.md's own note that
    # write()-only graphs never populate the graph half).
    await store.write("u1", "p1", Turn(role="user", text="dog"))
    await store.write("u1", "p1", Turn(role="user", text="dog cat"))
    await store.write("u1", "p1", Turn(role="user", text="dog cat friend"))

    results = await store.query("u1", "p1", "dog cat friend", k=5)

    # Most keyword overlap with the query first.
    assert [t.text for t in results] == ["dog cat friend", "dog cat", "dog"]


# --- graph-retrieval half: concept seed + activation propagation --------
# write() alone never creates concept nodes (see retrieval.py's module
# docstring) -- these tests build a small graph directly against the
# adapter's own internal GraphMemoryManager to exercise the activation
# half in isolation, per the task's explicit ask for "a few episodic +
# concept nodes with real embeddings from the same embedding function."


@pytest.mark.asyncio
async def test_concept_seed_and_activation_surface_linked_episodic_nodes(tmp_path):
    store = _store(tmp_path)
    scope = store._ensure_scope("u1", "p1")
    ef = _FakeEmbeddingFunction()

    concept_id = scope.manager.create_node(
        node_type="concept", summary="food allergy safety"
    )
    concept_node = scope.manager.get_node(concept_id)
    concept_node.embedding = [float(x) for x in ef(["food allergy safety"])[0]]

    episodic_id = scope.manager.create_node(
        node_type="episodic",
        summary="I am allergic to peanuts",
        metadata={"role": "user"},
    )
    episodic_node = scope.manager.get_node(episodic_id)
    episodic_node.embedding = [float(x) for x in ef(["I am allergic to peanuts"])[0]]

    scope.manager.create_edge(
        source_node=concept_id, target_node=episodic_id, relation_type="hierarchical"
    )

    graph_scores = store._graph_retrieval_scores(
        scope, [float(x) for x in ef(["allergy food safety"])[0]]
    )

    assert episodic_id in graph_scores
    assert concept_id in graph_scores  # the seed itself is always activated


@pytest.mark.asyncio
async def test_concept_seed_returns_none_when_no_concepts_exist(tmp_path):
    """The common case for this pass: write() only ever creates episodic
    nodes, so a store used only through write()/query() never has any
    concept nodes at all -- the graph half must degrade to empty, not
    error."""
    store = _store(tmp_path)
    scope = store._ensure_scope("u1", "p1")

    seed = store._find_concept_seed(scope.manager, [0.1] * 21)

    assert seed is None


# --- fallback half: episodic-only, keyword + recency -----------------


def test_fallback_scores_ignore_non_episodic_nodes(tmp_path):
    store = _store(tmp_path)
    scope = store._ensure_scope("u1", "p1")
    scope.manager.create_node(node_type="concept", summary="dog cat friend topic")
    scope.manager.create_node(
        node_type="episodic", summary="dog cat friend", metadata={"role": "user"}
    )

    scores = store._fallback_scores(scope.manager, "dog cat friend")

    assert len(scores) == 1  # only the episodic node scored


# --- real embedding function: the one gated smoke test -----------------


@pytest.mark.asyncio
async def test_real_embedding_function_smoke(tmp_path):
    """Confirms the actual default wiring (chromadb's
    DefaultEmbeddingFunction, no embedding_function override) genuinely
    works end to end -- not just the fake used everywhere else in this
    file. This is the test that caught a real bug during implementation:
    DefaultEmbeddingFunction returns numpy.float32 elements, which
    MemArkGraphStore's json.dumps(node.embedding) can't serialize without
    _embed's explicit float(x) conversion."""
    pytest.importorskip("chromadb")
    store = MemArkMemoryStore(persist_dir=str(tmp_path))  # real embedding_function

    await store.write("u1", "p1", Turn(role="user", text="I am allergic to peanuts"))
    results = await store.query("u1", "p1", "what am I allergic to", k=5)

    assert len(results) == 1
    assert results[0].text == "I am allergic to peanuts"
