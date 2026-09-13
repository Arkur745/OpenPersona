"""Tests for companion/memory/store.py (VectorMemoryStore, Chroma-backed).

See docs/MEMORY_STORE.md for the design this implements — in particular
section 3 (ordering) and section 7 (scoping + test plan). Collection
naming (`_collection_name`/`_user_collection_prefix`) is pure logic and
runs without chromadb installed; everything else needs a real Chroma
collection and is gated behind the `store` fixture's
`pytest.importorskip("chromadb")`, same clean-skip pattern as every other
optional-dependency backend in this repo.
"""

import re

import pytest

from companion.core.interfaces.types import Turn
from companion.memory.store import (
    VectorMemoryStore,
    _collection_name,
    _user_collection_prefix,
)


@pytest.fixture
def store(tmp_path):
    pytest.importorskip("chromadb")
    return VectorMemoryStore(persist_dir=str(tmp_path))


# --- Collection naming: pure logic, no chromadb needed ---------------------


def test_collection_name_only_uses_chroma_safe_characters():
    # Chroma collection names are restricted to [a-zA-Z0-9._-], 3-512
    # chars — verified directly against a real client. user_id (e.g. an
    # email) isn't guaranteed to fit that, hence the hash.
    name = _collection_name("someone@example.com", "mentor")
    assert re.fullmatch(r"[a-zA-Z0-9._-]{3,512}", name)


def test_collection_name_differs_by_user_and_by_persona():
    assert _collection_name("u1", "mentor") != _collection_name("u2", "mentor")
    assert _collection_name("u1", "mentor") != _collection_name("u1", "friend")


def test_user_collection_prefix_matches_that_users_own_collection_names():
    name = _collection_name("u1", "mentor")
    assert name.startswith(_user_collection_prefix("u1"))
    assert not name.startswith(_user_collection_prefix("someone-else"))


def test_unimplemented_backend_raises_not_implemented():
    # No chromadb import happens before this check — works with or
    # without the extra installed, same "named but deferred" pattern as
    # STT's whisper_cpp (see docs/MEMORY_STORE.md section 1).
    with pytest.raises(NotImplementedError, match="lancedb"):
        VectorMemoryStore(backend="lancedb")


# --- Real Chroma: write, query ordering, roundtrip, scoping, delete --------


@pytest.mark.asyncio
async def test_write_then_query_returns_it(store):
    await store.write("u1", "mentor", Turn(role="user", text="I really like astronomy"))

    results = await store.query("u1", "mentor", "space and stars")

    assert any(t.text == "I really like astronomy" for t in results)


@pytest.mark.asyncio
async def test_query_empty_store_returns_empty_list(store):
    results = await store.query("u1", "mentor", "anything at all")

    assert results == []


@pytest.mark.asyncio
async def test_query_relevance_ordering_is_most_similar_first(store):
    """PromptBuilder's docstring assumes retrieved_memory arrives
    most-relevant-first (see docs/MEMORY_STORE.md section 3, verified
    there against a real collection). Asserts the returned *order*, not
    just that the right turns are present."""
    await store.write("u1", "mentor", Turn(role="user", text="My favorite food is pizza"))
    await store.write(
        "u1", "mentor", Turn(role="user", text="I love hiking in the mountains")
    )
    await store.write(
        "u1",
        "mentor",
        Turn(role="user", text="I went hiking last weekend and saw a bear"),
    )

    results = await store.query("u1", "mentor", "Tell me about hiking", k=3)
    texts = [t.text for t in results]

    assert set(texts) == {
        "My favorite food is pizza",
        "I love hiking in the mountains",
        "I went hiking last weekend and saw a bear",
    }
    # both hiking-related turns should rank ahead of the unrelated one
    assert texts[-1] == "My favorite food is pizza"
    assert texts[0] in (
        "I love hiking in the mountains",
        "I went hiking last weekend and saw a bear",
    )


@pytest.mark.asyncio
async def test_write_query_roundtrip_preserves_role_and_timestamp_exactly(store):
    """The orchestrator's dedup filter (docs/MEMORY_STORE.md section 4)
    relies on a Turn read back from query() comparing structurally equal
    (role, text, AND timestamp) to the same Turn still held in
    recent_turns — only holds if timestamp round-trips exactly."""
    original = Turn(role="assistant", text="a distinctly unique roundtrip sentence")
    await store.write("u7", "friend", original)

    results = await store.query("u7", "friend", "unique roundtrip sentence")

    assert original in results


@pytest.mark.asyncio
async def test_scoping_isolates_by_user_and_persona(store):
    await store.write("alice", "mentor", Turn(role="user", text="alice mentor secret"))
    await store.write("alice", "friend", Turn(role="user", text="alice friend secret"))
    await store.write("bob", "mentor", Turn(role="user", text="bob mentor secret"))

    alice_mentor = await store.query("alice", "mentor", "secret", k=10)
    alice_friend = await store.query("alice", "friend", "secret", k=10)
    bob_mentor = await store.query("bob", "mentor", "secret", k=10)

    assert [t.text for t in alice_mentor] == ["alice mentor secret"]
    assert [t.text for t in alice_friend] == ["alice friend secret"]
    assert [t.text for t in bob_mentor] == ["bob mentor secret"]


@pytest.mark.asyncio
async def test_delete_single_persona_removes_only_that_persona(store):
    await store.write("u1", "mentor", Turn(role="user", text="mentor memory"))
    await store.write("u1", "friend", Turn(role="user", text="friend memory"))

    await store.delete("u1", "mentor")

    assert await store.query("u1", "mentor", "memory", k=10) == []
    remaining = await store.query("u1", "friend", "memory", k=10)
    assert [t.text for t in remaining] == ["friend memory"]


@pytest.mark.asyncio
async def test_delete_without_persona_wipes_every_persona_for_that_user_only(store):
    await store.write("u1", "mentor", Turn(role="user", text="u1 mentor memory"))
    await store.write("u1", "friend", Turn(role="user", text="u1 friend memory"))
    await store.write("u2", "mentor", Turn(role="user", text="u2 mentor memory"))

    await store.delete("u1")

    assert await store.query("u1", "mentor", "memory", k=10) == []
    assert await store.query("u1", "friend", "memory", k=10) == []
    remaining = await store.query("u2", "mentor", "memory", k=10)
    assert [t.text for t in remaining] == ["u2 mentor memory"]


@pytest.mark.asyncio
async def test_delete_on_a_user_with_no_memories_is_a_harmless_noop(store):
    await store.delete("never-written-to")  # should not raise
    await store.delete("never-written-to", "mentor")  # should not raise
