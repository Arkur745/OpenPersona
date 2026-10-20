# MemArk local persistence — design note (implemented)

**Status: implemented** as `companion/memory/memark_store.py`'s
`MemArkGraphStore`, and, since `docs/MEMARK_RETRIEVAL_ADAPTER.md`'s
adapter pass, wired into a real, selectable `MemoryStore` backend — see
`docs/ARCHITECTURE.md`'s "MemoryStore" section. The design below is left
as originally written; nothing in it changed on the way to shipping.
Later (2026-10-19), rolling summarization added two public methods:
`save_node_sync` (the synchronous body of a node write) and
`replace_nodes_sync` (insert one summary node and delete the old nodes
and their edges in a single SQLite transaction). See
`docs/ROLLING_SUMMARIZER.md` section 6.2.

Same gate as every prior MemArk note. This is the P0 prerequisite
`docs/MEMARK_COMPANION_EXTENSIONS.md` section 1 identified: without it,
nothing built on MemArk survives a process restart, which makes it
unusable as this project's memory backend regardless of anything else in
that note. Targeted, not another full source audit — the prior two notes
already established MemArk's shape; this one only goes back to source
where a specific design decision here needs it.

## 1. Exact data shape to persist

Grounded directly in the actual dataclasses and `GraphMemoryManager`'s
actual attributes (re-confirmed against source, not assumed):

- **`MemoryNode`** (`engine/core/memory_node.py`): `id`, `node_type`,
  `summary`, `embedding: Optional[List[float]]`, `timestamp_created`,
  `timestamp_updated`, `importance_score`, `salience_score`,
  `confidence_score`, `retrieval_count`, `last_accessed`, `parent_id`,
  `child_ids`, `raw_conversation_refs`, `metadata: Dict`. Every field
  needs to survive — this is the entire "memory," not just its text.
- **`MemoryEdge`** (`engine/core/memory_edge.py`): `id`, `source_node`,
  `target_node`, `relation_type`, `weight`, `confidence`, `created_at`,
  `last_accessed`, `retrieval_count`, `decay_rate`.
- **`GraphMemoryManager`'s own attributes** (`engine/core/graph_manager
  .py:15-26`): `self.nodes: Dict[str, MemoryNode]`, `self.edges: Dict[str,
  MemoryEdge]`, `self.graph: nx.DiGraph()`, plus `neo4j_repository`/
  `vector_search`/`_id_rng`, none of which need persisting themselves.
  `self.graph` is **derived, not independent state** — confirmed by
  reading `create_node`/`create_edge`: both call `self.graph.add_node(...)`
  /`self.graph.add_edge(...)` immediately after updating `self.nodes`/
  `self.edges`, mirroring them. Only `nodes` and `edges` need persisting;
  `self.graph` is rebuilt for free by replaying the same calls on load.

**`parent_id`/`child_ids` are a bookkeeping cache the caller maintains,
not something `create_edge` derives automatically** — confirmed in
`examples/minimal_example.py`: `concept_node.child_ids.append(summary_id)`
and `create_edge(concept_id, summary_id, "hierarchical")` are two
separate, manual steps, not one. `GraphLoader.load_from_neo4j`
(`engine/persistence/graph_loader.py:71-79`) doesn't persist/reload
`parent_id`/`child_ids` as their own stored fields at all — it
reconstructs them purely by replaying `hierarchical`-typed edges on load
(`if rel_type == "hierarchical": child_node.parent_id = source_id; ...`).
**This design follows the same pattern**: persist edges (which already
carry `relation_type`), don't persist `parent_id`/`child_ids` as separate
columns, reconstruct them from hierarchical edges on load exactly as
`GraphLoader` already does. Avoids a second, independently-writable copy
of the same relationship that could drift out of sync with the edges
themselves.

**The FAISS/embedding question, resolved concretely**: `MemoryNode
.embedding` already holds the full embedding vector on the node itself.
`FaissVectorStore` (`engine/retrieval/faiss_store.py`) holds nothing a
persisted node doesn't already give us — its `node_ids`/`embeddings`/
`metadata_map` are populated entirely via `add_embedding()`, and
`GraphLoader.load_from_neo4j` already demonstrates the intended
reconstruction pattern: `if node.node_type in indexable_node_types and
node.embedding: vector_search.add_embedding(node_id=node.id,
embedding=np.array(node.embedding), metadata={...})`. So: **persist
`MemoryNode.embedding` as part of the node's own row (a plain list of
floats), and rebuild the FAISS index on load by replaying `add_embedding`
calls from that persisted data.** No separate FAISS index file, and no
re-embedding (no need to re-run the embedding model at all on load) —
the vector *is* the persisted data, not something re-derived from it.

## 2. File format: SQLite, not JSON

`sqlite3` is Python's standard library — available with zero new
dependency regardless of whether Chroma/chromadb is installed at all
(more precise than "transitively available via Chroma": it doesn't need
Chroma present to be usable). That said, the real reason to pick it isn't
dependency cost, it's **write cost as the graph grows**, which is where
the two candidates actually diverge:

- **JSON** — simple and human-inspectable, but a JSON document has no
  incremental-write story: adding one node means re-serializing and
  rewriting the *entire* persisted graph, every time. For a companion
  relationship meant to accumulate memory over weeks or months, this is
  an ever-growing per-write cost precisely as the thing being persisted
  gets more valuable — the opposite of what a hot write path should do.
- **SQLite** — a single `INSERT`/`UPDATE` per node or edge, independent
  of total graph size. Two tables: `nodes` (one row per `MemoryNode`
  field, `metadata`/`embedding` serialized to a `TEXT` column each —
  SQLite has no native list/dict column type, JSON-encoding just those
  two fields is the standard, unavoidable exception, not a reason to
  JSON-encode the whole record) and `edges` (one row per `MemoryEdge`
  field). Also gets real crash-safety machinery for free — see section 3.

**Recommendation: SQLite.** One file per persisted store (see section 5
for scoping), written via the stdlib `sqlite3` module.

## 3. Write timing and crash safety

**Timing: per-turn, immediate — matching `docs/MEMORY_STORE.md` section
5's existing precedent for Chroma**, and for the same underlying reason
that precedent gives (`write()` is "cheap, no network, no latency reason
to defer") — which only holds for MemArk *because* of the SQLite choice
above. A single-row `INSERT` is cheap regardless of graph size; a
full-document JSON rewrite would not have been, and would have argued
for batching instead. So this recommendation is downstream of section 2,
not independent of it.

**Crash safety**: don't hand-roll temp-file-plus-atomic-rename here —
that's the right pattern for a whole-file-rewrite format (which is
exactly what JSON would have needed), but SQLite already provides a
strictly better mechanism for exactly this access pattern: each write
wrapped in a transaction, committed via SQLite's own atomic commit. A
process killed mid-write leaves the last transaction rolled back
automatically on next open — never a half-written row, never a
truncated/corrupt file from an interrupted write. Recommend **WAL mode**
specifically (`PRAGMA journal_mode=WAL`) — well-suited to this app's
single-writer, occasional-reader pattern, and reduces fsync overhead
relative to the default rollback-journal mode without weakening the
crash-safety guarantee.

**One concrete implementation detail this surfaces, not just a footnote**:
`sqlite3` connections are single-thread by default
(`sqlite3.ProgrammingError` if used from a different thread than the one
that opened them). `VectorMemoryStore` (`companion/memory/store.py`)
already establishes the pattern to follow here — one lazily-created
client (`self._client`), reused across every `query`/`write` call via
`asyncio.to_thread`, regardless of which thread-pool worker executes each
call (verified by reading `_ensure_client`/`query`/`write` directly).
Matching that for SQLite means opening the connection once with
`check_same_thread=False`, relying on the fact that this store, like
Chroma's, only ever has one write in flight at a time (each `await
asyncio.to_thread(_write)` completes before the next call), never truly
concurrent access from multiple threads at once.

**The real-time-audio interaction — flagged explicitly, with the actual
call sites, not a general worry.** `Orchestrator.handle_voice_turn`
awaits `self.memory.write(...)` twice, both directly on the main task
that also runs the response-generation loop and creates the barge-in
watcher: once for the user's transcript right after an utterance ends
(`orchestrator.py:429`, *before* `watcher = asyncio.create_task
(watch_for_barge_in())` is even created), and once for the assistant's
full reply after generation finishes (`orchestrator.py:488`, right before
looping back to listening). asyncio is single-threaded and cooperative —
**a blocking, non-threaded disk write at either point would freeze the
entire event loop for its duration, including `_pump_mic`'s real-time
frame draining**, not just make the response feel laggy. This is exactly
the class of bug this project has already been bitten by once (the
mic-ended deadlock two passes ago) — a real-time-timing interaction that
looked like ordinary sequential code until a real, continuous audio
source exposed it. The mitigation is not new to invent: it's the same
one `VectorMemoryStore.write()` already uses for the identical problem
with Chroma — **wrap the blocking sqlite3 write in `asyncio.to_thread`**,
exactly matching the existing, already-proven pattern, not a new
technique for this backend to introduce.

## 4. Load-on-startup shape

**Where**: `companion/app.py`'s `build_memory(config)` is the right seam
— it's already the single place `MemoryStore` gets constructed, called
once from `build_orchestrator`, which itself runs synchronously in
`__main__.py`'s `main()` **before** `asyncio.run(...)` starts the voice
loop. This matters: at that point no event loop exists yet, `_pump_mic`
isn't running, nothing time-sensitive is competing for the thread — so,
unlike the per-turn write path in section 3, a **synchronous, blocking
load at startup is not a real-time-correctness risk**, only a
startup-latency one. No `asyncio.to_thread` needed for the load itself,
only for per-turn writes/queries during a live session.

**Rough cost estimate** (directional, not measured, flagged as such
rather than presented as precise — matches this project's established
practice of not claiming a number it hasn't actually measured):
reconstruction is dominated by two things — deserializing rows into
`MemoryNode`/`MemoryEdge` dataclass instances (SQLite row reads are fast;
Python object construction is the real cost, roughly linear in node/edge
count) and rebuilding the FAISS flat index via repeated `add_embedding`
calls (also linear, and FAISS's flat index has no build-time indexing
overhead to speak of). For a graph sized to a genuinely long companion
relationship — thousands, not millions, of nodes — this should land
somewhere in the tens-to-low-hundreds of milliseconds on ordinary
hardware, not seconds. This needs to be measured for real before being
trusted, the same way `docs/MEMARK_BACKEND.md` section 4 declined to
trust MemArk's own unmeasured latency claims.

**Missing, empty, and corrupted are three different cases, handled
differently — not collapsed into one "start fresh" path:**

- **File doesn't exist** (first run for this `user_id`/`persona_id`):
  create a new, empty SQLite file with the schema, exactly analogous to
  Chroma's `get_or_create_collection` already doing the equivalent thing
  silently on first use. Not an error.
- **File exists but is empty/has the schema with zero rows**: also not
  an error — a real, valid state (e.g. a scope that's been created but
  never written to).
- **File exists and is genuinely corrupted** (fails to open, a query
  raises `sqlite3.DatabaseError`, a row fails to deserialize into
  `MemoryNode`/`MemoryEdge`): **must fail loudly and specifically, never
  silently fall back to an empty graph.** Silently treating corruption as
  "nothing was ever stored" would quietly destroy a user's real memory
  data from their perspective — worse than any recall gap this whole
  MemArk effort started from, and precisely the failure mode this
  project has consistently refused to allow elsewhere (`AudioDeviceError`
  wrapping every `PortAudioError` with an actionable message rather than
  letting a raw exception or silent fallback reach the user is the same
  principle, already established convention — see `companion/audio/errors
  .py`). The error needs to name the exact file path and the specific
  failure, so a user or operator can investigate or restore from backup,
  not guess why their companion "forgot everything."

## 5. Scoping: one store per `(user_id, persona_id)`, same granularity as Chroma

This matches `VectorMemoryStore`'s existing scoping and should for the
same reason: schema.md's requirement that memory never leak across a
different persona. Concretely, one SQLite file per scope (e.g.
`{persist_dir}/mem-{sha256(user_id)[:16]}-{persona_id}.db`, mirroring
Chroma's existing collection-naming pattern in `companion/memory/store
.py`'s `_collection_name`), rather than one shared file with a
scope-column filter.

**Graph structure does make this slightly less free than Chroma's
collection-per-scope, worth being honest about rather than assuming it
transfers cleanly**: Chroma's collections are independent by
construction — a collection is just a named bucket, nothing about one
collection's existence or content depends on another's. A *graph*, by
contrast, is edges between specific node IDs — as long as every edge
persisted for a given scope only ever references node IDs created within
that same scope's own file (true by construction here, since each scope
gets its own `GraphMemoryManager`/file and node IDs are UUIDs generated
per-instance, never shared across scopes), the graphs stay fully
independent too — there's no cross-scope edge to accidentally create.
So the same **structural** leak guarantee Chroma gets from separate
collections holds here too, for the same reason (separate storage, not
just a filter) — it isn't automatic from "it's a graph," it's true
because each scope gets its own completely separate `GraphMemoryManager`
instance and file, never a shared graph with scope-tagged nodes.

## 6. Confirming this stays entirely in this project's own adapter code

Everything above is a new module this project would write and own —
reading/writing SQLite via the stdlib `sqlite3` module, using only
`MemoryNode`/`MemoryEdge`'s public dataclass fields and
`GraphMemoryManager`'s public `nodes`/`edges`/`create_node`/`create_edge`
surface. The reconstruction algorithm is modeled on `GraphLoader
.load_from_neo4j`'s own logic (same field mapping, same
hierarchical-edges-reconstruct-`parent_id`/`child_ids` approach) but is a
**new, independent implementation** — nothing here edits, imports as a
base class, or depends on `engine/persistence/graph_loader.py` itself,
which stays exactly as it is, hardwired to Neo4j, untouched. Nothing
proposed in this note requires editing any file in `Arkur745/MemEngine`.

## Recommendation summary

1. **File format: SQLite** (section 2) — stdlib, zero new dependency,
   genuinely incremental writes (unlike JSON, which would force a
   full-file rewrite per turn as the graph grows).
2. **Write timing: per-turn, immediate** (section 3) — matches the
   existing Chroma precedent, justified the same way, and only cheap
   enough to justify that because of the SQLite choice.
3. **Crash safety: SQLite's own transactional commit + WAL mode** — not
   a hand-rolled temp-file/atomic-rename scheme, which is the right
   pattern for whole-file formats but unnecessary machinery once SQLite's
   own guarantees are available.
4. **The real-time-audio risk is real and specific, not hypothetical**:
   both `memory.write()` call sites inside `handle_voice_turn` must wrap
   their SQLite write in `asyncio.to_thread`, exactly matching
   `VectorMemoryStore.write()`'s already-proven pattern for the identical
   problem — not a new technique to design, a known one to reuse.
5. **Corruption must fail loudly**, never silently present as an empty
   graph — first-run-missing and genuinely-corrupted are different cases
   requiring different handling, matching this project's existing
   no-silent-failure convention elsewhere (`AudioDeviceError`).
6. Scoping and adapter-only boundaries both hold at the same strength as
   every prior MemArk note — one store per `(user_id, persona_id)`,
   nothing touching MemEngine's own repo.

Stopping here per the gate. Want to discuss any of this, or should the
next step be sequencing this against the recall-recovery work in
`docs/MEMARK_COMPANION_EXTENSIONS.md`?
