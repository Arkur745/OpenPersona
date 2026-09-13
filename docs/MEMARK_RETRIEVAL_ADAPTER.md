# MemArk retrieval adapter — design note (implemented and wired)

**Status: implemented** as `companion/memark/retrieval.py`'s
`MemArkMemoryStore`, and now wired into `companion/app.py`'s
`build_memory` as a real, `config.yaml`-selectable `memory.backend:
memark` — see `docs/ARCHITECTURE.md`'s "MemoryStore" section for the
as-shipped summary. The design below (sections 1-8) is left as
originally written and approved; nothing in it changed on the way to
shipping.

Same gate as every prior MemArk note. This is the last piece before
`MemArkGraphStore` (persistence, already implemented and tested per
`docs/MEMARK_PERSISTENCE.md`) can become a real `MemoryStore` and get
wired into `app.py`/`orchestrator.py`. Targeted, building directly on
the prior four notes' decisions — re-verified against source only where
this note's own design actually turns on the answer, per the same
standard the persistence pass held itself to.

## 1. The concrete `query()` sequence

Target shape, unchanged: `query(user_id, persona_id, query: str, k: int =
5) -> list[Turn]` — `docs/MEMARK_BACKEND.md`'s Option A decision (keep
`MemoryStore`'s contract as-is, absorb the shape difference at the
adapter boundary) isn't being reopened.

Per `docs/MEMARK_COMPANION_EXTENSIONS.md` section 3's correction, built
against `ActivationEngine.propagate_activation()` and a concept-seed
step — never `RetrievalPipeline`/`retrieve_semantic_nodes`. Concrete
sequence, each step grounded:

1. **Embed the query text.** Recommend reusing Chroma's own default
   embedding function rather than MemArk's `EmbeddingManager`
   (sentence-transformers, pulls in torch — the exact cost
   `docs/MEMARK_BACKEND.md` section 4 flagged). Verified directly this
   pass, not assumed: `chromadb.utils.embedding_functions
   .DefaultEmbeddingFunction()` is callable standalone —
   `ef(["text"]) -> list[list[float]]` — no Chroma client, collection, or
   `persist_dir` required. Confirmed producing 384-dimension vectors,
   which conveniently matches MemArk's own `EMBEDDING_DIM: int = 384`
   default (`engine/config/default_config.py`) — no dimension mismatch
   to reconcile. `MemoryNode.embedding`/`FaissVectorStore.add_embedding`
   only ever see a plain `list[float]`; neither cares which model
   produced it.
2. **Find a concept seed.** `_find_concept_seed_node` — the benchmark
   evaluators' actual seeding logic — is a private method on
   `HotpotQAEvaluator`/`WikiMultihopEvaluator`, not part of MemArk's
   public API, so it can't be imported or called; it has to be
   **reimplemented**, not reused. It's small enough that this is a real,
   honest distinction worth naming rather than glossing over as reuse:
   a linear scan over `graph_manager.nodes.values()`, filtering to
   `node_type == "concept"`, computing cosine similarity between the
   query embedding and each candidate's `.embedding` (`np.dot(q, n) /
   (norm(q) * norm(n) + 1e-9)`), keeping the single best match. Uses
   only public `GraphMemoryManager`/`MemoryNode` data (`.nodes`,
   `.embedding`) — nothing private reached into.
3. **Propagate activation.** `ActivationEngine(memory_repository,
   base_activation=None, decay_factor=None, activation_threshold=None,
   max_depth=None)` (constructed once, held by the adapter — see section
   3) — `.propagate_activation(start_node=<seed concept node>,
   allowed_concept_ids=None)` returns `{"nodes": {node_id:
   activation_score}, "edges": {...}}`. If no concept seed was found
   (empty graph, or no concepts have embeddings yet), skip straight to
   an empty activated-nodes dict rather than erroring — a genuinely
   empty/new scope should look like "nothing to retrieve yet," not a
   failure.
4. **Rank into a node list.** Sort `activated_nodes` by score descending
   — this *is* MemArk's own relevance signal, no separate reranking
   step. This ranked list feeds into section 2's merge before becoming
   `list[Turn]`.

## 2. Hybrid retrieval, concretely

Per `docs/MEMARK_COMPANION_EXTENSIONS.md` section 2's recommendation
(option a, built first).

**Graph-retrieval half**: section 1's sequence, entirely against the
adapter's own live, in-memory `GraphMemoryManager` — see section 3 for
why that's where it lives, not `MemArkGraphStore`/SQLite.

**Fallback half**: a plain scan over the same `graph_manager.nodes`
dict, filtered to `node_type == "episodic"` (turn-derived nodes — see
this section's node-to-`Turn` mapping below for why only episodic nodes
are turn-shaped), scored by recency (`node.timestamp_created`, newer
scores higher) plus simple substring/keyword overlap between the query
text and `node.summary`. No separate index, no second data source —
same in-memory dict the graph half reads.

**Merge rule, stated explicitly** (this section needs one, not "combine
them somehow"): min-max normalize each half's scores independently to
`[0, 1]` within its own result set, then for any `node_id` appearing in
either (or both) halves, take the **max** of its two normalized scores
— a node strongly found by *either* signal ranks highly; a node found
by neither doesn't appear. Max, not a weighted sum, because the design
goal per the prior note is a recall *floor*: one signal being confident
should be sufficient, not diluted by the other signal's uncertainty
about the same node.

**Node-to-`Turn` mapping, and a revision to the prior note's guess.**
`docs/MEMARK_BACKEND.md` section 2 leaned toward stamping every
retrieved memory `Turn.role="user"`, flagged explicitly as an
approximation since a graph node "isn't cleanly either" role. Now that
this note fully specifies `write()`'s storage shape (see section 5),
that guess can be **revised, not just confirmed**: `write()` stores the
turn's real role in `node.metadata["role"]` at creation time (a
`create_node()` parameter that already exists, no extra step needed —
see `engine/core/graph_manager.py:48`'s `metadata: Optional[Dict] =
None`), so `query()`'s conversion reads it straight back —
`Turn(role=node.metadata["role"], text=node.summary,
timestamp=node.timestamp_created)`. No approximation needed at all for
this pass's scope, because every node this adapter's own `write()`
creates originates from exactly one `Turn` with a real role. (A future
pass adding consolidation — concept/summary nodes not tied to a single
original turn — would need a real answer for those; out of scope here,
per section 6.)

One naming/usage tension worth flagging plainly: `MemoryNode.summary`'s
field name suggests a shortened summary, but for dedup (section 5) and
faithful recall to work, it must hold the **full, verbatim** turn text,
not an actual summary. Not a blocker, just worth being explicit about
rather than let the field name mislead a future reader of the adapter
code.

## 3. Where retrieval executes relative to persistence

**Checked directly, not assumed — and the premise needs correcting.**
Re-read `MemArkGraphStore`'s actual state: it holds exactly
`self._connections` (a `dict[tuple[str, str], sqlite3.Connection]`) and
`self._connections_lock`. `save_node`/`save_edge` go straight to
SQLite via `asyncio.to_thread` — neither touches any in-memory graph
structure. **`MemArkGraphStore` has no in-memory graph state at all.**
So "every write already updates in-memory state before persisting" is
false as a description of `MemArkGraphStore` itself — it's pure
persistence, nothing more.

What this means concretely: the retrieval adapter must **own a
separate, live `GraphMemoryManager` (+ `VectorSearch`/FAISS) instance**
of its own — not something `MemArkGraphStore` provides. Hydrated once
at startup: `MemArkGraphStore.load()` (sync, per
`docs/MEMARK_PERSISTENCE.md` section 4) returns nodes/edges, fed into a
fresh `GraphMemoryManager` (replaying `create_node`/`create_edge`), then
`populate_vector_search()` rebuilds the FAISS index from persisted
embeddings. From that point on, `write()` must keep **both** in sync on
every call: mutate the in-memory graph immediately (so the *same
session's* very next `query()` sees it) and call `MemArkGraphStore
.save_node()`/`save_edge()` (so it survives a restart). Given this,
`query()` never needs to re-read SQLite — the in-memory graph is always
current within a session by construction, not by assumption.

**One more thing this forces into `write()`'s design, found while
tracing this, not assumed**: `GraphMemoryManager.create_node()`'s public
signature (`node_type, summary, importance_score=0.5, salience_score=0.5,
confidence_score=1.0, parent_id=None, metadata=None`) has **no
`embedding` or timestamp parameter at all** — confirmed directly against
its signature. `MemoryNode.timestamp_created`/`timestamp_updated`
auto-generate via `field(default_factory=datetime.utcnow)` inside
`create_node()`, with no way to override them through the public API.
So `write()` must, after calling `create_node()`, fetch the node back
(`graph_manager.get_node(node_id)`) and explicitly set **both**
`.embedding = <computed embedding>` **and** `.timestamp_created =
turn.timestamp` (see section 5 for why the timestamp override is
load-bearing, not cosmetic) via direct attribute mutation — the same
pattern MemArk's own example scripts already use for embedding
(`examples/retrieval_demo.py`), extended to timestamp for this
project's own dedup requirement.

**Does `query()` need `asyncio.to_thread`?** Yes — but for a different
reason than the framing in the task assumed. Under this design,
`query()` never touches SQLite or disk at all, so the persistence
design's disk-I/O justification doesn't apply here. It still needs
`to_thread` for a separate, equally real reason: it does synchronous,
CPU-bound work — query-embedding inference (ONNX), the concept-seed
cosine scan, activation-propagation graph traversal, and the fallback
scan — none of it disk I/O, all of it capable of occupying the event
loop's single thread for its duration if left unwrapped. This is the
exact justification `VectorMemoryStore.query()` already uses (embedding
inference alone already justified `to_thread` there, before this design
adds graph traversal on top). **Recommendation: wrap `query()` in
`asyncio.to_thread`, for CPU-bound-computation reasons, not disk-I/O
reasons.**

**`write()` needs it too, for the same reason, on its own distinct
half.** `write()` now has two pieces of blocking-or-CPU-bound work: (a)
embed the turn text + mutate the in-memory graph (CPU-bound, no I/O),
and (b) persist via `MemArkGraphStore.save_node()`/`save_edge()`
(already `to_thread`-wrapped internally, confirmed and tested in the
persistence pass). Recommend wrapping the *whole* `write()` body —
embed, mutate, persist — in a single `asyncio.to_thread` call, rather
than two separate ones. Simpler, and avoids any question of whether the
in-memory mutation and the SQLite write could be observed out of order
across two independent thread-pool dispatches.

## 4. Ordering guarantee

Stated explicitly, per `docs/MEMORY_STORE.md`'s own standard for
Chroma: the merged score from section 2 (max of each half's
min-max-normalized score) determines order — `list[Turn]` is produced by
sorting merged candidates by that score **descending**, then taking the
top `k`. `PromptBuilder.build()`'s "most relevant first" assumption
holds by construction, not by incidental ordering of a dict or a scan.

## 5. The dedup gap

**Checked against `_dedup_retrieved_memory`'s actual matching logic, not
assumed to just work because the interface matches.** Read directly
(`orchestrator.py:51-56`): it drops a `retrieved_memory` turn that's
also in `recent_turns` by **full `Turn` value equality** — `Turn` is
`@dataclass(frozen=True)`, so `role`, `text`, **and `timestamp`** must
all match exactly, not just text.

This is achievable for MemArk, but it's a real requirement this note
needs to state, not something that falls out for free the way it does
for Chroma. Chroma's `write()` stores `turn.timestamp.isoformat()`
directly as metadata — there's no intermediate object with its own,
different auto-generated timestamp in the way. MemArk's `create_node()`
*does* auto-generate its own timestamp with no override parameter (see
section 3) — so without the explicit post-creation `node
.timestamp_created = turn.timestamp` override already specified there,
every retrieved memory `Turn` would carry a *different* timestamp than
its counterpart still sitting in `recent_turns`, and dedup would
silently fail to catch it (the turn would appear to the user as if
recalled from memory, worded identically, redundant with something they
just said in the same exchange). With that override in place —
confirmed this pass to be necessary, not assumed — round-tripping
holds, and `_dedup_retrieved_memory` needs **zero modification**: it
operates purely on `list[Turn]`, with no knowledge of which
`MemoryStore` produced them.

`MemArkGraphStore`'s own persistence layer isn't the gap here — its
tests already prove exact datetime round-tripping through SQLite
(`test_node_round_trips_including_embedding_and_metadata` asserts
`timestamp_created`/`timestamp_updated`/`last_accessed` equality
directly). The gap, now closed by this note's design, was specifically
at the `create_node()` boundary one level up.

## 6. Explicitly deferred, not folded in

`docs/MEMARK_COMPANION_EXTENSIONS.md` sections 2b (decay-rate tuning)
and 2c (a companion-specific node type) stay exactly as staged there —
secondary, additive follow-ups once hybrid retrieval (2a, this note) is
built and evaluated. This note's `write()` only ever creates
`node_type="episodic"` nodes; no new node type, no custom decay rates,
no `LongTermDecayEngine` wiring here.

## 7. Confirming this stays in this project's own adapter code

Everything above is buildable using only public MemArk surface:
`GraphMemoryManager` (`create_node`, `create_edge`, `get_node`,
`.nodes`), `ActivationEngine.propagate_activation()`, `VectorSearch
.add_embedding()`, plus `MemArkGraphStore`/`populate_vector_search()`
(this project's own code, already built). The one thing that could have
looked like reaching into something private — concept-seeding —
**is reimplemented as a small, independent function in our own adapter
code, not imported or called from MemArk's benchmark harness**, per
section 1. Nothing here requires editing any file in
`Arkur745/MemEngine`.

## 8. Addendum: partial-failure handling when persist fails after the in-memory mutation succeeds

Flagged after the note above shipped: `write()`'s design (embed, mutate
the in-memory `GraphMemoryManager`, persist via `MemArkGraphStore`, one
`asyncio.to_thread` call) doesn't say what happens if the persist half
raises *after* the in-memory mutation already succeeded. Concretely: the
node becomes queryable for the rest of this process, but never reached
disk — a quiet version of exactly the failure mode
`docs/MEMARK_PERSISTENCE.md`'s corruption handling exists to prevent
(memory that looks present but silently evaporates on restart).

**Option (a): roll back the in-memory mutation on persist failure**,
checked against `GraphMemoryManager`'s real API rather than assumed
simple. Re-confirmed its complete method list directly (`engine/core
/graph_manager.py`): `create_node`, `create_edge`, `sync_node`,
`sync_edge`, `get_node`, `get_children`, `get_connected_nodes`,
`reinforce_node`, `reinforce_edge`, `decay_edge`, `prune_weak_edges`,
`prune_pendant_nodes`, `get_edges_for_node`, `get_neighbors`,
`get_edge_data`, `has_edge`, `get_edge_between`. **No
`delete_node`/`remove_node`/`delete_edge` exists anywhere on it** — only
threshold-based bulk pruning, nothing for targeted single-node removal
by id. So rollback is not "call one clean undo method" — it would need
to be hand-rolled: `del graph_manager.nodes[node_id]` (reaching directly
into a dict that's technically public but has no supported removal
API), `graph_manager.graph.remove_node(node_id)` (clean — this part is
real, stable `networkx` API, not MemArk's own), and, since this pass's
`write()` design also explicitly calls `vector_search.add_embedding()`
itself (section 1 — `create_node()` doesn't do this for us), `vector_search
.delete(node_id)` (clean — `FaissVectorStore.delete()` is a real public
method, confirmed earlier in this series). Two of three steps are clean;
the third (removing from `GraphMemoryManager.nodes`) is direct,
unsupported dict surgery against internal state MemArk itself provides
no sanctioned way to undo.

**Option (b): let the in-memory mutation stand, propagate the persist
failure to the caller.** `write()` does not swallow whatever
`MemArkGraphStore.save_node()`/`save_edge()` raises
(`MemArkPersistenceError`, already a real, specific exception type from
the persistence pass) — it propagates, exactly as `MemArkGraphStore`'s
own corruption handling already does one layer down. This session's
memory of that one turn is disclosed as session-only (visible in an
exception, not silently accepted) rather than guaranteed to survive a
restart, until a later write succeeds.

**Recommendation: (b).** Two reasons, not just "simpler":

1. Option (a)'s rollback isn't actually clean, per the API audit above —
   it requires hand-rolled mutation of `GraphMemoryManager.nodes`
   directly, a dict MemArk provides no supported way to remove from.
   Rollback code is also notoriously under-exercised in practice (it
   only runs when something else has already failed), which is exactly
   the wrong place to add unsupported-API surgery this project doesn't
   control the stability of. Building "clean" recovery logic on top of
   an operation MemArk itself doesn't support cleanly risks a *second*,
   harder-to-diagnose inconsistency (e.g. a node still reachable via
   `graph.neighbors()` traversal after `del`-ing it from `.nodes`, if
   any other internal structure this audit didn't surface also
   references it) in exchange for protecting against a failure this rare.
2. Option (b) matches this project's own existing convention exactly —
   `MemArkGraphStore` doesn't try to be clever about recovering from a
   corrupt file either; it raises a specific, named error and leaves the
   decision to the caller (`AudioDeviceError` follows the same shape).
   Consistency with an already-established pattern beats inventing a new
   one for a rare case.

**Resolved in a later pass, not by this note**: whether `Orchestrator`
should let a persist failure end the current turn, end the session, or
catch-and-continue with a logged warning was left open here deliberately
(a bounded addendum, not the place to make an orchestrator-layer call).
It's since been decided and implemented — end the turn only:
`Orchestrator._write_turn_safely` catches `MemoryStoreError` (the shared
interface-level type `MemArkPersistenceError` subclasses, so `Orchestrator`
never imports anything MemArk-specific) around every `memory.write()`
call, logs it at ERROR level with scope and turn details via
`logger.exception(...)`, and lets the already-generated/already-yielded
response reach the user completely unaffected — no retry, no ending the
session. See `companion/core/orchestrator/orchestrator.py`'s
`_write_turn_safely` for the full reasoning against the other two
options, and `tests/test_orchestrator_text_turn.py`/
`tests/test_orchestrator_voice_turn.py` for the tests proving it (each
verified to actually fail without the fix, not just assumed correct).

## Recommendation summary

1. **Query sequence** (section 1): embed via Chroma's own standalone
   `DefaultEmbeddingFunction` (verified callable with no Chroma
   client/collection needed, avoiding the torch dependency MemArk's own
   `EmbeddingManager` would pull in) → reimplement concept-seeding as a
   small local function (not reusable from MemArk, confirmed private) →
   `ActivationEngine.propagate_activation()` → sort by activation score.
2. **`query()` needs `asyncio.to_thread`** — confirmed, but for CPU-bound
   computation (embedding inference + graph traversal), not disk I/O,
   since this design never touches SQLite from `query()` at all.
   **`write()` needs it too**, wrapping its combined embed-mutate-persist
   body in one `to_thread` call.
3. The retrieval adapter must own its own live `GraphMemoryManager`,
   separate from `MemArkGraphStore` (which has no in-memory state of its
   own — checked directly, the task's premise didn't hold as stated).
4. Two adapter-side requirements found while tracing this, not
   optional/cosmetic: `write()` must explicitly override
   `node.embedding` (matching MemArk's own example pattern) **and**
   `node.timestamp_created` (this project's own addition, needed
   specifically for dedup) after `create_node()`, since neither is
   settable through its public parameters.
5. Dedup needs zero changes to `_dedup_retrieved_memory` itself, given
   the timestamp-override requirement above is honored.
6. **If persist fails after the in-memory mutation succeeds** (section
   8, added after this note's initial approval): `write()` propagates
   the failure rather than attempting rollback — `GraphMemoryManager`
   has no supported node-removal API, confirmed by auditing its full
   method list, so "clean rollback" isn't actually available to build.
   Matches `MemArkGraphStore`'s own no-silent-failure convention.
   Whether the orchestrator should end the turn, end the session, or
   catch-and-continue on that propagated failure is explicitly left
   open, for a separate pass.

Stopping here per the gate. Want to discuss any of this, or should the
next step be sequencing this against actually wiring `MemArkGraphStore`
+ this adapter into `app.py`?
