# MemoryStore design note (build-order step 8)

**Status: implemented as designed below**, dedup resolution included —
see `companion/memory/store.py` and `docs/ARCHITECTURE.md`'s "MemoryStore"
section. Left as written (rather than rewritten past-tense) as the record
of the design gate this went through before any code was written.

Design-note-first, same gate as `PromptBuilder` (step 5): this interface
was already depended on by `PromptBuilder` (`retrieved_memory` ordering
assumption) and the orchestrator (`recent_turns=[]`/`retrieved_memory=[]`
stubs in `handle_voice_turn`, and all of `handle_text_turn`), so a bad
assumption here was expensive to unwind. Written before any
implementation existed — covers storage choice, embedding model,
ordering, the dedup gap flagged in `docs/PROMPT_BUILDER.md`, and the
write path, each verified against the real library where the claim was
checkable, not assumed.

## 1. Storage choice: Chroma

Picking **Chroma** over LanceDB. Both are genuinely embedded — verified
directly, not assumed:

```python
>>> import chromadb
>>> client = chromadb.PersistentClient(path="./data/memory")  # local dir, no server
```

`PersistentClient` is a pure local-file client (SQLite + a local vector
index under `path`); there is no daemon/server process to run or manage.
Verified on disk with chromadb 1.5.9 by listing the tables of a real
`data/memory/chroma.sqlite3`. That SQLite file holds the collections,
every stored turn's text (`embeddings` + full-text-search tables), its
`role`/`timestamp` metadata (`embedding_metadata`), and the write log
including the vectors themselves (`embeddings_queue`). The HNSW
nearest-neighbour index lives beside it as binary files, one directory
per collection (`data_level0.bin`, `link_lists.bin`, ...). So SQLite is
the system of record, and the HNSW files are the search index over it.
Persistence across real process restarts is tested; see
`tests/test_memory_persistence_across_processes.py`.
LanceDB is equally embedded (Arrow/Lance files on disk, no server either)
— so "requires no server" doesn't decide it either way, both qualify.

The actual deciding factor is the **embedding story** (see §2): Chroma
ships a default embedding function that runs fully locally with no extra
heavy dependency, verified by installing plain `chromadb` and finding
`onnxruntime`/`tokenizers`/`tqdm` already pulled in transitively — no
`sentence-transformers`, no `torch`. LanceDB's embedding integration
(its "embedding registry") isn't zero-config the same way; achieving the
same local-model default there would mean reaching for
`sentence-transformers` (torch dependency) ourselves. Chroma's collection
API (`add`/`query`/`delete`, metadata `where` filters) is also a more
direct fit for turn-level memory with relevance search than LanceDB's
more table/schema-oriented API. `config.example.yaml` already defaults
`memory.backend: chroma`, consistent with this pick.

**LanceDB status going forward**: kept as a named-but-unimplemented
option in config, same pattern as `stt.backend: whisper_cpp` (named,
documented, `NotImplementedError` if selected) — not implemented in this
pass. `companion/memory/store.py`'s existing shape (one class,
`backend:`-selected via a constructor string, unlike the
one-file-per-backend convention in `backends/{stt,tts,llm,vad}/`) is a
pre-existing scaffold decision from before this pass, not something I'm
introducing — I'm keeping it as-is rather than restructuring to match
`backends/`, since `companion/memory/` was already scaffolded distinctly
from `backends/` and restructuring it is out of scope for a MemoryStore
implementation pass.

## 2. Embedding model: Chroma's bundled ONNX MiniLM (local, no new heavy dep)

Verified directly: `chromadb.utils.embedding_functions.DefaultEmbeddingFunction`
delegates to `ONNXMiniLM_L6_V2` — a local ONNX build of
`all-MiniLM-L6-v2` (384-dim), downloaded once (~90MB, from a Chroma-hosted
S3 URL, sha256-checked) to `~/.cache/chroma/onnx_models/` on first use and
run locally via `onnxruntime` (CPU by default) from then on. This is the
same "download a small model once, run it locally forever after" pattern
already established by `FasterWhisperSTT`, `PiperTTS`, and `SileroVAD` —
consistent with local-first, not a new category of exception to it.

Confirmed via `pip install chromadb` alone (no extra install step):
`onnxruntime`, `tokenizers`, and `tqdm` are already transitive
dependencies of `chromadb` itself — **no torch, no
`sentence-transformers`**, unlike VAD (`silero-vad` pulls in `torch`) or
Coqui TTS. `memory-chroma = ["chromadb>=0.5"]` in `pyproject.toml` is
already sufficient; no new extra needed.

**Trade-off to flag**: MiniLM-L6 is a small, fast, English-centric model
— good enough for short conversational-turn similarity search (which is
what this is for) but not state-of-the-art retrieval quality, and not
tuned for other languages. This is the same size/speed-over-quality
trade-off `PromptBuilder`/`OllamaLLM` already made picking small local
models over larger ones — consistent with the project's stated latency
priorities, not a new pattern. Swapping the embedding function later
(e.g. a larger `sentence-transformers` model) is a config-level change
`VectorMemoryStore` should accept as a constructor option, not a
hardcoded choice — same "config decides, code doesn't branch on names"
principle as everywhere else in this codebase.

## 3. Ordering guarantee: native to the query, verified, and testable

Verified directly against a real collection (three short documents, one
query) that Chroma's own `collection.query()` already returns results
**pre-sorted most-similar-first** — ascending by distance:

```
ids order:  ['1', '3', '2']
distances:  [0.576, 1.032, 1.754]
```

(id 1 = "I love hiking in the mountains", id 3 = "...hiking last
weekend...", id 2 = "My favorite food is pizza", queried with "Tell me
about hiking" — the two hiking-related turns rank first, in similarity
order, exactly as `PromptBuilder`'s docstring assumes.)

So `VectorMemoryStore.query()` doesn't need to re-sort anything — it maps
`results["ids"][0]` (already in the right order) straight into a
`list[Turn]` in that same order. This is directly testable, and the task
asks for a real order-assertion test, not just a presence check: the test
plan (§7) writes turns with obviously distinct topics, queries with a
topically-slanted query, and asserts the *returned order* of `Turn.text`
values matches expected relevance — not merely that the right turns
showed up.

## 4. The dedup gap: resolved at the orchestrator, not by changing `MemoryStore`

Recap of the gap (from `docs/PROMPT_BUILDER.md`): nothing stops
`MemoryStore.query()` from returning a turn that's *also* already in the
orchestrator's `recent_turns` window, if recent turns get written to
memory as they happen (which they do — see §5).

**Picking**: orchestrator-side post-query filtering, no change to
`MemoryStore.query()`'s abstract signature. Concretely, after calling
`memory.query(...)`, the orchestrator drops any returned `Turn` that's
already present (by equality) in `recent_turns` before passing
`retrieved_memory` to `PromptBuilder.build()`. `Turn` is a frozen
dataclass (`role`, `text`, `timestamp`) with structural `__eq__`, so this
works *as long as* `write()`/`query()` round-trip `Turn.timestamp`
exactly (ISO-8601 string in Chroma metadata, parsed back with
`datetime.fromisoformat()` — confirmed round-trip-exact in Python 3.11+)
rather than re-stamping a fresh timestamp on read. This is a concrete
implementation requirement of this pick, not an incidental detail.

**Why this one, not the alternatives**:
- *Excluding a recency window from indexing* (turns don't get written to
  memory until they age out of `recent_turns`) needs a second, deferred
  write path — something has to notice a turn "graduating" out of
  `recent_turns` and write it then. That's real added complexity for a
  gap that a one-line filter closes just as well.
- *A query-time exclude parameter* on `MemoryStore.query()` (e.g.
  `exclude: Sequence[Turn] = ()`) is technically viable as a
  backwards-compatible optional addition, and would avoid the minor
  "fewer than k useful results if the top match was a duplicate"
  imprecision post-filtering has. I'm not picking it, to avoid touching
  the abstract contract when a call-site fix works — the same reasoning
  already applied to VAD's `frame_bytes` (resolved via `getattr` at the
  orchestrator, not by extending `VoiceActivityDetector`). **Flagging
  this explicitly since it's a real alternative, not a strawman** — if
  you'd rather have the interface carry this, say so before I implement
  and I'll do that version instead.

`k` stays effectively soft — a caller asking for `k=5` may get fewer than
5 after filtering. Not solved further here; acceptable for v1 given
`recent_turns` windows are small (a handful of turns) relative to typical
`k`.

## 5. Write path: per-turn, immediate, synchronous with the turn loop

Each `Turn` gets written to `MemoryStore` right when it's finalized — one
`await memory.write(...)` call for the user's transcript once an
utterance ends, and one more for the assistant's full reply once
generation finishes (or is cut short by barge-in — whatever text was
actually produced gets written, since that's genuinely what the persona
said in that turn). Not batched: `write()` is a local file operation
(cheap, no network), so there's no latency reason to defer or queue it,
and immediate writes mean memory is never stale relative to what
`recent_turns` already holds.

For voice turns specifically: the assistant side is currently assembled
sentence-by-sentence for TTS (`_sentences_from_llm`) — the write path
needs to accumulate the full response text across those sentences and
write **one** assistant `Turn` per response, not one per sentence
fragment, so memory reads back as coherent turns rather than sentence
shrapnel.

**Interaction with `summarizer.py`**: none, deliberately, in this pass.
`RollingSummarizer.summarize_if_due()` operates on turns *after* they're
already in `MemoryStore` — collapsing old ones into a summary turn once
`trigger_turn_count` is crossed. This pass committed only to a stable
Chroma metadata schema per stored turn (`role`, `text` as the document
body, `timestamp` as ISO-8601 metadata) so a summarizer could read raw
turns back out later.

> **Update (2026-10-19): the summarizer is built**, and the guess above
> that it could work through the existing `write()`/`delete()` surface
> turned out wrong. `delete()` only wipes whole scopes, and there was no
> way to list a scope's turns. `MemoryStore` gained two methods,
> `list_turns` and an all-or-nothing `replace_turns`, implemented for
> Chroma and MemArk. It runs as a background task after each turn. Full
> design: `docs/ROLLING_SUMMARIZER.md`.

## 6. Out of scope for this pass (confirmed real, not blocking)

- **Automatic pruning/expiry policy** (e.g. auto-delete after N days) —
  not implemented; would be a future policy layered on top of `delete()`.
- **User-facing deletion controls** (a UI affordance for "wipe this
  persona's memory" / "wipe everything") — `MemoryStore.delete()` itself
  *is* implemented for real in this pass (it's already part of the
  abstract interface, not deferred), but no UI calls it yet since there's
  no UI. One line each, confirmed as real future work per the original
  brief, not addressed further here.

## 7. Scoping mechanic + test plan (implementation-adjacent, included for review)

`(user_id, persona_id)` scoping uses one Chroma collection per pair,
named `mem-{sha256(user_id)[:16]}-{persona_id}` — hashing `user_id`
because collection names are constrained (verified:
`[a-zA-Z0-9._-]`, 3–512 chars, alnum start/end) and `user_id` isn't
guaranteed to satisfy that (an email address contains `@`, for instance),
while `persona_id` already is constrained by the persona loader.
Per-collection (not one shared collection filtered by metadata) gives a
structural leak guarantee — a `where`-filter bug can't cross-contaminate
personas if there's no shared collection for it to leak within — which
matters more here than for the dedup filter, given schema.md's explicit
"must never leak into a conversation with a different persona"
requirement. `delete(user_id, persona_id)` drops one collection;
`delete(user_id, persona_id=None)` lists and drops every collection
matching that user's hash prefix. Both verified directly against
`list_collections()`/`delete_collection()`.

Test plan: real Chroma in a `tmp_path` dir (skip cleanly, same pattern as
every prior backend, if `chromadb` isn't installed), covering — write
then query returns it; query relevance order (§3's assertion, not just
presence); dedup filtering removes a turn present in `recent_turns`;
scoping (a query for `(user_id, persona_id)` never returns another
persona's or user's turns); `delete()` for both the single-persona and
whole-user cases.
