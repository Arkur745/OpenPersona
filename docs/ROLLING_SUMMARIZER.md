# RollingSummarizer design note

**Status: approved and implemented as designed below** (2026-10-19):
`companion/memory/summarizer.py`, both backends, Orchestrator/Streamlit
wiring, tests in `tests/test_summarizer.py` and the cross-process suite.
Left as written, as the record of the design gate.

`companion/memory/summarizer.py`'s `RollingSummarizer.summarize_if_due()`
is the last unimplemented piece CLAUDE.md lists. Its job, per its own
docstring: once a (user, persona) history has grown past
`trigger_turn_count`, replace the oldest turns with one summary turn, as
a background job, never inline in a chat turn. `docs/MEMORY_STORE.md`
section 5 left two things open: when it runs, and how it reads and
rewrites turns through the `MemoryStore` surface.

## 1. The gap in `MemoryStore`

Today the interface has `query()` (top-k by relevance), `write()` (append
one turn) and `delete()` (wipe a whole scope or user). The summarizer
needs two things none of these give it:

- **The oldest turns, in order.** `query()` ranks by relevance, not age,
  and is capped at `k`. There's no way to enumerate a scope.
- **Remove specific turns.** `delete()` only wipes whole scopes. `Turn`
  has no id, so there is nothing to name a single stored turn by.

`docs/MEMORY_STORE.md` section 5 assumed a future summarizer could "rewrite
a collapsed summary turn through the same `write()`/`delete()` surface".
That was wrong: `delete()` can't remove individual turns.

## 2. Proposed interface addition: two abstract methods

```python
async def list_turns(self, user_id: str, persona_id: str) -> list[Turn]:
    """Every stored turn for this user+persona, oldest first
    (by Turn.timestamp). Empty if nothing is stored."""

async def replace_turns(
    self, user_id: str, persona_id: str, old: Sequence[Turn], new: Turn
) -> bool:
    """Store `new` and remove every stored turn equal to one in `old`, as
    one operation. If any turn in `old` is no longer stored, change
    nothing and return False. Returns True if the replacement happened."""
```

Design choices, each with the alternative it beat:

- **Identity is `Turn` equality** (role, text, timestamp), not a new id
  field. Both backends already round-trip `timestamp` exactly, because the
  orchestrator's dedup filter depends on it (`docs/MEMORY_STORE.md`
  section 4, `docs/MEMARK_RETRIEVAL_ADAPTER.md` section 5). Adding an id to
  `Turn` would touch `types.py`, `PromptBuilder`, the orchestrator and every
  test that builds a `Turn`, for no gain the summarizer needs.
- **`replace_turns` is all-or-nothing on `old`.** This is what stops a
  memory wipe from being undone. If the user runs `delete()` while a
  summary is being generated (seconds of LLM time), the old turns are gone
  by the time `replace_turns` runs. Without the check, the summary would
  write the deleted content back. With it, nothing is written. Section
  6.3 covers the remaining window in Chroma.
- **No separate `count()`.** The summarizer calls `list_turns()` and
  counts. Both backends are local; listing a few hundred turns of text is
  milliseconds (Chroma `get()` reads SQLite; MemArk already holds every
  node in memory). If histories ever reach a size where that matters, a
  `count()` can be added then.
- **Abstract, not optional.** Recommended: both methods are
  `@abstractmethod`, same as `delete()`. Every real backend implements
  them, and a store that silently can't be summarized would let history
  grow without anyone noticing. Cost: the six test fakes that subclass
  `MemoryStore` (`test_app.py`, `test_agent_langgraph.py`,
  `test_orchestrator_text_turn.py`, `test_orchestrator_voice_turn.py`,
  `test_streamlit_app.py`, `test_tools.py`) each get two small stub
  methods. **The alternative is real**: concrete methods on the base
  class that raise `NotImplementedError`, so summarization is an optional
  capability and no fake changes. Say so if you'd prefer that.

## 3. What a summary turn is

- **`role="system"`.** Nothing writes `system` turns to memory today
  (checked: the orchestrator writes only `user`/`assistant`; the agent
  path writes only the final answer, as `assistant`). So `role ==
  "system"` in a `MemoryStore` means "summary". The summarizer counts and
  batches only `user`/`assistant` turns. `PromptBuilder` already renders
  retrieved memory as `- [date] (role): text`, so a retrieved summary
  reads as `(system): Summary of earlier conversation ...` with no
  `PromptBuilder` change. No new `Role` value or `Turn` field is needed.
- **Timestamp = the newest summarized turn's timestamp.** The summary
  then sorts after everything it replaced and before every turn it didn't
  touch, so `list_turns()` stays chronological.
- **Summaries are not re-summarized.** Each summary covers one batch.
  Feeding old summaries back in (the classic "rolling summary of the
  summary") compounds loss with every pass, and one ever-growing summary
  embeds badly for vector search: one vector for months of topics. Many
  small, era-specific summaries retrieve better. Cost: summaries
  accumulate at one per batch (one per 25 turns at the default), which
  is still a 25× reduction in stored turns. Retrieval is already capped at
  `k`, so prompt size stays bounded either way.

## 4. `summarize_if_due` semantics

```
raw = [t for t in await memory.list_turns(u, p) if t.role != "system"]
if len(raw) < trigger_turn_count: return False
batch = raw[: trigger_turn_count // 2]          # oldest half
summary_text = <LLM over batch>
if not summary_text.strip(): return False       # never replace with nothing
return await memory.replace_turns(u, p, batch, Turn("system", summary_text, batch[-1].timestamp))
```

With the shipped default (`trigger_turn_count: 50`, already in
`config.example.yaml`), each run turns the oldest 25 raw turns into one
summary, and at most about 50 turns per scope are ever kept word for
word. **Trade-off to flag:** a detail from 200 turns ago survives only if
the summary kept it. That's what the existing docstring asks for
("replace the oldest turns"), but it's lossy by design. Anyone who wants
more verbatim history raises `trigger_turn_count`; it's one config value.

**The LLM call** goes through the `LLMBackend` interface only:
`generate_stream(prompt, context)` with `prompt` = a fixed summarization
instruction (third person, keep names, preferences, facts, plans,
commitments, and anything the user asked to be remembered; drop small
talk) and `context` = one `user` turn holding the batch as a dated
transcript. This prompt lives in `summarizer.py`, not `PromptBuilder`.
CLAUDE.md's "PromptBuilder is the only place prompts get assembled" is
about persona prompts (persona + memory + recent window). This prompt has
none of those, and putting it in `PromptBuilder` would mix two unrelated
jobs.

**Which LLM instance:** a separate one, built by `app.build_llm` from the
base `llm.options`, never with `voice_mode_override` applied. Voice-mode
overrides exist to keep spoken replies short (e.g. a lower `num_predict`),
which would cut a summary off partway.

## 5. Trigger: after a turn, off the request path

**CLI text mode and voice mode** each run one event loop for the whole
session. `Orchestrator` gets an optional `summarizer: RollingSummarizer |
None = None` (same opt-in pattern as `agent`). After a turn's memory
writes, it calls `asyncio.create_task(summarizer.summarize_if_due(...))`
and keeps a reference in a set so the task isn't garbage-collected. At
most one task runs per scope at a time; while one is running, later
triggers for that scope are skipped. Exceptions are logged at ERROR and
never reach the conversation (same policy as `_write_turn_safely`). The
turn never awaits the task. `summarize_if_due` is cheap when nothing is
due (one `list_turns`), so checking after every turn is fine.

When the session ends, `asyncio.run` cancels a summary that's still
running. That's safe because each backend's `replace_turns` does all its
work inside one `asyncio.to_thread` call. Cancelling an `await` can't
stop a thread function partway, so a summary is either fully applied or
never started. It will be retried on the next session's first turn.

**Streamlit** runs each turn under its own `asyncio.run(...)`, so a task
created during the turn would be cancelled as soon as the turn returns.
The Streamlit app therefore builds its `Orchestrator` with
`summarizer=None`. After the reply is rendered (and spoken, if enabled),
it calls `asyncio.run(summarizer.summarize_if_due(...))` itself. The
reply is never delayed. **Trade-off to flag:** on the one turn in 25 where
a summary is due, Streamlit's script keeps running for a few seconds
after the reply appears, so a message sent during that window waits for
it. Avoiding that needs a long-lived background event loop thread for
Streamlit, which brings real cross-thread access to both stores. I think
that's not worth it for a once-in-25-turns delay. Say if you disagree.

**Not addressed, flagged:** Ollama queues requests beyond its
`OLLAMA_NUM_PARALLEL` setting. If the server is configured to run one
request at a time, a summary that's still generating can delay the next
chat reply (once per 25 turns, only if the user replies within the
summary's few seconds). Measuring that belongs to the latency work in
`HARDWARE_REQUIREMENTS.md`, not here.

**Config:** `memory.summarizer.trigger_turn_count` (already in
`config.example.yaml`) gets parsed into `AppConfig`. If the section is
missing or the value is `0`, there's no summarizer. If it's present and
positive, `app.build_summarizer(config, memory)` builds one. A
non-integer or negative value is a `ConfigError`, same as every other
section.

## 6. Per-backend implementation

### 6.1 Chroma (`VectorMemoryStore`)

- `list_turns`: `collection.get(include=["documents", "metadatas"])`,
  mapped to `Turn` the same way `query()` does, sorted by timestamp.
  `get()` returns ids in insertion order, which isn't a timestamp
  guarantee, so the sort is explicit.
- `replace_turns`: one `to_thread` function. `get()`, then match `old`
  against (role, text, timestamp) to find ids. If any are missing, return
  False. Otherwise `add(new)`, then `delete(ids=...)`. Chroma has no
  multi-operation transaction, so the order is chosen for safety: if the
  process dies between `add` and `delete`, you get the summary plus the
  old turns (redundant, and that batch is summarized a second time
  later), never the old turns gone with no summary.

### 6.2 MemArk (`MemArkMemoryStore` + `MemArkGraphStore`)

- `MemArkGraphStore` gains `replace_nodes_sync(user_id, persona_id,
  delete_ids, new_node)`: one SQLite transaction (`INSERT` the summary
  node, `DELETE` the old node rows plus any edge touching them). That's
  truly atomic: a crash leaves all of the change or none of it.
- `MemArkMemoryStore.replace_turns`: match `old` against the scope's
  in-memory episodic nodes, call the above, then **drop the in-memory
  `_Scope`** so it re-hydrates from SQLite on next use. That's the same
  approach `delete()` already uses, for the same reason:
  `GraphMemoryManager` has no supported single-node removal
  (`docs/MEMARK_RETRIEVAL_ADAPTER.md` section 8), so rebuilding beats
  editing it in place.
- Summary nodes are `node_type="episodic"` with `metadata={"role":
  "system"}`, not MemArk's own `"summary"` type. `_fallback_scores` scans
  only episodic nodes, and graph seeding uses only concept nodes, so a
  `"summary"` node would never be retrieved.
- `list_turns`: the scope's episodic nodes mapped like `query()` does,
  sorted by `timestamp_created`.
- **Concurrency fix this needs:** `MemArkGraphStore`'s connection comment
  says it's safe "ONLY because ... never more than one write in flight at
  a time for a given scope". A background summarizer breaks that: a
  turn's `write()` and the summarizer's `replace_turns()` can now both run
  in worker threads at once. `MemArkMemoryStore` gets a `threading.Lock`
  held around each sync body (write's embed+mutate+persist, replace,
  delete). To do that, `write()` moves to a single `to_thread` call, using
  a new public `MemArkGraphStore.save_node_sync`. That reverses an earlier
  note's choice not to call the private `_save_node_sync`, and only
  because this pass is changing that class anyway. The partial-failure
  behaviour from that note's section 8 doesn't change: the in-memory
  mutation still happens first and persist errors still propagate. A
  thread lock rather than an `asyncio.Lock`, because Streamlit runs
  successive calls under different event loops.

### 6.3 The remaining wipe window

The all-or-nothing check in `replace_turns` closes the main
wipe-then-resurrect race. For MemArk the store lock covers what's left.
For Chroma, a `delete()` landing between `replace_turns`' `get()` and
`add()` (microseconds, separate threads) would make `add()` fail against
a dropped collection. That error gets logged and nothing is written.
It's not silent resurrection either way.

## 7. Tests

- `tests/test_summarizer.py`: a fake `LLMBackend` that returns a fixed
  summary and records its input, run against **real** stores in
  `tmp_path` (Chroma with its real default embedding, as
  `tests/test_memory_store.py` uses; MemArk following
  `tests/test_memark_retrieval.py`), parametrized over both backends.
  Cases: not due → False and nothing changed; due → oldest
  `trigger // 2` raw turns gone, one `system` turn with the newest batch
  timestamp present, newer turns untouched, `list_turns` still
  chronological; summaries are excluded from the count and the batch;
  empty LLM output → nothing replaced; `delete()` during the LLM call
  (fake LLM awaits an event) → returns False and the wipe holds;
  `replace_turns` contract tests on their own; summary retrievable via
  `query()`.
- Orchestrator: the trigger fires after a text turn and the turn doesn't
  wait for it (the fake summarizer blocks on an event and the turn
  returns anyway); failures are logged, not raised; one in-flight task
  per scope.
- Config: parse, disable, and reject cases in `tests/test_config_loader.py`
  style.
- Cross-process, in `tests/test_memory_persistence_across_processes.py`
  style, both backends: process A writes turns past the trigger, process B
  runs `summarize_if_due` (fake LLM), process C sees the summary and the
  shrunken raw set via `list_turns()` and `query()`.
- `scripts/check_offline.py`: no new backend, so nothing to add. The
  summarizer uses the LLM that script already exercises.

## 8. Out of scope

- Re-summarizing summaries, or a size cap on how many summaries a scope
  can hold (section 3).
- A schedule-based (timer) trigger. After-turn is enough for a
  single-user local app.
- Measuring summary quality or latency against a real model. The fake-LLM
  tests prove the mechanism, not how good the summaries are. Docs will
  say so.
