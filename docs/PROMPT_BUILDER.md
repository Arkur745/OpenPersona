# PromptBuilder design note (seam with MemoryStore)

**Status: implemented**, per the decisions below — Option A confirmed,
memory renders into the prompt string, turn-count + character-cap
bounding. `companion/core/prompt_builder/prompt_builder.py` implements
this; `MemoryStore.query()` is unchanged, as predicted — real
`MemoryStore` (build-order step 8, see docs/MEMORY_STORE.md) has since
landed and validated this seam without needing any change to
`PromptBuilder` itself, confirming the assumption this note made in
advance of that. (The interface later gained `list_turns`/
`replace_turns` for rolling summarization, which `PromptBuilder` doesn't
use. A summary turn has `role="system"` and renders in the memory
section as `(system)`, like any other retrieved turn.)

Two refinements were found while actually implementing against this note
— both are noted inline below where they change the "concretely" shape,
and both are explained in `prompt_builder.py`'s own module docstring:

1. `build()` returns a `BuiltPrompt(prompt, context)`, not a bare `str` —
   see "Bounding the recent conversation window" below.
2. The new `user_message` is appended to `context` as the final turn
   (role="user"), not folded into the prompt string — see "A constraint
   from Part A" below.

## What already exists, and constrains this

Two decisions from the earlier scaffolding pass are already load-bearing
here — this note ratifies or revises them, not proposes from a blank page:

1. **`MemoryStore.query()`'s return shape is already fixed**, in
   `companion/core/interfaces/memory_store.py`:
   ```python
   async def query(self, user_id: str, persona_id: str, query: str, k: int = 5) -> list[Turn]
   ```
   Raw `Turn`s (`role`, `text`, `timestamp`), ordered most-relevant-first,
   not pre-summarized text and not scored/ranked snippet objects. Changing
   this is possible but would mean re-opening an interface that's already
   published and (per the previous pass) implicitly relied on.
2. **`PromptBuilder.build()`'s stub signature already takes pre-fetched
   memory as a parameter**, not a `MemoryStore` reference:
   ```python
   def build(self, persona: Persona, recent_turns: list[Turn], retrieved_memory: list[Turn], user_message: str) -> str
   ```
   And `docs/ARCHITECTURE.md`'s data-flow diagram already draws
   `MemoryStore.query()` as a separate step *before* `PromptBuilder.build()`,
   feeding it `retrieved_memory: list[Turn]` as an input.

## The actual open question

Your instructions said to have PromptBuilder "call MemoryStore through
[the] seam" — which reads as PromptBuilder holding a `MemoryStore`
reference and querying it itself. That's a real alternative, but it's a
**change** from what's already committed above, not a gap-fill. Laying out
both so the choice is explicit:

**Option A — caller fetches, PromptBuilder stays pure** (what's already
committed in the stub + diagram):
- `Orchestrator` (once `handle_text_turn` is implemented) calls
  `MemoryStore.query(user_id, persona_id, query_text, k=...)` itself, then
  passes the result into `PromptBuilder.build(..., retrieved_memory=...)`.
- `PromptBuilder` never imports or holds a `MemoryStore`. It's a pure
  function of its four arguments — deterministic, trivial to test, no
  mocking required for memory at all (just pass a `list[Turn]` literal).
- Downside: `PromptBuilder` doesn't decide *what query text* to search
  memory with — that decision (probably just `user_message`, but maybe
  something smarter later) lives at the call site instead.

**Option B — PromptBuilder fetches memory itself:**
- `PromptBuilder.__init__` takes a `MemoryStore`; `build()` takes
  `user_id`/`persona_id` instead of `retrieved_memory`, and calls
  `self.memory.query(...)` internally.
- Every `PromptBuilder` test now needs a fake `MemoryStore` injected
  (real mocking, not just data) — including the "clearly marked as
  temporary" stub this task asks for as a class, not a literal.
- Bundles two responsibilities (retrieval + assembly) into one object,
  contradicting `PromptBuilder`'s own current docstring ("the one place
  persona + memory + context get assembled" — assembled, not fetched).

**Recommendation: Option A.** It's not just "less work" — it's what the
data-flow diagram and stub signature already describe, so keeping it
avoids silently diverging from published architecture docs for no
functional gain. It also keeps `PromptBuilder` testable with plain data
instead of a mock object graph. If you want Option B instead, say so and
I'll revise the stub signature and this note before implementing — this
is exactly the kind of thing better caught now.

The rest of this note assumes **Option A**. Under Option A, "call
MemoryStore through a stub" (per your instructions) becomes: `build()`
tests pass a hand-constructed `list[Turn]` as `retrieved_memory`, standing
in for what a real `MemoryStore.query()` call would have returned — no
`MemoryStore` mock object needed since `PromptBuilder` never touches that
interface directly. If a temporary fake is still wanted for the
*orchestrator*-level path once `handle_text_turn` is attempted, that's a
separate, smaller decision for build-order step 9, not this file.

## A constraint from Part A that affects this: memory turns are not chat turns

`OllamaLLM.generate_stream(prompt, context)` (just implemented) maps each
`Turn` in `context` directly to an Ollama chat message —
`{"role": turn.role, "content": turn.text}`. `Turn.role` is typed as
`Literal["user", "assistant", "system"]`; there's no `"memory"` role, and
there shouldn't be one added just for this, because Ollama's chat API
(and every mainstream chat-completion API) doesn't accept arbitrary
role strings — sending one invalid role breaks the request.

That means **retrieved memory must not be merged into `context` as
additional chat-role turns.** `retrieved_memory` is rendered as text and
folded into the returned `prompt` *string* (grounded alongside persona
tone/boundaries/example_dialogue). This also matches the legacy
prototype's original prompt design (informally, before this rewrite) of
narrating past context as prose ("Here is the User's past conversation
and long-term emotional context: ...") rather than as replayed messages,
which reads more naturally for a companion persona recalling something
than a memory being misrepresented as literal dialogue.

**As implemented** (revised from this note's original sketch — see the
two refinements at the top of this file):
```
prompt  = <persona name/tone/boundaries> + <example_dialogue, if any> + <rendered retrieved_memory, if any>
context = <bounded recent_turns> + [Turn(role="user", text=user_message)]
```
`user_message` ended up in `context`, not `prompt`: standard
chat-completion shape (verified directly against Ollama in the LLM pass)
is `[system: grounding, ...history, user: the new message]` — putting the
live question inside the system-role string is non-standard and risks the
model reading it as an instruction rather than something to respond to.
This also gives `recent_turns` a real reason to be a `build()` parameter
at all: bounding it to `max_recent_turns` needed to happen *somewhere*,
and nothing else in this codebase touches `recent_turns`, so `build()`
returning the bounded result (as part of `BuiltPrompt`, not a bare `str`)
is what actually enforces that constructor parameter instead of leaving
it dead.

## Bounding the recent conversation window

The stub already has `max_recent_turns: int = 10` and
`max_memory_turns: int = 5` (the latter conveniently matches
`MemoryStore.query`'s own `k: int = 5` default). Turn-*count* is the
proposed bound for v1, not token-count, for a concrete reason: token
budgeting requires a tokenizer matched to whatever model Ollama is
actually running, which is itself config-driven and swappable — coupling
`PromptBuilder` to one model's tokenizer to count tokens accurately would
cut against the backend-agnostic principle in `docs/ARCHITECTURE.md`.

That said, turn-count alone doesn't bound the *content* — a companion app
plausibly gets long, narrative user messages, and 10 turns of those could
still blow past a small model's context window. Proposed pragmatic
middle ground for v1: also cap each individual turn's rendered length at a
fixed character count (e.g. 2000 chars — cheap, no tokenizer, no per-model
config) when folding it into the prompt, truncating with a marker rather
than silently cutting mid-thought. Real token-aware budgeting is a
reasonable follow-up once a specific model's tokenizer is available to
count against, not a v1 requirement.

## Ordering, and a known gap to flag rather than solve now

- `recent_turns`: chronological (oldest → newest), per how conversation
  history is naturally read.
- `retrieved_memory`: relevance-ranked (most-relevant-first), per
  `MemoryStore.query`'s existing docstring — **not** chronological.
  `build()` needs to render these differently (e.g. recent history as a
  natural back-and-forth, memory as a bullet-ish "relevant past context"
  block) rather than concatenating both the same way.
- **Resolved (build-order step 8):** `MemoryStore.query()` can return a
  turn that's *also* already in `recent_turns`, since recent turns get
  written to memory as they happen. Resolved at the orchestrator, not
  here — `Orchestrator._dedup_retrieved_memory` filters `retrieved_memory`
  against `recent_turns` by value equality before either reaches
  `PromptBuilder.build()`, so `PromptBuilder` itself stays untouched and
  still never needs to know this gap existed. See
  docs/MEMORY_STORE.md section 4 for why that resolution was picked over
  the alternatives (excluding a recency window from indexing, or adding
  an exclude parameter to `MemoryStore.query()`).

## Later addition: tool-use rules (2026-10-20)

For a persona with `tools:`, `build()` appends a fixed "How to use your
tools" section as the **last** section of the prompt, after the memories.
It says:
- act when asked;
- only claim what this turn's tool results confirm;
- handle `Error:` results;
- don't announce tools;
- plain text, no LaTeX.

The memory section's heading now also says remembered actions happened
in the past. Personas without tools get neither change. The order was
measured: rules after memories stopped the model copying remembered "Got
it, reminder set" replies. See `docs/ARCHITECTURE.md`, "Agentic personas",
for the numbers. This keeps PromptBuilder the only place prompt text is
assembled.

## Decisions (confirmed)

1. Option A — `PromptBuilder` stays pure, receives `retrieved_memory` as
   a parameter, never touches `MemoryStore`.
2. Retrieved memory renders into the `prompt` string, not into `context`.
3. Turn-count + per-turn character cap for v1 bounding, token-aware
   budgeting deferred.

Everything else in this note (ordering, the dedup gap) is now resolved by
`MemoryStore`/`Orchestrator` (build-order step 8, see
docs/MEMORY_STORE.md) — `PromptBuilder` itself never had to change to get
there.
