# MemArk backend design note (proposal — now implemented and wired)

**Status: implemented.** Every recommendation in this note's final
section ("What I'd propose doing next, if this is approved") has since
landed: `docs/MEMARK_PERSISTENCE.md` (local SQLite persistence),
`docs/MEMARK_RETRIEVAL_ADAPTER.md` (the `MemoryStore` adapter, Option A
per section 2 below), and finally `companion/app.py`'s `build_memory`
wiring `memory.backend: memark` as a real, config-selected,
startup-time-only second backend (per section 3 below — no per-session
picker) — see `docs/ARCHITECTURE.md`'s "MemoryStore" section for the
as-shipped summary. The analysis below is left as originally written,
except section 4's latency paragraph, since updated with real measured
numbers in place of the original "no evidence-based claim" placeholder
(section 5's recall/decay tradeoffs still apply to the shipped backend
exactly as originally written).

Same gate `docs/MEMORY_STORE.md` and `docs/PROMPT_BUILDER.md` got: this is
a design note only. No adapter code, no `MemoryStore` subclass, nothing
written against MemArk's actual modules yet. Everything below is grounded
in an actual clone-and-read of `github.com/Arkur745/MemEngine` (cloned
locally, inspected, deleted — nothing was guessed from the README alone),
cross-checked against this project's own `MemoryStore`/`PromptBuilder`/
`Orchestrator` as they exist today, not as remembered from an earlier
design pass.

## 1. What MemEngine actually is, structurally

It's a research repo with unusually honest packaging metadata, not a
stable library with a public contract. Concretely:

- The GitHub repo is `Arkur745/MemEngine`; the project renamed itself
  "MemArk" and the installable package is `memark` (its own README says
  so explicitly: "formerly marketed as MemEngine"). `pyproject.toml`
  declares `name = "memark"`, `version = "1.0.0"`, MIT license, and maps
  `memark/` itself as the package root (`package-dir = {"" = "memark"}`).
  It genuinely is `pip install`-able.
- But `memark/__init__.py` is two lines (`__version__ = "1.0.0"`) — no
  re-exports, no public surface at the package's own declared import
  root. The project's own README Quick Start doesn't `import memark`; it
  does `sys.path.insert(0, os.path.abspath("memark"))` and then imports
  `engine.core.graph_manager.GraphMemoryManager` etc. directly — bare
  `engine.*`/`agent.*`/`llm.*` roots, not `memark.*`. Every example script
  under `examples/` repeats the same `sys.path` hack. This is the
  project's own documented, intended usage pattern, not a workaround I'm
  inventing — it just isn't normal Python package consumption.
- There's no single call that means "query memory." The closest thing is
  `engine/core/retrieval_pipeline.py`'s `RetrievalPipeline`, which itself
  needs 8 pre-constructed collaborators passed into its constructor
  (`GraphMemoryManager`, `EmbeddingManager`, `VectorSearch`,
  `ScoringEngine`, `ReinforcementManager`, `ActivationEngine`,
  `HierarchyTraversal`, `ContextCompressor`) — there's no factory or
  default constructor that wires these for you. The two official example
  scripts (`examples/minimal_example.py`, `examples/retrieval_demo.py`)
  each hand-wire a different subset directly and never use
  `RetrievalPipeline` at all.
- The internal API isn't even self-consistent: `GraphMemoryManager
  .create_node(node_type, summary, metadata=...)` accepts `metadata`;
  the wrapper `MemoryRepository.create_node(node_type, summary)` (meant
  to be the higher-level entry point) doesn't — it silently drops the
  parameter's availability rather than forwarding it.
- I found a real correctness bug while reading, not a style nit:
  `RetrievalPipeline.retrieve_semantic_nodes` (`engine/core
  /retrieval_pipeline.py`, the `return ranked_results[:top_k]` line) has
  its `return` indented one level too deep — inside the `for result in
  results:` loop, after `.sort()`. It returns after processing exactly
  one candidate, every time, regardless of `top_k`. This is the method
  `RetrievalPipeline.retrieve()` (its own top-level "do a retrieval" call)
  depends on for its first stage. I'm not asserting anything about the
  benchmark numbers being wrong because of this — the benchmark harness
  may call a different path — but it's concrete evidence this is
  pre-production research code, not hardened against the kind of use an
  adapter would put it through.
- `memark/agent/` and `memark/llm/` are a different scope entirely:
  `CognitiveAgent` (`memark/agent/cognitive_agent.py`) takes the LLM
  itself as a constructor argument, plus 8 more collaborators, plus
  optional metacognition/intent/curiosity engines — it's designed to
  *own the whole conversation loop*, not to be called into by a host
  orchestrator. Using MemArk "as designed" would mean handing it the
  entire turn loop this project's `Orchestrator`/`PromptBuilder`/
  `LLMBackend` already own — not an option here. The plan below
  deliberately uses only `engine/core`, `engine/retrieval`,
  `engine/consolidation`, `engine/hierarchy`, `engine/memory` and ignores
  `agent/`/`llm/` entirely. Worth stating outright: that's using a slice
  of what this dependency is designed to be, not the whole thing.

**Conclusion**: "add it as a backend" means writing an adapter against a
research codebase's internal modules directly, with no versioned public
API and no stated stability/deprecation policy. If this goes ahead, pin
an exact commit hash, not a version range — there's nothing here to
declare compatibility against.

## 2. What MemArk's retrieval actually returns — and which shape to adapt to

Ground truth first, because it changes the framing of the two options as
originally posed: MemArk's own highest-level retrieval call,
`RetrievalPipeline.retrieve(query, top_k) -> str`, already flattens
everything into a plain multi-line string before you'd ever see it —
semantic search → graph-neighbor expansion → hierarchy expansion →
reinforcement → context compression → `assemble_context()`, which joins
`f"[{node.node_type.upper()}] {node.summary}"` lines with `\n`. There is
no subgraph, edge list, or multi-hop path in that return value — just an
ordered list of node summaries as one string. Even reaching one level
deeper, `ActivationEngine.propagate_activation()` returns
`{"nodes": {node_id: activation_score}}` — node IDs and floats, still no
edges. The "rich graph shape" the original framing pictured isn't
actually what MemArk's own retrieval surface hands back today; you'd have
to write new code *against* MemArk's internals (walking `GraphMemoryManager
.get_edges_for_node()` yourself) to produce something structurally richer
than a ranked list. That's a real option, but it's "build a retrieval
mode MemArk doesn't currently expose," not "plumb through what's already
there."

With that corrected, the two options:

**a. Force-fit into `list[Turn]`.** Adapter calls down to the same
node-ranking stages `RetrievalPipeline.retrieve()` already chains
(semantic → graph expansion → hierarchy expansion → compression), and
instead of joining the surviving nodes into one string, wraps each into
one synthetic `Turn(role=..., text=node.summary, timestamp=node
.timestamp_created)`. `MemoryStore`, `PromptBuilder`, and `Orchestrator`
need zero changes — the adapter alone absorbs the shape difference,
exactly like `VectorMemoryStore` does for Chroma today.
  - Real cost, but smaller than it first looks: `PromptBuilder
    ._memory_section` already renders every `Turn` as an independent
    bullet line (`- [date] (role): text`) with no way to express "these
    two are linked" or "this is two hops out" even for the *existing*
    Chroma backend — so Option A isn't discarding structure the current
    renderer could show anyway.
  - One real wrinkle worth naming, not glossing over: `Turn.role` is
    `Literal["user", "assistant", "system"]`. A MemArk node (a summary or
    concept, possibly synthesized from many turns) isn't cleanly either
    "user" or "assistant" — it's a consolidated memory, not a verbatim
    thing someone said. This is a real, if minor, adapter judgment call
    (I'd lean `"user"`, since these represent facts *about* the user's
    history rather than the persona's own prior speech, but it's a
    genuine approximation either way, worth a comment in the adapter
    when written).

**b. Extend `MemoryStore`'s contract** (a richer return type, or a second
graph-shaped query method) and teach `PromptBuilder` to render it.
  - Touches an interface with three real dependents today (`Orchestrator`,
    `PromptBuilder`, the Chroma backend) for the benefit of a backend
    that doesn't yet exist. `PromptBuilder` would need to branch on which
    shape it received — the same "branch on backend outside the backend
    layer" smell `CLAUDE.md` already calls out as the rule most likely to
    get violated by a well-intentioned shortcut, just one layer up (shape
    instead of name).
  - And, per the ground truth above: the payoff this option is buying —
    "preserve the graph's real structure in the prompt" — isn't actually
    sitting in MemArk waiting to be plumbed through. It would have to be
    *built*, on top of MemArk's lower-level graph/edge accessors, as new
    code this project would own and maintain going forward.

**Recommendation: Option A**, for the same reason `PromptBuilder`
picked Option A over B in its own design note: keep the shared interface
serving its existing, real dependents unchanged, and let the one new
backend absorb the impedance mismatch at its own boundary. Three specific
reasons, not just "consistency for its own sake":

1. Option B's presumed benefit isn't actually available from MemArk's
   current retrieval surface (see above) — building it would mean this
   project writing and maintaining MemArk's missing rendering layer, which
   is out of scope for "add a backend."
2. `MemoryStore` has two other real consumers today, both fully served by
   the current contract — changing it for one backend's hypothetical
   benefit inverts "the abstraction serves the many, not the one," the
   same principle every prior backend pass in this project has followed.
3. If MemArk's graph richness is worth showcasing later, that's better
   served by an additive, separate surface (e.g. a debug/inspector view
   reading MemArk's graph directly) than by bending the interface every
   `MemoryStore` backend must implement.

## 3. Per-session backend switching — how big a change, honestly

I grepped this codebase for `conversation_id`, `session_id`, and
`ConversationId` before writing this section: zero matches, anywhere.
There is no conversation/session-level concept in this project today —
none, not a partial one. The only addressing scheme that exists is
`(user_id, persona_id)`, and both are selected exactly once, at process
startup:

- `companion/__main__.py` reads `--persona`/`--user-id` once as CLI args.
- `companion/app.py`'s `build_orchestrator()` constructs one
  `Orchestrator` with one `MemoryStore` instance baked into its
  constructor (`Orchestrator.__init__`'s `memory: MemoryStore` parameter)
  — held as `self.memory` for that `Orchestrator`'s entire lifetime.
- `handle_voice_turn` is a single long-running state machine over that
  one `Orchestrator`/one `MemoryStore` for the whole process; `python -m
  companion`'s "session" is really "the process," not a stored,
  addressable entity that could carry a per-conversation decision.
- Text mode's notion of a "chat" is thinner still: `handle_text_turn` is
  stateless request/response — the caller owns and passes `recent_turns`
  fresh every call. There isn't even a runtime object today that *is* "a
  conversation" for a picker to attach a backend choice to.

So "let a user pick a memory backend per chat at creation time" isn't a
parameter addition — it depends on things that don't exist yet:

1. A first-class `Conversation`/session entity (id, `user_id`,
   `persona_id`, created-at, and now a `memory_backend` field) — currently
   not even a bare dataclass anywhere in this codebase.
2. Something that persists *that mapping* across process restarts.
   Today, only memory *content* is persisted (via `MemoryStore`, scoped
   by `user_id`+`persona_id`) — nothing about "which conversation is
   this" is stored anywhere; there's nothing to persist it in.
3. `Orchestrator` would need to stop owning one `self.memory` for its
   whole lifetime — either accepting a `MemoryStore` per call
   (`handle_text_turn(..., memory=...)` / `handle_voice_turn(...,
   memory=...)`) or being constructed per-conversation by something that
   currently doesn't exist (a session/conversation factory). Either way
   is a real signature change to code three things already depend on:
   `app.py`'s `build_orchestrator`, `__main__.py`, and every orchestrator
   test.
4. `app.py`'s `build_memory(config)` currently builds *one* backend,
   named once by `config.yaml`. Per-conversation picking needs it to
   become capable of constructing *either* backend on demand, by name, at
   conversation-creation time — a different function shape, not a bigger
   config file.

Said plainly, per your ask: **this is an application-model change, not a
config-flag-sized one, and it has nothing to do with MemArk specifically**
— introducing a first-class `Conversation` concept is a prerequisite any
per-session picker would need regardless of which second backend exists.
It deserves its own gated design note when it's actually being scoped,
not a paragraph inside this one.

**Scoping recommendation**: land MemArk the same way Chroma already
works — a second `memory.backend:` value in `config.yaml`, selected once
at startup, exactly like every other backend swap this project has done.
Treat "per-session, user-facing picker" as a separate, later initiative
that starts from "does this app need a `Conversation` concept" as its own
question — not something to fold into landing MemArk itself.

## 4. Compute/dependency cost

MemArk's `pyproject.toml`/`requirements.txt` (identical list in both)
name every one of these as an unconditional runtime dependency — no
extras/optional-dependency split on MemArk's side at all:

```
networkx, numpy, sentence-transformers, faiss-cpu, matplotlib, pandas,
python-dotenv, neo4j, qdrant-client, google-generativeai, groq,
streamlit, pytest
```

The good news, verified by reading `GraphMemoryManager.__init__`, not
just the README's claim: it really is local-first at its core.
`GraphMemoryManager(neo4j_repository=None, vector_search=None)` — both
genuinely optional, defaulting to an in-memory `networkx.DiGraph()` — is
the path every example script actually uses; Neo4j/Qdrant writes are
gated behind `if self.neo4j_repository:` / equivalent, not required. "Zero
dependency by default" is MemArk's own stated design goal, and it holds
up against the source, not just the marketing line.

The real cost: `sentence-transformers` (used for its `all-MiniLM-L6-v2`
embeddings) pulls in **PyTorch** as a hard transitive dependency. This
project has explicitly avoided torch everywhere it could — Silero VAD is
the one place it's unavoidable, and both `CLAUDE.md` and `pyproject.toml`
call it out by name as "genuinely heavy" and deliberately excluded from
the `dev` extra. `docs/MEMORY_STORE.md` section 2 made the *opposite*
choice for Chroma specifically to avoid this: Chroma's bundled embedding
model runs on ONNX runtime, no torch. Adding MemArk as-is reintroduces the
exact dependency that decision was written to avoid. `faiss-cpu` is real
but reasonable — prebuilt wheels, no server, roughly comparable weight to
Chroma's own vector layer. `google-generativeai`/`groq` are cloud LLM
client SDKs that only `memark/llm/` imports — this plan never calls that
subpackage, so nothing here reaches out to a cloud API at runtime, but
they'd still land on disk from a plain `pip install memark`, worth being
honest about rather than silent on.

**Latency — measured, not estimated** (originally this section found zero
published numbers anywhere in MemEngine's own repo and left this as a
structural risk to measure later; that measurement has since been done —
what follows replaces the old placeholder, not just adds to it).

Important caveat up front: the structural worry this section originally
raised — `RetrievalPipeline.retrieve()` chaining embedding + FAISS +
graph-edge walks + hierarchy traversal + reinforcement + compression, all
synchronous — never actually applies to what this project shipped.
`docs/MEMARK_RETRIEVAL_ADAPTER.md`'s adapter deliberately never calls
`RetrievalPipeline`/`retrieve_semantic_nodes` at all (see
`docs/MEMARK_COMPANION_EXTENSIONS.md` section 3); it's built on a much
lighter concept-seed cosine-similarity step +
`ActivationEngine.propagate_activation()` + a plain recency/keyword scan.
So the worst case this section feared was never the thing actually
measured below — worth being explicit about, not just quietly measuring
something narrower and calling it the same question.

Real numbers, via `scripts/benchmark_memory_backends.py` (this project's
own script, rerunnable — see that file for the method: each backend
measured in its own subprocess so neither's "cold" timing benefits from
the other's warm-up; write and query timings use separate fresh
scopes so a cold write doesn't warm the scope a cold query is about to
measure). Both backends use the same embedding function (Chroma's
`DefaultEmbeddingFunction`) by default, so embedding cost is controlled
for — this isolates the storage/retrieval machinery itself. Fixture: the
16-turn hand-built companion conversation in that script (per this
document's own proposed shape, section 4 of
`docs/MEMARK_COMPANION_EXTENSIONS.md` — that note's fixture was never
actually built; this is a fresh one, same spirit). Hardware: this
project's own dev machine, Windows 11, 20 logical cores, CPU-only (no
GPU) — three separate runs, averaged:

| Metric | Chroma | MemArk | Delta |
|---|---|---|---|
| write, cold (first call, client+model construction) | ~1032ms | ~941ms | MemArk ~9% faster |
| write, warm avg (steady-state, n=15) | ~243ms | ~272ms | MemArk ~12% slower |
| query, cold, on an empty/new scope | ~482ms | ~930ms | **MemArk ~93% slower** |
| query, warm avg (steady-state, n=9) | ~446ms | ~473ms | MemArk ~6% slower |

**Reading these honestly**: in steady state — the case that matters for
an ongoing conversation — MemArk is roughly comparable to Chroma, not the
multi-x sequential-chain slowdown this section originally worried about;
both are dominated by the same shared embedding call, and the
lighter-than-`RetrievalPipeline` adapter design (above) is why. The one
real, disclosed exception: the *first* query against a brand-new
`(user_id, persona_id)` scope is nearly 2x slower for MemArk — that's
`_ensure_scope`'s hydration (`MemArkGraphStore.load()` + constructing
`GraphMemoryManager`/`FaissVectorStore`, even against an empty graph)
paying a real, one-time-per-scope-per-process cost Chroma's
`get_or_create_collection` doesn't have an equivalent of. It's memoized
after that first touch (`_ensure_scope`'s cache) — a real cost on the
very first turn of a new conversation with a new persona/user pairing
after a process (re)start, not a per-turn cost thereafter. Against
`handle_voice_turn`'s real STT/TTS/VAD latency budget, both backends'
numbers here are a small fraction of total turn time either way — this
isn't the thing that would make a voice turn feel slow.

**What this does and doesn't settle**: latency is no longer an open risk
blocking anything — it's measured, disclosed, and roughly comparable.
It does *not* override section 5's separate, still-fully-applicable
finding: MemArk trades raw recall for context cleanliness on its own
published benchmarks, and salience decay is a real behavioral difference
from Chroma's "everything stays retrievable forever." Those, not speed,
remain the reason this ships labeled experimental — see README.md's
MemArk section for exactly how that's now disclosed to someone choosing
this backend.

One more thing worth flagging here because it's a *behavioral* cost, not
just a performance one: salience decay/dormancy is a core, intentional
MemArk feature — nodes lose salience over time and stop surfacing in
retrieval until a matching cue "re-awakens" them. Chroma's store has no
such decay; everything ever written stays equally retrievable forever.
A companion that "used to remember X" and later doesn't — by design, not
by bug — is a materially different memory *feel* than what this project
ships today, independent of raw speed.

## 5. Showcase framing vs. product framing — a real tension, not just a note

MemArk's own README states its headline result plainly, and it cuts
against a companion app's core memory pitch. Quoting the numbers directly
(N=500 each, HotpotQA / 2WikiMultihopQA, against a flat Vector RAG
baseline):

| Metric | MemArk | Vector RAG |
|---|---|---|
| Context Compression Ratio (2Wiki) | 96.42% | 81.64% |
| Contamination Leakage Ratio (2Wiki) | 10.00% | 43.16% |
| Supporting Fact Recall (2Wiki) | 0.422 | **0.596** |
| Supporting Fact Recall (HotpotQA) | 0.369 | **0.575** |

That's MemArk's own paper's conclusion, not my inference: it trades raw
recall for context cleanliness. Translated to what a companion-app user
actually feels: "does it remember what I told it" is almost entirely a
recall question — the user never sees context-window compression or
contamination-avoidance as such, they only see whether the companion
recalls the right thing. On the metric users actually feel, MemArk's own
numbers say it's *worse* than what's already shipped. That's a concrete
tension, not a vague concern.

Where MemArk's real, honestly-earned strength shows up — multi-hop
evidence chains, resisting context contamination — is long,
information-dense, entity-and-relation-heavy conversation, closer to a
research-assistant persona than this project's existing three starter
personas (mentor/friend/companion), which trend emotionally continuous
rather than fact-dense. Run through ordinary daily companion chat, the
existing personas likely wouldn't exercise MemArk's strength at all, and
a straight A/B against Chroma could make MemArk look strictly worse —
which would be a misleading showcase in the *opposite* direction from
what's intended.

Salience decay (section 4) cuts the same way: genuinely interesting and
honest framed as "this memory ages like real memory" in a showcase
context, but it needs to be a disclosed, expected product behavior if
shipped as a real user-selectable option — "why did my companion forget
this" is a trust problem if it's a surprise.

None of this is a reason not to build it. It's a reason the eventual
pitch to a user choosing this backend should be specific and honest —
"keeps long, complex conversations from cluttering context, at some cost
to perfectly recalling every older detail" — not "better memory," the
same way this project has consistently preferred documenting real,
earned tradeoffs over marketing framing (e.g. every backend's honest
tested-vs-wired distinction in `docs/ARCHITECTURE.md`).

## 6. Explicitly out of scope for this note

- The user-facing backend-picker UI — a future UI-layer concern, not
  touched here.
- No adapter code, no `MemoryStore` subclass, nothing written against
  MemArk's modules. Design note only, per the gate.

## What I'd propose doing next, if this is approved

1. Land MemArk as a second, **config-selected, startup-time** backend
   only — `memory.backend: memark` as a new allowed value, chosen once at
   app startup exactly like every existing backend swap. No per-session
   picker in this pass (see section 3).
2. Adapter conforms MemArk's node-ranking output down to `list[Turn]`
   (Option A, section 2) — `MemoryStore`, `PromptBuilder`, `Orchestrator`
   all stay untouched.
3. Pin an exact commit hash of `Arkur745/MemEngine`, not a version range
   — there's no versioned public API to depend on instead (section 1).
4. ~~Before it ships as anything but experimental/opt-in: measure real
   wall-clock latency against Chroma on this project's own hardware~~ —
   done, see section 4's real numbers (`scripts/benchmark_memory_backends.py`):
   roughly comparable in steady state, one disclosed ~2x-slower cold-scope
   exception. Document the recall-vs-compression and salience-decay
   tradeoffs honestly wherever a user would pick this backend, not only
   in this internal note (section 5) — done in README.md's MemArk
   section, which now states plainly that speed isn't the reason this
   stays experimental; recall/decay still are.
5. Treat per-session backend switching as a separate, later, explicitly
   gated initiative, starting from whether this app needs a first-class
   `Conversation` concept at all (section 3) — not something to bundle
   into landing MemArk.

Stopping here per the gate. Want me to proceed with any part of this, or
discuss the recommendations first?
