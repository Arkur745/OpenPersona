# MemArk companion-domain extensions — design note (partly implemented)

> **Status (2026-10-19), item by item:**
> - **Built:**
>   - persistence, the section 1 prerequisite: `docs/MEMARK_PERSISTENCE.md`;
>   - hybrid retrieval, section 2a: `docs/MEMARK_RETRIEVAL_ADAPTER.md` section 2;
>   - the section 3 correction (the adapter uses `ActivationEngine`
>     directly, never `RetrievalPipeline`).
> - **Not built:**
>   - decay-rate tuning (2b);
>   - a companion-specific node type (2c);
>   - the recall@k evaluation (section 4).
> The rest of this note is the original proposal.

Same gate as `docs/MEMARK_BACKEND.md`: design note only, nothing
implemented. Builds directly on that note's finding — MemArk's own
published benchmarks show it recalling fewer supporting facts than flat
vector RAG, a tradeoff tuned against dense, fact-heavy, single-shot QA
retrieval (HotpotQA, 2WikiMultihopQA), not the conversational, recurring,
often emotionally-continuous memory a companion app actually needs.
Everything below is grounded in re-reading MemEngine's actual source
(same clone used for the prior note and the `retrieve_semantic_nodes`
investigation) — constructor signatures, not impressions.

**One thing this pass found that neither prior note did, stated up front
because it changes the shape of everything else here**: MemArk's graph
has no local-disk persistence at all. See section 1's last point before
reading the rest as if the recall question were the only open one.

## 1. What's actually tunable from outside `engine/`, audited concretely

**Genuinely clean constructor injection — no core edits, no global state:**

- `LongTermDecayEngine(memory_repository, decay_rates: Dict[str, float] =
  None)` (`engine/consolidation/long_term_decay_engine.py:12`) — pass a
  custom per-node-type decay-rate dict directly; if omitted it copies
  `_cfg.DECAY_RATES` once at construction. This is the real lever for
  "companion memories should fade slower than QA-benchmark facts."
- `ActivationEngine(memory_repository, base_activation=None,
  decay_factor=None, activation_threshold=None, max_depth=None)`
  (`engine/retrieval/activation_engine.py:11`) — all four spreading-
  activation parameters are overridable per-instance; each falls back to
  `_cfg.*` only if not given.
- `ScoringEngine(alpha=1.0, beta=0.5, gamma=0.3, delta=0.2, epsilon=0.4)`
  (`engine/retrieval/scoring_engine.py:7`) — semantic/importance/
  confidence/frequency/recency weights, all constructor args.
- `ReinforcementManager(reinforcement_factor=0.05, max_importance=1.0,
  max_confidence=1.0)` (`engine/memory/reinforcement_manager.py:6`) —
  clean, no hidden state.

**Extensible via subclassing or post-construction attribute mutation on
an object *we* own (not shared/global — safe, local, no core edit):**

- `ScoringEngine.type_weights` (`{"concept": 1.4, "summary": 1.2,
  "episodic": 0.8}`) is a plain instance attribute set inside `__init__`,
  not a constructor parameter. `type_weights.get(node.node_type, 1.0)`
  falls back to a neutral `1.0` for anything not in the dict — so either
  subclassing `ScoringEngine` to set a different dict, or just doing
  `scoring_engine.type_weights["commitment"] = 1.3` on an instance our
  own adapter constructed, works with zero risk to anything else, since
  we're mutating an object nothing else shares.

**One real wrinkle — global module state, not a file edit, but not clean
DI either:** `LongTermDecayEngine.__init__` reads `self.DORMANCY_THRESHOLD
= _cfg.DORMANCY_THRESHOLD` directly from `engine.config.default_config`
at construction time — there's no constructor parameter for it. Several
behavior flags (`ENABLE_GOAL_BIASING`, `ENABLE_SALIENCE_DECAY`,
`ENABLE_DORMANCY`, `ENABLE_SPREADING_ACTIVATION`, etc., read via
`getattr(_cfg, "FLAG", True)`) are read live on every call, confirmed by
reading `LongTermDecayEngine.run_decay_cycle` and `ActivationEngine
.propagate_activation` directly — this is in fact how the benchmark
evaluators themselves configure ablation modes (`import engine.config
.default_config as _cfg; _cfg.ENABLE_SPREADING_ACTIVATION = False`, e.g.
`benchmarks/hotpotqa/evaluator.py:106-126`). So it's genuinely possible
to tune these from our adapter, without touching any MemEngine file — but
only by mutating a live import of MemEngine's own shared module state,
which is process-global and not scoped per instance or per call.
**Recommendation for this note's design: don't depend on it.** Everything
recommended below uses only the clean constructor-level knobs above;
`DORMANCY_THRESHOLD`'s global-only override is noted as a known,
available-but-not-clean lever, not something the recommended design
relies on.

**Node types are not a closed system at runtime — confirmed by absence,
not by a stated design choice.** `GraphMemoryManager.create_node`'s
signature is `create_node(self, node_type: str, summary: str, ...)` —
typed as plain `str`, not `NodeType` (`engine/core/graph_manager.py:48`).
I grepped the entire `engine/` tree for `NodeType.` usage outside
`engine/core/types.py` itself: **zero matches.** Every runtime check I
found (`LongTermDecayEngine`, `ScoringEngine`, both benchmark evaluators)
compares `node.node_type` against plain string literals (`"goal"`,
`"concept"`, ...), which works today only because `NodeType` is a
`StrEnum` — but nothing anywhere validates that `node_type` is one of the
nine declared values. `HierarchyTraversal.traverse_hierarchy`
(`engine/hierarchy/hierarchy_traversal.py`) is fully type-agnostic — it
walks `parent_id`/`child_ids` structurally, never inspecting
`node_type` at all. `ActivationEngine.propagate_activation` never gates
on the seed node's type either. Net effect: **a novel node type like
`"personal_fact"` or `"commitment"` is usable today, from our adapter,
with zero MemEngine changes** — it just won't get a custom decay rate or
scoring weight unless we also supply those (both possible per the two
points above).

**The critical finding this pass surfaced: no local persistence for the
graph itself.** I searched exhaustively (`def (save|load|dump|serialize
|persist)` across `engine/`, plus `pickle`/`sqlite`/`shelve` usage) and
found exactly one loader in the entire codebase:
`engine/persistence/graph_loader.py`'s `GraphLoader.load_from_neo4j` —
hardwired to a `neo4j_repository`, the only place graph state is ever
reconstructed from anywhere. Every caller of `GraphLoader` in the repo
(`archive/scratch/*.py`, `tests/agent_test.py`, `tests/benchmark_test
.py`, `tools/app.py`) constructs it with a real Neo4j repository.
`GraphMemoryManager` itself has no `save`/`dump`/`export` method at all
(confirmed by listing every method on the class). `VectorSearch`'s FAISS
backend (`engine/retrieval/faiss_store.py`) has no disk persistence
either — `add_embedding`/`search`/`delete`/`clear`, nothing that writes
to disk. `QdrantVectorStore` genuinely does support a local, serverless,
on-disk mode (`QdrantClient(path=...)`, no server) — but that only
persists the *vector index*, not the graph itself (node metadata,
salience scores, hierarchy edges); without the graph surviving too, a
persisted vector index pointing at node IDs that no longer exist is
useless. **In short: without a running Neo4j server, everything
`GraphMemoryManager` holds is process-memory-only and evaporates the
moment the process exits.** MemArk's own "zero-dependency by default"
claim (quoted approvingly in the prior note) is true for a single
in-process benchmark run, where the graph only needs to exist for the
duration of one Python process — it says nothing about surviving a
restart, which is the entire point of persistent memory in a companion
app that gets invoked fresh every session. This means **local persistence
for the graph is a new, previously-unscoped, must-build prerequisite** —
not something either this note or `docs/MEMARK_BACKEND.md` accounted for
until now — and it's a real, non-trivial adapter component, not a
one-liner. It is, however, fully buildable in our own code without
touching MemEngine's files: `GraphLoader`'s own `_reconstruct_node`/
`_node_to_dict` methods (`dataclasses.asdict()` on `MemoryNode`/
`MemoryEdge`, timestamp string round-tripping) are the exact shape needed
— we'd write an equivalent loader/saver in our adapter that reads/writes
JSON (or sqlite) instead of Neo4j, using the same public `MemoryNode`/
`MemoryEdge` dataclasses and `GraphMemoryManager.nodes`/`.edges`/
`create_node`/`create_edge`. This doesn't change the "no core touch"
scoping — it adds a genuinely new item to what "add MemArk as a backend"
actually requires, on top of everything scoped in the prior note.

## 2. A recall-recovery mechanism specific to this domain

All three options evaluated below assume the persistence prerequisite in
section 1 gets solved first — none of them matter for a real product
until the graph survives a restart. The recommendation among them doesn't
change based on that, though.

**a. Hybrid retrieval — MemArk's graph retrieval + a simple recency/
keyword fallback, merged at the adapter boundary.** Concretely: run our
adapter's own graph-retrieval call (concept-seed via cosine similarity,
then `ActivationEngine.propagate_activation()` — see section 3 for why
this, not `RetrievalPipeline`) *and*, independently, a trivial scan over
the same stored episodic/personal-fact nodes scored by simple recency +
keyword/substring overlap against the query — merge and dedupe both
candidate sets before converting to `list[Turn]`. Entirely adapter-side;
the fallback half needs nothing from MemArk's internals beyond
`MemoryRepository.get_all_nodes()`.
  - Directly, unconditionally addresses the exact gap the prior note
    measured: graph-seeded activation can simply miss a fact a flat scan
    wouldn't, because it depends on finding the right entry concept first
    (see the benchmark evaluators' own "seed routing drift" metric,
    which exists precisely because that seeding step misses sometimes).
    A parallel fallback puts a floor under recall regardless of how well
    seeding/activation performs on a given query.
  - Doesn't require understanding or correctly tuning any of MemArk's
    internal salience/activation math to get a benefit — which matters,
    because (per section 1) some of the machinery whose tuning *would*
    help (decay rates, activation thresholds) has non-obvious interaction
    effects that are hard to reason about from outside without the kind
    of ablation infrastructure MemArk's own benchmark suite has and this
    project doesn't.

**b. Tune salience decay for conversational cadence.** Now grounded
concretely (section 1): `decay_rates` is a clean constructor param, so
per-type rates are genuinely settable — e.g. a much slower rate for
whatever type holds "things the persona said it would remember." Two
real limits, not just a tuning exercise:
  - Decay is not automatic/background — `run_decay_cycle()` has to be
    explicitly invoked (there's no timer or hook anywhere in the engine
    that calls it). *When* our adapter calls it (once per session start?
    on a schedule?) is itself a real design decision this option doesn't
    resolve by itself.
  - Decay only controls whether old memories go dormant/suppressed — it
    does nothing about the actual gap the prior note measured, which is
    entry-point seeding accuracy. Tuning it doesn't make a badly-seeded
    query find the right concept; it only affects whether a *found* node
    survives long enough to still be findable later.
  Worth doing, but as a secondary knob layered onto something else
  (section 2c below), not a fix for the recall gap on its own.

**c. A companion-specific node type** (`"personal_fact"`, `"commitment"`,
`"recurring_topic"`). Confirmed genuinely, cleanly buildable from outside
(section 1: unvalidated string `node_type`, type-agnostic hierarchy
traversal, per-type decay rate and scoring weight both settable without
touching core). But it's a *labeling and weighting* mechanism, not a
retrieval-improving one by itself — a new type still needs to be
embedded, registered in vector search, and (if desired) linked into the
hierarchy exactly like any existing type before anything can find it via
graph traversal at all. It's a real, cheap complement to (a), not an
alternative to it.

**Recommendation: build (a) first.** It's the only one of the three that
directly, unconditionally targets the specific, measured gap (recall)
rather than something that might help it indirectly. Concretely staged:
(a) hybrid retrieval as the core recall-recovery mechanism, (c) a
`"personal_fact"`/`"commitment"` node type as a cheap, additive follow-up
so decay and scoring can treat those distinctly, (b) a slower decay rate
specifically for that new type (not a global retune of MemArk's default
rates) as the concrete use of (c) once it exists. This mirrors the same
reasoning the prior note used for Option A over B on the `MemoryStore`
contract question: solve the measured problem at the boundary we
control, rather than trying to re-tune internals whose interaction
effects aren't something this project has the infrastructure to validate
the way MemArk's own benchmark suite does.

## 3. Interaction with the still-open `retrieve_semantic_nodes` investigation

**This design does not depend on that bug's outcome, and it's worth being
precise about why rather than just asserting it.** The investigation (now
resolved: the bug doesn't reach the paper's numbers, because
`RetrievalPipeline`/`retrieve_semantic_nodes` is never called by the
actual benchmark harness — see that investigation's report) also
established *what the benchmark-validated retrieval path actually is*:
`_find_concept_seed_node` (a hand-rolled cosine scan)  →
`ActivationEngine.propagate_activation()`, called directly, never through
`RetrievalPipeline`. That's the path this note's hybrid-retrieval design
(2a) uses for its "graph retrieval" half.

This is worth flagging as a **correction to the prior note's Option A
wording**, not just a non-dependency: `docs/MEMARK_BACKEND.md` section 2
described the adapter as calling "the same node-ranking stages
`RetrievalPipeline.retrieve()` already chains." Now that `RetrievalPipeline`
is known to be an orphaned code path — zero test coverage before this
investigation, zero callers anywhere except one unreferenced demo script,
and it had a live, unnoticed correctness bug for however long it's
existed — the adapter should **not** call `RetrievalPipeline` or
`retrieve_semantic_nodes` at all, regardless of the bug now being fixed.
It should replicate the stages using the same underlying components the
benchmark harness itself uses and that produced the paper's actual
numbers (`ActivationEngine`, direct concept-seed lookup), which is both
more validated and avoids depending on a path this project has no reason
to trust yet. If the bug had turned out to affect the paper's numbers,
this design would be unaffected either way, precisely because it was
never built on that path.

## 4. How this would be evaluated

Per `docs/MEMARK_BACKEND.md` section 4: zero published latency numbers
exist anywhere in MemEngine's repo, and no companion-domain evaluation
exists anywhere (MemArk's own benchmarks are QA-retrieval, not
conversational-memory). Nothing to lean on here either — this needs a
small, honest, purpose-built evaluation, not a full benchmark suite.

**Proposed shape**: a hand-built set of ~10-20 multi-turn companion
conversations, each seeded with a small number (2-4) of concrete facts a
real user might mention in passing (an allergy, a sibling's name, a
recurring commitment like "remind me to call my mom on her birthday"),
spread across *simulated* elapsed time — fabricated timestamps spanning
days/weeks, not real wall-clock waiting, since decay/recency need to be
exercised at conversational timescales, and this project can't sit
around for a week per test run. Each conversation pairs with a handful of
later-session recall queries whose correct answer is one of the earlier
facts. Three arms, same fixture set: plain Chroma (today's backend),
unmodified MemArk (no extensions), MemArk + this note's extension layer.

**Primary metric: recall@k** — does the correct fact surface in the
top-k retrieved memories for each recall query. This directly targets
the gap the whole investigation started from. **Secondary metrics**:
retrieved-context size (to confirm the extension layer doesn't fully
sacrifice MemArk's one genuine, benchmark-proven strength — compression —
while fixing recall) and wall-clock latency per query (closing the actual
gap identified in `docs/MEMARK_BACKEND.md` section 4, specific to this
project's own hardware and this domain, not extrapolated from QA-
benchmark numbers that don't exist anyway).

**One dependency worth stating plainly**: testing recall *across
sessions* — the realistic scenario, and the one a companion app actually
needs — requires the persistence prerequisite from section 1 to exist
first, or the eval harness has to fake "a new session" by keeping the
same in-process `GraphMemoryManager` alive across simulated turns rather
than truly restarting, which would not exercise the same code path a
real `python -m companion` restart would. Worth building the persistence
layer before or alongside this evaluation, not after.

This should be scoped and framed as a small fixture-driven smoke
evaluation this project owns and can rerun — a few hand-authored
conversations and expected-recall assertions, closer to an integration
test than a paper-grade benchmark — enough to know "does this help,
directionally, on our own domain" honestly, which is the standard this
project has held every other component to, not enough to publish.

## 5. Confirming this stays out of MemEngine's own repo

Every mechanism proposed above is buildable entirely in this project's
own adapter/extension code:

- Hybrid retrieval (2a): pure adapter code — our own fallback scorer,
  our own merge logic, calling only public `GraphMemoryManager`/
  `MemoryRepository`/`ActivationEngine` methods.
- Node type extension (2c): `create_node(node_type="personal_fact", ...)`
  — a public method call with a plain string argument, nothing to edit.
- Decay-rate tuning (2b): the `decay_rates` constructor parameter — no
  core file changes.
- Scoring-weight tuning: subclassing `ScoringEngine` or mutating
  `.type_weights` on an instance we constructed — no core file changes,
  no shared state.
- Local persistence (the section 1 finding): a new module in *our*
  codebase, reading/writing our own file format, using MemArk's public
  dataclasses and manager methods — modeled on `GraphLoader`'s own
  reconstruction logic but never editing or depending on
  `engine/persistence/graph_loader.py` itself.

**The one thing flagged as not clean**, restated plainly rather than
glossed over: overriding `DORMANCY_THRESHOLD` requires mutating
`engine.config.default_config`'s live module state — not a file edit,
so it doesn't cross the "don't touch MemEngine's repo" line, but it *is*
process-global shared state belonging to someone else's package, not
scoped dependency injection. The recommended design (section 2) doesn't
use it. If it's ever genuinely needed later, it should be called out
explicitly at that point as "mutates MemEngine's global config," not
quietly folded in as if it were as clean as the constructor-level knobs.

**Nothing proposed in this note requires editing `Arkur745/MemEngine`'s
own files.** The one thing that *would* have required touching core — a
save/load mechanism, since none exists — is solvable entirely in our own
code instead, per section 1's last point, so it doesn't force an
exception to this rule; it just adds real scope.

## Recommendation summary

1. Treat local graph persistence (section 1) as a new P0 prerequisite,
   scoped and estimated on its own — it blocks everything else here from
   being usable in a real product, not just this note's extensions.
2. Build hybrid retrieval (2a) first among the three recall-recovery
   options — it's the one that directly targets the measured gap without
   depending on correctly tuning internals this project can't easily
   validate the way MemArk's own benchmark suite can.
3. Layer in a companion-specific node type (2c) with its own tuned decay
   rate (2b) as a cheap, additive follow-up, not a first move.
4. Build the adapter's graph-retrieval path against `ActivationEngine`
   directly (concept-seed → propagate), never against `RetrievalPipeline`
   — independent of, and unaffected by, the `retrieve_semantic_nodes` bug
   investigation's outcome. Treat this as a correction to
   `docs/MEMARK_BACKEND.md` section 2's wording, not just a footnote here.
5. Before treating any of this as done: run the small hand-built
   recall@k evaluation from section 4 against Chroma and unmodified
   MemArk, and report the honest numbers — directionally useful is the
   bar, not publishable.

Stopping here per the gate. Want to discuss any of this before anything
gets implemented, starting with the persistence prerequisite since it's
the biggest change to the plan this pass turned up?
