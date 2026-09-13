# Vendored MemEngine source

This directory contains **unmodified, byte-identical copies** of specific
files from [Arkur745/MemEngine](https://github.com/Arkur745/MemEngine)
(the "MemArk" project) — not the whole package, and not installed via
pip. See `docs/MEMARK_RETRIEVAL_ADAPTER.md` for the design this exists to
support, and the conversation that led to vendoring rather than adding
`memark` as a full pip dependency: the whole package's own
`pyproject.toml` declares torch (via sentence-transformers), neo4j,
qdrant-client, google-generativeai, groq, and streamlit as unconditional
dependencies, none of which anything in this adapter ever calls. Tracing
the actual imports (including function-level ones) of just the files
this project needs showed the real cost is `networkx` + `faiss-cpu` —
both added as this project's own `memory-memark` optional-dependency
group, nothing else.

**Source**: `https://github.com/Arkur745/MemEngine.git`
**Commit**: `5a67fbb1e1ae8316732afe8f0cda37e016ca20f7`
**License**: MIT (see `LICENSE_MEMENGINE` in this directory — copied
verbatim from the source repo's own `LICENSE`) — see this project's own
`THIRD_PARTY_LICENSES.md` for how this fits alongside every other
bundled/vendored dependency.

## Files included, and why

Only what the retrieval adapter's approved design
(`docs/MEMARK_RETRIEVAL_ADAPTER.md`) actually calls — traced by reading
every file's own imports (module-level and function-level) before
copying anything, not guessed:

- `engine/core/graph_manager.py` (`GraphMemoryManager`) — the live
  in-memory graph the adapter owns and mutates.
- `engine/core/memory_node.py` (`MemoryNode`), `engine/core/memory_edge.py`
  (`MemoryEdge`), `engine/core/types.py` (`NodeType`/`RelationType`) —
  the data shapes those manage.
- `engine/retrieval/activation_engine.py` (`ActivationEngine`) — the
  spreading-activation retrieval algorithm itself; the actual reason
  this needs real MemArk code rather than a local reimplementation (see
  `docs/MEMARK_RETRIEVAL_ADAPTER.md` — reimplementing this would defeat
  the point of using MemArk's own, benchmarked algorithm).
- `engine/retrieval/faiss_store.py` (`FaissVectorStore`) and
  `engine/retrieval/vector_store.py` (`BaseVectorStore`, the abstract
  base it implements) — used **directly**, not through
  `engine/retrieval/vector_search.py`'s wrapper class, because that
  wrapper unconditionally imports `qdrant_store.py` (→ `qdrant_client`,
  a dependency this project has no use for and doesn't want). Confirmed
  by reading `vector_search.py`'s own imports before deciding to skip it.
- `engine/config/default_config.py` (+ its own `__init__.py`) —
  `ActivationEngine` reads default parameters from this module's globals
  when a caller doesn't override them via constructor arguments (see
  `docs/MEMARK_COMPANION_EXTENSIONS.md` section 1 on this being real,
  process-global state — not depended on for overrides in this adapter,
  only for the library's own internal defaults to resolve at all).
- `engine/storage/memory_repository.py` (`MemoryRepository`) — thin
  wrapper `GraphMemoryManager` construction doesn't strictly require, but
  `ActivationEngine`'s constructor takes a `memory_repository`, not a
  `GraphMemoryManager`, directly.
- `engine/logging/__init__.py` + `engine/logging/cognitive_logger.py` —
  a function-level `from engine.logging import get_logger` inside
  `graph_manager.py`'s `create_node`/`sync_node` and
  `memory_repository.py`'s `create_semantic_edges` would otherwise fail
  at call time; found by grepping every target file's full body for
  `import`, not just its header.

**Not included, deliberately**: `engine/agent/`, `engine/llm/`,
`engine/retrieval/vector_search.py` (the qdrant-coupled wrapper),
`engine/retrieval/qdrant_store.py`, `engine/hierarchy/` (not used by
this adapter's approved query sequence — that was `RetrievalPipeline`'s
own pipeline, which the design explicitly avoids), `engine/consolidation/`,
`engine/planning/`, `engine/goals/`, `engine/metacognition/`,
`engine/observability/`, `engine/validation/`, `engine/persistence/`
(MemEngine's own Neo4j-only loader — this project's persistence is
`companion/memory/memark_store.py`, unrelated and untouched by this).

## Integrity

Every file here was diffed against the source clone before being
committed and confirmed byte-identical — zero modifications, including
no import-path rewrites (these files use absolute imports like `from
engine.core.memory_node import MemoryNode`, which resolve correctly as
long as this `_vendor/` directory's path is on `sys.path` — see
`companion/memark/_engine_path.py` for the one place that happens).

## Keeping this in sync

This is a frozen snapshot, not a live dependency — no automatic
upstream bugfixes. If MemEngine fixes something in one of these files
later, it has to be manually re-vendored (re-copy, re-diff, update the
commit hash above) — there's no tooling for this yet, deliberately
(revisit if vendoring more files later makes that worth building).
