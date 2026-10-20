"""Memory storage and maintenance:
- `store.py`: `VectorMemoryStore`, the default Chroma-backed `MemoryStore`,
  one collection per (user, persona).
- `memark_store.py`: `MemArkGraphStore`, SQLite persistence for the
  experimental MemArk backend (the `MemoryStore` adapter itself is in
  `companion/memark/retrieval.py`).
- `summarizer.py`: `RollingSummarizer`, a background job that replaces a
  scope's oldest turns with one summary turn.

See docs/MEMORY_STORE.md and docs/ROLLING_SUMMARIZER.md.
"""
