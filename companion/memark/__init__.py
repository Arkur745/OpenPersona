"""The MemArk retrieval adapter — makes companion/memory/memark_store.py's
MemArkGraphStore (persistence) into a real MemoryStore backend, per
docs/MEMARK_RETRIEVAL_ADAPTER.md. See that module's own docstring for
what is/isn't wired in yet.

`_vendor/` holds specific MemEngine source files, vendored (not pip
installed) — see `_vendor/PROVENANCE.md` for exactly what and why.
"""
