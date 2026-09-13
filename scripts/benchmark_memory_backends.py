"""Wall-clock latency smoke benchmark: `VectorMemoryStore` (Chroma) vs
`MemArkMemoryStore` (MemArk), `write()`/`query()`, cold vs warm, on this
project's own hardware.

Per `docs/MEMARK_BACKEND.md` section 4: no published latency numbers
exist for either backend, and this project explicitly held off shipping
MemArk as anything but experimental until this was actually measured
here, not estimated. Small, honest, rerunnable smoke measurement, not a
publishable benchmark suite — same "directionally useful, not
publishable" standard `docs/MEMARK_COMPANION_EXTENSIONS.md` section 4
set for a (separate, unbuilt) recall-quality evaluation.

Both backends use the exact same embedding function by default (Chroma's
own standalone `DefaultEmbeddingFunction`, ONNX MiniLM — `MemArkMemoryStore`
was specifically built to reuse it, see `companion/memark/retrieval.py`'s
module docstring), so embedding cost is controlled for; what this measures
is the storage/retrieval machinery layered on top of it.

## Why each backend runs in its own subprocess

Running both backends in one process would let whichever one measures
its "cold" (first-ever-call) timing second unfairly benefit from
in-process caching the first backend already paid for (ONNX runtime
session setup, embedding model weights already resident in memory,
Python import caches). Each backend's full measurement — one fresh
`persist_dir` for the write timings, a second fresh `persist_dir` for
the query timings (kept separate so a cold *write* doesn't warm the
scope a cold *query* is about to measure, and vice versa) — runs as its
own subprocess via `--backend`, so "cold" genuinely means "the first
thing this process ever did against this store," for both backends
equally.

Usage:
    python scripts/benchmark_memory_backends.py            # both backends, combined report
    python scripts/benchmark_memory_backends.py --backend chroma   # one backend, prints JSON to stdout
    python scripts/benchmark_memory_backends.py --backend memark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from companion.core.interfaces.types import Turn

USER_ID = "benchmark-user"
PERSONA_ID = "benchmark-persona"

# A hand-built, ~16-turn (8-exchange) companion conversation seeded with a
# handful of concrete, recallable facts — the shape
# docs/MEMARK_COMPANION_EXTENSIONS.md section 4 proposed (10-20 turns,
# 2-4 concrete facts) for evaluating this domain, not extrapolated from
# MemArk's own QA-benchmark fixtures. That note's fixture was never
# actually built (design note only) — this is a fresh one, same spirit,
# scoped to latency here rather than that note's recall@k proposal.
CONVERSATION: list[Turn] = [
    Turn(role="user", text="Hey, I wanted to tell you I'm allergic to peanuts, just so you know for future reference."),
    Turn(role="assistant", text="Thanks for telling me — I'll keep that in mind. Peanut allergies can be serious, so I'm glad you mentioned it."),
    Turn(role="user", text="My sister Maya is training for a marathon in June, she's been really dedicated about it."),
    Turn(role="assistant", text="That's impressive! June marathons take a lot of preparation. Is this her first one?"),
    Turn(role="user", text="Yeah, her first. Also, remind me to call my mom on her birthday, it's coming up soon."),
    Turn(role="assistant", text="Got it — I'll remember you want a reminder to call your mom on her birthday."),
    Turn(role="user", text="Work has been stressful lately, we're launching a new product next month."),
    Turn(role="assistant", text="That sounds like a lot of pressure. How are you holding up with the launch prep?"),
    Turn(role="user", text="Okay I guess. Been trying to go for runs in the evening to clear my head."),
    Turn(role="assistant", text="Evening runs are a great way to decompress. Have you kept a consistent routine?"),
    Turn(role="user", text="Sort of, three or four times a week. Anyway, what was that recipe you mentioned last week?"),
    Turn(role="assistant", text="I don't have a specific recipe on file from last week — want to tell me again what you were looking for?"),
    Turn(role="user", text="Never mind, I'll figure it out. Did I mention my nephew just started kindergarten?"),
    Turn(role="assistant", text="You didn't, that's exciting! How's he adjusting so far?"),
    Turn(role="user", text="He loves it, apparently. Okay, I should get back to work."),
    Turn(role="assistant", text="Sounds good — good luck with the launch. Talk soon!"),
]

QUERIES = [
    "What am I allergic to?",
    "What is my sister training for?",
    "What did I ask you to remind me about?",
]


async def _timed(coro) -> float:
    start = time.perf_counter()
    await coro
    return time.perf_counter() - start


def _build_store(backend: str, persist_dir: str):
    if backend == "chroma":
        from companion.memory.store import VectorMemoryStore

        return VectorMemoryStore(backend="chroma", persist_dir=persist_dir)
    if backend == "memark":
        import logging

        from companion.memark._engine_path import ensure_engine_importable

        # Must run as an explicit call, not just an import, and before
        # importing engine.logging below — puts the vendor dir on
        # sys.path so `engine.*` is importable at all (companion.memark
        # .retrieval does this too on its own import, but this script
        # needs engine.logging configured *before* that import triggers
        # the first create_node() call, so it's called explicitly here
        # rather than relying on retrieval.py's own import as a side
        # effect). NOTE: keep this a function call sandwiched between the
        # two imports below, not a plain import statement itself — `ruff
        # --fix`'s isort integration doesn't know these two imports have
        # a real ordering dependency and will happily "sort" them back
        # into the broken order if nothing non-import separates them
        # (confirmed by hitting exactly that break once already).
        ensure_engine_importable()

        # The vendored engine.logging.cognitive_logger attaches a
        # StreamHandler(sys.stdout) and logs every create_node() call at
        # INFO — harmless noise in the real app, but fatal here:
        # --json-only mode needs this script's own JSON line to be the
        # only thing on stdout. Its configure_logging() doesn't actually
        # run at `companion.memark.retrieval` import time, though —
        # graph_manager.py imports engine.logging lazily, *inside*
        # create_node() itself (confirmed by testing this exact ordering
        # directly, not assumed) — so importing engine.logging here
        # explicitly first (forcing configure_logging() to run now, once)
        # before overriding the level is required; overriding the level
        # first and importing second would just get silently clobbered
        # back to INFO the first time create_node() runs. Raising the
        # level from this script, not touching the vendored file.
        import engine.logging as _engine_logging  # noqa: F401 — see above

        logging.getLogger("memengine").setLevel(logging.WARNING)

        from companion.memark.retrieval import MemArkMemoryStore

        return MemArkMemoryStore(persist_dir=persist_dir)
    raise ValueError(f"unknown backend {backend!r}")


async def _bench_writes(backend: str, persist_dir: str) -> dict:
    store = _build_store(backend, persist_dir)
    times = [await _timed(store.write(USER_ID, PERSONA_ID, turn)) for turn in CONVERSATION]
    return {
        "write_cold_s": times[0],
        "write_warm_avg_s": statistics.mean(times[1:]),
        "write_warm_stdev_s": statistics.stdev(times[1:]),
        "write_warm_n": len(times[1:]),
    }


async def _bench_queries(backend: str, persist_dir: str) -> dict:
    store = _build_store(backend, persist_dir)
    # True cold query: the very first call this process makes against this
    # store, on a completely empty scope — nothing written yet. Matches
    # what Orchestrator.handle_text_turn actually does each real turn
    # (query() before write()), not an artificial isolated case.
    cold_query_s = await _timed(store.query(USER_ID, PERSONA_ID, QUERIES[0], k=5))

    for turn in CONVERSATION:
        await store.write(USER_ID, PERSONA_ID, turn)  # untimed — just populating

    warm_times = []
    for query_text in QUERIES:
        for _ in range(3):
            warm_times.append(await _timed(store.query(USER_ID, PERSONA_ID, query_text, k=5)))

    return {
        "query_cold_on_empty_scope_s": cold_query_s,
        "query_warm_avg_s": statistics.mean(warm_times),
        "query_warm_stdev_s": statistics.stdev(warm_times),
        "query_warm_n": len(warm_times),
    }


async def _run_one_backend(backend: str) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"memory_bench_{backend}_"))
    try:
        write_dir = tmp / "writes"
        query_dir = tmp / "queries"
        write_results = await _bench_writes(backend, str(write_dir))
        query_results = await _bench_queries(backend, str(query_dir))
        return {"backend": backend, **write_results, **query_results}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_in_subprocess(backend: str) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__, "--backend", backend, "--json-only"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _print_report(results: list[dict]) -> None:
    header = (
        f"{'backend':<8} {'write cold':>12} {'write warm avg':>16} {'query cold':>12} "
        f"{'query warm avg':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['backend']:<8} "
            f"{r['write_cold_s']*1000:>10.1f}ms "
            f"{r['write_warm_avg_s']*1000:>14.1f}ms "
            f"{r['query_cold_on_empty_scope_s']*1000:>10.1f}ms "
            f"{r['query_warm_avg_s']*1000:>14.1f}ms"
        )
    print()
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["chroma", "memark"], default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    if args.backend is not None:
        result = asyncio.run(_run_one_backend(args.backend))
        if args.json_only:
            print(json.dumps(result))
        else:
            _print_report([result])
        return

    results = [_run_in_subprocess("chroma"), _run_in_subprocess("memark")]
    _print_report(results)


if __name__ == "__main__":
    main()
