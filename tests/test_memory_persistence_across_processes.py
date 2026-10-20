"""Memory persists across real process restarts, for both MemoryStore backends.

Every step here runs in its own short-lived Python process
(tests/_persistence_child.py), against one shared on-disk directory. The
writer process has fully exited before the reader starts, so nothing can
be carried over in process memory — only what actually reached disk. That
is the gap left by the same-process "fresh store instance" tests in
tests/test_memark_retrieval.py and the absence of any restart test for
Chroma at all.

Three levels:
- store level (write / query / delete), parametrized over `chroma` and
  `memark`, built through `companion.app.build_memory` exactly as the app
  builds it;
- orchestrator level: a fact said in one session's `handle_text_turn`
  shows up in the next session's LLM prompt, with an empty
  `recent_turns` (deterministic: a recording fake LLM, real memory, real
  PromptBuilder);
- the real app: two separate `python -m companion` text-mode runs against
  a real local Ollama server, where the second run must answer a question
  only the first run's memory can answer. Skipped when Ollama isn't
  reachable or has no models, like every other real-Ollama test here.

Both backends use the real default embedding model (Chroma's ONNX
MiniLM), downloaded on first use; these tests are slower than the rest of
the suite for that reason (each child process imports chromadb).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CHILD = Path(__file__).resolve().parent / "_persistence_child.py"
CHILD_TIMEOUT_S = 300


def _backend_param(name: str, *required_modules: str):
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    marks = [pytest.mark.skip(reason=f"needs {', '.join(missing)}")] if missing else []
    return pytest.param(name, marks=marks, id=name)


BACKENDS = [
    _backend_param("chroma", "chromadb"),
    _backend_param("memark", "chromadb", "networkx", "faiss"),
]


def _child(command: str, backend: str, persist_dir: Path, **kwargs) -> dict:
    argv = [sys.executable, str(CHILD), command, "--backend", backend,
            "--persist-dir", str(persist_dir)]
    for key, value in kwargs.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    proc = subprocess.run(
        argv, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
        timeout=CHILD_TIMEOUT_S, check=False,
    )
    assert proc.returncode == 0, f"child {command} failed:\n{proc.stderr}"
    # Libraries may log to stdout; the child's JSON result is the last line.
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _turn(role: str, text: str, minute: int) -> dict:
    ts = datetime(2026, 10, 1, 12, minute, tzinfo=UTC)
    return {"role": role, "text": text, "timestamp": ts.isoformat()}


# --- store level ----------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_turns_written_in_one_process_are_returned_exactly_in_the_next(backend, tmp_path):
    turns = [
        _turn("user", "I am allergic to peanuts", 0),
        _turn("assistant", "Got it, I'll remember your peanut allergy.", 1),
    ]
    assert _child("write", backend, tmp_path, turns=json.dumps(turns)) == {"written": 2}

    results = _child("query", backend, tmp_path, query="peanut allergy", k=5)["results"]

    # Exact equality including timestamps: the orchestrator's dedup
    # compares whole Turns, so a timestamp that drifts on the way through
    # disk would silently break it after every restart.
    assert sorted(results, key=lambda t: t["timestamp"]) == turns


@pytest.mark.parametrize("backend", BACKENDS)
def test_persona_scoping_still_holds_after_restart(backend, tmp_path):
    _child("write", backend, tmp_path, persona="mentor",
           turns=json.dumps([_turn("user", "My mentor goal is learning Rust", 0)]))
    _child("write", backend, tmp_path, persona="friend",
           turns=json.dumps([_turn("user", "My friend plan is hiking on Sunday", 1)]))

    mentor = _child("query", backend, tmp_path, persona="mentor", query="goal plan", k=10)

    assert [t["text"] for t in mentor["results"]] == ["My mentor goal is learning Rust"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_delete_in_one_process_is_still_deleted_in_the_next(backend, tmp_path):
    _child("write", backend, tmp_path,
           turns=json.dumps([_turn("user", "My locker code is 4512", 0)]))
    _child("delete", backend, tmp_path)

    results = _child("query", backend, tmp_path, query="locker code", k=5)["results"]

    assert results == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_summary_made_in_one_process_replaces_turns_for_the_next(backend, tmp_path):
    """docs/ROLLING_SUMMARIZER.md section 7: write, summarize and read back
    in three separate processes. The summarizer's LLM is a canned fake in
    the child; the store and its files are real."""
    turns = [
        _turn("user", "My sister Maya signed up for the Berlin marathon", 0),
        _turn("assistant", "That's exciting, how is her training going?", 1),
        _turn("user", "She runs every morning before work", 2),
        _turn("assistant", "Impressive discipline.", 3),
        _turn("user", "I might fly over to watch her", 4),
        _turn("assistant", "You should, it would mean a lot.", 5),
    ]
    _child("write", backend, tmp_path, turns=json.dumps(turns))

    assert _child("summarize", backend, tmp_path, trigger=6) == {"ran": True}

    stored = _child("list", backend, tmp_path)["turns"]
    assert stored[0]["role"] == "system"
    assert stored[0]["timestamp"] == turns[2]["timestamp"]
    assert "Berlin marathon" in stored[0]["text"]
    assert stored[1:] == turns[3:]

    results = _child("query", backend, tmp_path, query="Maya Berlin marathon", k=5)["results"]
    assert stored[0] in results

    # A fourth process: nothing is due any more (3 raw turns, summary not counted).
    assert _child("summarize", backend, tmp_path, trigger=6) == {"ran": False}


# --- orchestrator level ---------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_fact_from_previous_session_reaches_the_prompt_after_restart(backend, tmp_path):
    first = _child("chat", backend, tmp_path,
                   message="My sister Maya just moved to Lisbon.")
    assert "Lisbon" not in first["prompt"]  # nothing on disk yet

    second = _child("chat", backend, tmp_path,
                    message="Which city did my sister move to?")

    # New process, empty recent_turns: the only way "Lisbon" can be in this
    # prompt is via MemoryStore.query reading what session one wrote.
    assert "Lisbon" in second["prompt"]
    assert all("Lisbon" not in t["text"] for t in second["context"][:-1])


# --- the real app, real Ollama -------------------------------------------


def _ollama_models() -> list[str]:
    httpx = pytest.importorskip("httpx")
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return []
    return [m["name"] for m in resp.json().get("models", [])]


def _run_app(config_path: Path, stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "companion", "--config", str(config_path),
         "--persona", "tester", "--user-id", "restart-test"],
        cwd=REPO_ROOT, input=stdin, capture_output=True, text=True,
        encoding="utf-8", timeout=CHILD_TIMEOUT_S, check=False,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_real_app_recalls_previous_session_after_restart(backend, tmp_path):
    models = _ollama_models()
    if not models:
        pytest.skip("No local Ollama server with a pulled model at localhost:11434")

    personas_dir = tmp_path / "personas"
    personas_dir.mkdir()
    (personas_dir / "tester.yaml").write_text(yaml.safe_dump({
        "id": "tester",
        "name": "Tester",
        "tone": "Brief and factual. Answer in one short sentence.",
        "boundaries": "None.",
        "voice_id": "piper:unused",
    }), encoding="utf-8")

    config = yaml.safe_load(
        (REPO_ROOT / "companion/config/config.example.yaml").read_text(encoding="utf-8")
    )
    config["mode"] = "text"
    config["personas"]["directory"] = str(personas_dir)
    config["tts"]["voices_dir"] = str(tmp_path)
    config["llm"]["options"].update({"model": models[0], "temperature": 0.0, "seed": 7})
    config["memory"]["backend"] = backend
    config["memory"]["options"] = {"persist_dir": str(tmp_path / "memory")}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    first = _run_app(config_path, "My sister Maya just moved to Lisbon.\nexit\n")
    assert first.returncode == 0, first.stderr

    second = _run_app(
        config_path, "Which city did my sister Maya move to? Answer with the city name.\nexit\n"
    )
    assert second.returncode == 0, second.stderr
    # input()'s "you> " prompt has no newline, so the reply shares its line.
    replies = [line.split("Tester>", 1)[1] for line in second.stdout.splitlines()
               if "Tester>" in line]
    assert replies and "lisbon" in replies[0].lower(), second.stdout


# --- agent tools: notes ----------------------------------------------------


def test_note_saved_in_one_process_is_found_in_the_next(tmp_path):
    saved = _child("add-note", "chroma", tmp_path, persona="mentor",
                   message="Learn Rust this winter")["result"]
    assert saved == "Saved note #1: Learn Rust this winter"

    found = _child("search-notes", "chroma", tmp_path, persona="mentor", query="rust")["result"]
    other_persona = _child("search-notes", "chroma", tmp_path, persona="planner",
                           query="rust")["result"]

    assert "Learn Rust this winter" in found
    assert other_persona == "No matching notes."
