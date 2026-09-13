# Contributing

## Setup

```
pip install -e ".[dev]"
# add whichever backend extras you're working on, e.g.:
pip install -e ".[stt-whisper,tts-piper,llm-ollama,vad-silero,memory-chroma]"
cp companion/config/config.example.yaml companion/config/config.yaml
```

Ollama itself is a separate install (not a pip package) — see
[ollama.com](https://ollama.com), then `ollama pull <model>` for whatever
`config.yaml`'s `llm.options.model` names.

Once that's done, `python -m companion` runs a real text-mode
conversation — see README.md's "Running the app" for details and useful
flags (`--persona`, `--user-id`, `--config`, `--debug`). Add
`audio-sounddevice` (plus `stt-whisper`/`tts-piper`/`vad-silero` if not
already installed) and set `mode: voice` in `config.yaml` for voice
mode — same README section covers the extra setup that needs (a real
voice file for whichever persona you pick; the starter personas'
`voice_id`s are placeholders).

## Where things go

Read `docs/ARCHITECTURE.md` first. In short:

- Adding/changing behavior for **one backend** (a specific STT/TTS/LLM/VAD
  implementation): edit only the matching file under `companion/backends/`.
  Never change `companion/core/orchestrator/` or `companion/core/prompt_builder/`
  to special-case a backend — if you need that, the interface in
  `companion/core/interfaces/` is missing something and should be extended
  for all implementations, not worked around for one.
- Adding a **new backend** (e.g. a different STT engine): implement the
  relevant interface in a new file under `companion/backends/<kind>/`, add
  an optional-dependency group in `pyproject.toml`, and document the new
  `backend:` value in `companion/config/config.example.yaml`. Don't touch
  the orchestrator.
- Adding a **persona**: see `docs/PERSONA_AUTHORING.md` — this should never
  require a code change.
- Changing **how the app is bootstrapped or run** (config loading,
  backend construction, either CLI loop): `companion/config/loader.py`,
  `companion/app.py`/`companion/__main__.py` (shared + text mode), and
  `companion/voice_app.py` (voice mode) — see `docs/ARCHITECTURE.md`'s
  "App bootstrap" and "Voice mode audio I/O" sections. This is also the
  one place branching on a `backend:` name by string is expected, per
  those sections.
- Changing **real mic/speaker I/O**: `companion/audio/` (`capture.py`,
  `playback.py`) — not a `core/interfaces/` ABC (there's no swappable
  "audio backend" family the way STT/TTS/LLM/VAD/MemoryStore have one),
  so this is plain infrastructure, not a pattern to extend for a
  hypothetical second audio library unless one actually gets added.
- `legacy/` is an archived prior project (Node/React), kept for reference
  only. Do not add new code there and do not assume anything in it reflects
  current architecture.

## Style

- Python 3.11+, type hints on all public functions/methods.
- `ruff` for lint/format (installed via the `dev` extra).
- Interfaces in `companion/core/interfaces/` are the contract — implementations
  must satisfy the abstract methods exactly (signature and semantics as
  documented in the docstring), not "close enough."

## Tests

```
pytest
```

`tests/` mirrors the `companion/` package layout. Interface conformance
tests (does a given backend implementation honor its interface's contract)
are especially valuable — prefer those over tests that assume one specific
backend is installed, so CI doesn't require every optional dependency.
