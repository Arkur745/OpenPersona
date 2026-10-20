"""Concrete backend implementations of the five interfaces in
`companion/core/interfaces/`, one subpackage per kind:
`stt/` (faster-whisper), `tts/` (Piper, Coqui), `llm/` (Ollama),
`vad/` (Silero), plus `agent/` (the LangGraph tool-calling runner).

Rules: one file per backend; a new backend also needs an extra in
`pyproject.toml`, a `backend:` value in `config.example.yaml` and a step
in `scripts/check_offline.py`. Only `companion/app.py` chooses between
them by name. Heavy dependencies (torch, model libraries) are imported
lazily inside methods, so importing a backend module never requires its
extra to be installed.
"""
