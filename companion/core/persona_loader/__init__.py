"""Loads `companion/personas/*.yaml` into `Persona` objects and validates
them strictly (file + field + reason on any error; unknown fields
rejected). It checks `voice_id` and `tools` only for shape; whether a
voice or tool actually exists is checked later by `companion/app.py`.
See `companion/personas/schema.md`.
"""

from companion.core.persona_loader.persona_loader import (
    PersonaError,
    PersonaLoader,
    load_persona,
)

__all__ = ["PersonaError", "PersonaLoader", "load_persona"]
