"""Loads and validates persona YAML configs (companion/personas/*.yaml)
into companion.core.interfaces.types.Persona.

See companion/personas/schema.md for the on-disk format this validates
against. Validation deliberately fails loudly and specifically (naming the
file, the field, and what was wrong) rather than surfacing a bare YAML
parse error or silently ignoring a typo'd/missing field — a malformed
community-contributed persona should be easy to diagnose without reading
this module's source.

Namespace validation for `voice_id` (e.g. confirming "piper:..." is a
namespace a given deployment's TTS backend actually recognizes) is
deliberately NOT done here — that's each backend's `from_voice_id()`
classmethod's job (see piper_local.py / coqui_local.py), so this loader
never has to branch on backend name. This loader only validates the
shape ("<namespace>:<value>", both non-empty).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from companion.core.interfaces.types import Persona

_ID_RE = re.compile(r"^[a-z0-9_-]+$")
_REQUIRED_STRING_FIELDS = ("id", "name", "tone", "boundaries", "voice_id")
_KNOWN_FIELDS = set(_REQUIRED_STRING_FIELDS) | {"example_dialogue", "tools", "filler_phrases"}
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PersonaError(ValueError):
    """A persona YAML file failed to load or validate. Always carries the
    offending file path so a malformed persona is easy to trace back to
    its source file."""


def _validate_and_build(data: object, path: Path) -> Persona:
    if not isinstance(data, dict):
        raise PersonaError(
            f"{path}: persona file must contain a YAML mapping at the top "
            f"level, got {type(data).__name__}"
        )

    for field_name in _REQUIRED_STRING_FIELDS:
        if field_name not in data:
            raise PersonaError(f"{path}: missing required field {field_name!r}")
        value = data[field_name]
        if not isinstance(value, str) or not value.strip():
            raise PersonaError(
                f"{path}: field {field_name!r} must be a non-empty string, "
                f"got {value!r}"
            )

    unknown = set(data) - _KNOWN_FIELDS
    if unknown:
        raise PersonaError(
            f"{path}: unknown field(s) {sorted(unknown)} — see "
            f"companion/personas/schema.md for the accepted shape"
        )

    persona_id = data["id"]
    if not _ID_RE.match(persona_id):
        raise PersonaError(
            f"{path}: id {persona_id!r} must be filesystem- and URL-safe "
            f"(lowercase letters, digits, '_', '-' only) — see "
            f"companion/personas/schema.md"
        )

    voice_id = data["voice_id"]
    namespace, sep, voice_name = voice_id.partition(":")
    if not sep or not namespace or not voice_name:
        raise PersonaError(
            f"{path}: voice_id {voice_id!r} must be namespaced as "
            f"'<tts-backend>:<voice>' (e.g. 'piper:en_US-lessac-medium') — "
            f"see companion/personas/schema.md"
        )

    example_dialogue = data.get("example_dialogue", [])
    if not isinstance(example_dialogue, list) or not all(
        isinstance(entry, str) for entry in example_dialogue
    ):
        raise PersonaError(
            f"{path}: 'example_dialogue' must be a list of strings, got "
            f"{example_dialogue!r}"
        )

    tools = _string_list(data, "tools", path)
    bad_names = [name for name in tools if not _TOOL_NAME_RE.match(name)]
    if bad_names:
        raise PersonaError(
            f"{path}: 'tools' entries must be tool names like 'get_datetime' "
            f"(lowercase letters, digits, '_'), got {bad_names!r}"
        )
    duplicates = sorted({name for name in tools if tools.count(name) > 1})
    if duplicates:
        raise PersonaError(f"{path}: 'tools' lists {duplicates!r} more than once")

    return Persona(
        id=persona_id,
        name=data["name"],
        tone=data["tone"],
        boundaries=data["boundaries"],
        voice_id=voice_id,
        example_dialogue=list(example_dialogue),
        tools=tools,
        filler_phrases=_string_list(data, "filler_phrases", path),
    )


def _string_list(data: dict, field_name: str, path: Path) -> list[str]:
    """An optional list of non-empty strings; absent means empty."""
    value = data.get(field_name, [])
    if not isinstance(value, list) or not all(
        isinstance(entry, str) and entry.strip() for entry in value
    ):
        raise PersonaError(
            f"{path}: {field_name!r} must be a list of non-empty strings, got {value!r}"
        )
    return [entry.strip() for entry in value]


def load_persona(path: str | Path) -> Persona:
    """Parse and validate a single persona YAML file."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise PersonaError(f"{path}: invalid YAML — {exc}") from exc
    return _validate_and_build(data, path)


class PersonaLoader:
    """Loads every persona in a directory once, then serves id-based
    lookups from that cache — matching how a persona directory is meant to
    be used (scanned at startup, not re-read per message)."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._cache: dict[str, Persona] | None = None

    def load_all(self) -> dict[str, Persona]:
        """(Re)scan the directory and cache the result, keyed by id."""
        personas: dict[str, Persona] = {}
        for path in sorted(self.directory.glob("*.yaml")):
            persona = load_persona(path)
            if persona.id in personas:
                raise PersonaError(
                    f"{path}: duplicate persona id {persona.id!r} — already "
                    f"defined by another file in {self.directory}"
                )
            personas[persona.id] = persona
        self._cache = personas
        return personas

    def get(self, persona_id: str) -> Persona:
        personas = self._cache if self._cache is not None else self.load_all()
        try:
            return personas[persona_id]
        except KeyError:
            raise PersonaError(
                f"no persona with id {persona_id!r} found in {self.directory}"
            ) from None
