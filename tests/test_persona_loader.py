"""Tests for the persona YAML loader (companion/core/persona_loader).

Uses the real starter personas in companion/personas/ for the happy path
(cheap, no fakes needed — it's just YAML parsing), plus tmp_path-based
fixtures for validation failures, so a deliberately malformed persona file
is exercised directly rather than only asserted about in prose.
"""

from pathlib import Path

import pytest
import yaml

from companion.core.interfaces.types import Persona
from companion.core.persona_loader import PersonaError, PersonaLoader, load_persona

REAL_PERSONAS_DIR = Path(__file__).resolve().parents[1] / "companion" / "personas"

VALID_PERSONA = {
    "id": "test-persona",
    "name": "Test Persona",
    "tone": "Friendly.",
    "boundaries": "None.",
    "voice_id": "piper:en_US-test-medium",
    "example_dialogue": ["User: hi\nPersona: hello"],
}


def _write_persona(directory: Path, filename: str, data: dict) -> Path:
    path = directory / filename
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- Happy path, against the real starter personas -------------------------


def test_loads_all_starter_personas():
    loader = PersonaLoader(REAL_PERSONAS_DIR)
    personas = loader.load_all()

    assert set(personas) == {"mentor", "friend", "companion", "study_buddy", "planner"}
    for persona_id, persona in personas.items():
        assert isinstance(persona, Persona)
        assert persona.id == persona_id
        assert persona.name
        assert persona.tone
        assert persona.boundaries
        assert ":" in persona.voice_id
        assert isinstance(persona.example_dialogue, list)


def test_get_looks_up_a_single_persona_by_id():
    loader = PersonaLoader(REAL_PERSONAS_DIR)

    mentor = loader.get("mentor")

    assert mentor.id == "mentor"
    assert mentor.voice_id == "piper:en_US-mentor-medium"


def test_get_unknown_id_raises():
    loader = PersonaLoader(REAL_PERSONAS_DIR)

    with pytest.raises(PersonaError, match="no persona with id"):
        loader.get("does-not-exist")


def test_load_persona_single_file():
    persona = load_persona(REAL_PERSONAS_DIR / "friend.yaml")

    assert persona.id == "friend"
    assert persona.voice_id == "piper:en_US-friend-medium"


# --- Caching -----------------------------------------------------------


def test_get_uses_cache_from_first_load_all(tmp_path):
    _write_persona(tmp_path, "a.yaml", VALID_PERSONA)
    loader = PersonaLoader(tmp_path)
    loader.load_all()

    # A second persona added after the initial scan should NOT be picked
    # up by get() without an explicit reload — load_all() is "load once
    # at startup", not "re-read the directory on every lookup".
    _write_persona(tmp_path, "b.yaml", {**VALID_PERSONA, "id": "second"})

    with pytest.raises(PersonaError, match="no persona with id"):
        loader.get("second")

    assert loader.get("test-persona").id == "test-persona"


# --- Validation failures, via deliberately malformed persona files -----


def test_missing_required_field_fails_loudly(tmp_path):
    data = {k: v for k, v in VALID_PERSONA.items() if k != "boundaries"}
    path = _write_persona(tmp_path, "broken.yaml", data)

    with pytest.raises(PersonaError, match="boundaries") as exc_info:
        load_persona(path)
    assert str(path) in str(exc_info.value)


def test_empty_required_field_fails(tmp_path):
    path = _write_persona(tmp_path, "broken.yaml", {**VALID_PERSONA, "tone": "   "})

    with pytest.raises(PersonaError, match="tone"):
        load_persona(path)


def test_invalid_id_format_rejected(tmp_path):
    path = _write_persona(tmp_path, "broken.yaml", {**VALID_PERSONA, "id": "Not Valid!"})

    with pytest.raises(PersonaError, match="filesystem- and URL-safe"):
        load_persona(path)


def test_malformed_voice_id_missing_namespace_rejected(tmp_path):
    path = _write_persona(
        tmp_path, "broken.yaml", {**VALID_PERSONA, "voice_id": "en_US-test-medium"}
    )

    with pytest.raises(PersonaError, match="voice_id"):
        load_persona(path)


def test_malformed_voice_id_empty_namespace_rejected(tmp_path):
    path = _write_persona(
        tmp_path, "broken.yaml", {**VALID_PERSONA, "voice_id": ":en_US-test-medium"}
    )

    with pytest.raises(PersonaError, match="voice_id"):
        load_persona(path)


def test_unknown_field_rejected(tmp_path):
    path = _write_persona(tmp_path, "broken.yaml", {**VALID_PERSONA, "typo_field": "oops"})

    with pytest.raises(PersonaError, match="typo_field"):
        load_persona(path)


def test_example_dialogue_must_be_list_of_strings(tmp_path):
    path = _write_persona(
        tmp_path, "broken.yaml", {**VALID_PERSONA, "example_dialogue": [{"not": "a string"}]}
    )

    with pytest.raises(PersonaError, match="example_dialogue"):
        load_persona(path)


def test_example_dialogue_defaults_to_empty_list_when_absent(tmp_path):
    data = {k: v for k, v in VALID_PERSONA.items() if k != "example_dialogue"}
    path = _write_persona(tmp_path, "minimal.yaml", data)

    persona = load_persona(path)

    assert persona.example_dialogue == []


def test_non_mapping_yaml_rejected(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(["just", "a", "list"]), encoding="utf-8")

    with pytest.raises(PersonaError, match="mapping"):
        load_persona(path)


def test_invalid_yaml_syntax_rejected(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("id: mentor\n  bad_indent: [unclosed\n", encoding="utf-8")

    with pytest.raises(PersonaError, match="invalid YAML"):
        load_persona(path)


def test_duplicate_id_across_files_rejected(tmp_path):
    _write_persona(tmp_path, "a.yaml", VALID_PERSONA)
    _write_persona(tmp_path, "b.yaml", VALID_PERSONA)  # same id, different file
    loader = PersonaLoader(tmp_path)

    with pytest.raises(PersonaError, match="duplicate persona id"):
        loader.load_all()


# --- tools / filler_phrases (agentic personas) -------------------------------


def test_tools_and_filler_phrases_are_loaded(tmp_path):
    data = {**VALID_PERSONA, "tools": ["get_datetime", "notes"],
            "filler_phrases": ["  One moment.  ", "Let me check."]}
    persona = load_persona(_write_persona(tmp_path, "p.yaml", data))

    assert persona.tools == ["get_datetime", "notes"]
    assert persona.filler_phrases == ["One moment.", "Let me check."]


def test_tools_and_filler_phrases_default_to_empty(tmp_path):
    persona = load_persona(_write_persona(tmp_path, "p.yaml", VALID_PERSONA))

    assert persona.tools == []
    assert persona.filler_phrases == []


@pytest.mark.parametrize("tools", ["notes", [1], [""], ["Notes"], ["web-search"]])
def test_malformed_tools_rejected(tmp_path, tools):
    path = _write_persona(tmp_path, "p.yaml", {**VALID_PERSONA, "tools": tools})

    with pytest.raises(PersonaError, match="tools"):
        load_persona(path)


def test_duplicate_tool_names_rejected(tmp_path):
    path = _write_persona(tmp_path, "p.yaml", {**VALID_PERSONA, "tools": ["notes", "notes"]})

    with pytest.raises(PersonaError, match="more than once"):
        load_persona(path)


def test_filler_phrases_must_be_non_empty_strings(tmp_path):
    path = _write_persona(tmp_path, "p.yaml", {**VALID_PERSONA, "filler_phrases": ["ok", "  "]})

    with pytest.raises(PersonaError, match="filler_phrases"):
        load_persona(path)
