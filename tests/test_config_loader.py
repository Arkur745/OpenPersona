"""Tests for companion/config/loader.py.

Mirrors tests/test_persona_loader.py's style: happy path against the real
repo config file, deliberately malformed tmp_path fixtures for every
validation rule. See loader.py's own module docstring for why validation
scope here is deliberately narrower than persona validation (structure,
not `options:` contents).
"""

from pathlib import Path

import pytest
import yaml

from companion.config.loader import (
    AppConfig,
    ConfigError,
    load_config,
    resolve_config_path,
)

REAL_CONFIG_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "companion" / "config" / "config.example.yaml"
)

VALID_CONFIG = {
    "personas": {"directory": "./companion/personas"},
    "stt": {"backend": "faster_whisper", "options": {"model_size": "small"}},
    "tts": {
        "backend": "piper",
        "voices_dir": "./data/voices",
        "options": {"voice_model_path": "./data/voices/x.onnx"},
    },
    "llm": {"backend": "ollama", "options": {"model": "qwen2.5:7b-instruct"}},
    "vad": {"backend": "silero", "options": {"threshold": 0.5}},
    "memory": {"backend": "chroma", "options": {"persist_dir": "./data/memory"}},
    "audio": {"backend": "sounddevice", "options": {"sample_rate": 16000}},
    "mode": "text",
}


def _write_config(directory: Path, data: dict) -> Path:
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --- Happy path ------------------------------------------------------------


def test_loads_the_real_config_example_yaml():
    config = load_config(REAL_CONFIG_EXAMPLE)

    assert isinstance(config, AppConfig)
    assert config.personas_dir == "./companion/personas"
    assert config.stt.backend == "faster_whisper"
    assert config.tts.backend == "piper"
    assert config.tts.voices_dir == "./data/voices"
    assert config.llm.backend == "ollama"
    assert config.llm.options["model"] == "qwen2.5:7b-instruct"
    assert config.vad.backend == "silero"
    assert config.memory.backend == "chroma"
    assert config.audio.backend == "sounddevice"
    assert config.audio.options["sample_rate"] == 16000
    assert config.mode == "text"
    assert config.llm_voice_override == {"model": "qwen2.5:3b-instruct"}


def test_valid_config_parses_all_sections(tmp_path):
    path = _write_config(tmp_path, VALID_CONFIG)

    config = load_config(path)

    assert config.stt.options == {"model_size": "small"}
    assert config.tts.options == {"voice_model_path": "./data/voices/x.onnx"}
    assert config.vad.options == {"threshold": 0.5}
    assert config.mode == "text"


def test_options_defaults_to_empty_dict_when_absent(tmp_path):
    data = {**VALID_CONFIG, "vad": {"backend": "silero"}}  # no options block
    path = _write_config(tmp_path, data)

    config = load_config(path)

    assert config.vad.options == {}


# --- Validation failures -----------------------------------------------


@pytest.mark.parametrize("section", ["personas", "stt", "tts", "llm", "vad", "memory", "audio"])
def test_missing_required_section_fails_loudly(tmp_path, section):
    data = {k: v for k, v in VALID_CONFIG.items() if k != section}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match=section):
        load_config(path)


def test_missing_backend_field_fails_loudly(tmp_path):
    data = {**VALID_CONFIG, "llm": {"options": {"model": "x"}}}  # no backend
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="llm.backend"):
        load_config(path)


def test_empty_backend_field_rejected(tmp_path):
    data = {**VALID_CONFIG, "stt": {"backend": "  ", "options": {}}}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="stt.backend"):
        load_config(path)


def test_options_not_a_mapping_rejected(tmp_path):
    data = {**VALID_CONFIG, "llm": {"backend": "ollama", "options": "not-a-dict"}}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="llm.options"):
        load_config(path)


def test_tts_missing_voices_dir_rejected(tmp_path):
    data = {**VALID_CONFIG, "tts": {"backend": "piper", "options": {}}}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="tts.voices_dir"):
        load_config(path)


def test_personas_missing_directory_rejected(tmp_path):
    data = {**VALID_CONFIG, "personas": {}}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="personas.directory"):
        load_config(path)


def test_mode_missing_rejected(tmp_path):
    data = {k: v for k, v in VALID_CONFIG.items() if k != "mode"}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="mode"):
        load_config(path)


def test_mode_invalid_value_rejected(tmp_path):
    data = {**VALID_CONFIG, "mode": "video-call"}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="mode"):
        load_config(path)


def test_non_mapping_yaml_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(["just", "a", "list"]), encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_invalid_yaml_syntax_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("stt:\n  bad_indent: [unclosed\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_section_not_a_mapping_rejected(tmp_path):
    data = {**VALID_CONFIG, "stt": "not-a-mapping"}
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="stt"):
        load_config(path)


# --- llm.voice_mode_override ----------------------------------------------


def _with_llm(llm_section: dict) -> dict:
    return {**VALID_CONFIG, "llm": llm_section}


def test_voice_override_defaults_to_empty_when_absent(tmp_path):
    config = load_config(_write_config(tmp_path, VALID_CONFIG))

    assert config.llm_voice_override == {}


def test_voice_override_options_are_parsed(tmp_path):
    data = _with_llm({
        "backend": "ollama",
        "options": {"model": "big"},
        "voice_mode_override": {"options": {"model": "small", "temperature": 0.2}},
    })

    config = load_config(_write_config(tmp_path, data))

    assert config.llm_voice_override == {"model": "small", "temperature": 0.2}
    assert config.llm.options == {"model": "big"}


def test_voice_override_rejects_backend_switching(tmp_path):
    data = _with_llm({
        "backend": "ollama",
        "voice_mode_override": {"backend": "other", "options": {}},
    })

    with pytest.raises(ConfigError, match="voice_mode_override only supports 'options'"):
        load_config(_write_config(tmp_path, data))


def test_voice_override_must_be_a_mapping(tmp_path):
    data = _with_llm({"backend": "ollama", "voice_mode_override": "small"})

    with pytest.raises(ConfigError, match="voice_mode_override must be a mapping"):
        load_config(_write_config(tmp_path, data))


# --- resolve_config_path -------------------------------------------------


def test_resolve_explicit_path_that_exists(tmp_path):
    path = _write_config(tmp_path, VALID_CONFIG)

    assert resolve_config_path(str(path)) == path


def test_resolve_explicit_path_that_is_missing_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        resolve_config_path(str(tmp_path / "nope.yaml"))


def test_resolve_prefers_user_config_yaml_over_example(tmp_path):
    (tmp_path / "config.example.yaml").write_text("placeholder", encoding="utf-8")
    user_config = tmp_path / "config.yaml"
    user_config.write_text("placeholder", encoding="utf-8")

    assert resolve_config_path(None, base_dir=tmp_path) == user_config


def test_resolve_falls_back_to_example_when_no_user_config(tmp_path):
    example = tmp_path / "config.example.yaml"
    example.write_text("placeholder", encoding="utf-8")

    assert resolve_config_path(None, base_dir=tmp_path) == example


# --- agent section ----------------------------------------------------------


def test_agent_section_is_optional(tmp_path):
    assert load_config(_write_config(tmp_path, VALID_CONFIG)).agent is None


def test_agent_section_is_parsed(tmp_path):
    data = {**VALID_CONFIG, "agent": {"backend": "langgraph", "options": {"max_steps": 3}}}

    agent = load_config(_write_config(tmp_path, data)).agent

    assert (agent.backend, agent.options) == ("langgraph", {"max_steps": 3})


def test_real_config_example_enables_the_agent_offline():
    agent = load_config(REAL_CONFIG_EXAMPLE).agent

    assert agent.backend == "langgraph"
    assert agent.options["model"] == "qwen2.5:7b-instruct"
    assert agent.options["allow_network"] is False
