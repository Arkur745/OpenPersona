"""Loads and validates companion/config/*.yaml (app-level backend
selection + persona directory + runtime mode) into `AppConfig`.

Companion piece to `companion/core/persona_loader/` — same "fail loudly
and specifically" posture (name the file, the section, and what was
wrong, not a bare YAML/KeyError traceback), but a deliberately *narrower*
validation scope. A persona file is a small, closed, fully-specified
community format, so rejecting unknown fields there catches real typos.
This top-level config's `options:` blocks are inherently backend-defined
and meant to grow — a new backend needs new option keys without ever
touching this loader (per config.example.yaml's own header: "every
options: block is passed as keyword args to that backend's
constructor"). So this loader validates *structure* (the sections and
sub-keys the app itself depends on to construct backends: `backend`,
`options`, `personas.directory`, `tts.voices_dir`, `mode`; `audio` is
structurally required too, even for a text-mode-only config — unlike
stt/tts/llm/vad/memory it isn't constructed by `companion/app.py`'s
`build_orchestrator` at all, only by `companion/voice_app.py`, but
`load_config` doesn't know or care which mode a given config is for) but
does not
reject unrecognized keys inside an `options:` block. The one nested key it
does parse is `llm.voice_mode_override` (into `AppConfig.llm_voice_override`;
`companion.app.build_llm` applies it in voice mode). Whether a given `options:` entry is actually valid for its backend
is enforced for free by that backend's own constructor (a bad key raises
`TypeError`) — companion/app.py is responsible for that failure surfacing
as an actionable message, not this loader re-validating it up front.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VALID_MODES = ("text", "voice")


class ConfigError(ValueError):
    """A config YAML file failed to load or validate. Always carries the
    offending file path and the specific section/field at fault."""


@dataclass(frozen=True)
class BackendConfig:
    """`{backend: str, options: dict}` — the shape shared by stt/llm/vad/
    memory. `options` is passed straight through as constructor kwargs to
    whichever concrete class `backend` names."""

    backend: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSConfig:
    """Like `BackendConfig`, plus `voices_dir` — TTS is the one category
    where persona `voice_id` resolution needs a directory to resolve
    against (see `PiperTTS.from_voice_id`/`CoquiTTS.from_voice_id`)."""

    backend: str
    voices_dir: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AppConfig:
    personas_dir: str
    stt: BackendConfig
    tts: TTSConfig
    llm: BackendConfig
    vad: BackendConfig
    memory: BackendConfig
    audio: BackendConfig
    mode: str
    # `llm.voice_mode_override.options`, merged over `llm.options` by
    # `companion.app.build_llm` when `mode == "voice"`. Empty = no override.
    llm_voice_override: dict[str, Any] = field(default_factory=dict)
    # Optional `agent:` section (agentic personas). None = no agent: every
    # persona uses the plain LLM, tools or not.
    agent: BackendConfig | None = None


def _require_section(data: dict, key: str, path: Path) -> dict:
    if key not in data:
        raise ConfigError(f"{path}: missing required top-level section {key!r}")
    section = data[key]
    if not isinstance(section, dict):
        raise ConfigError(
            f"{path}: section {key!r} must be a mapping, got {type(section).__name__}"
        )
    return section


def _require_str(section: dict, key: str, section_name: str, path: Path) -> str:
    if key not in section:
        raise ConfigError(f"{path}: {section_name}.{key} is required")
    value = section[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"{path}: {section_name}.{key} must be a non-empty string, got {value!r}"
        )
    return value


def _options(section: dict, section_name: str, path: Path) -> dict[str, Any]:
    options = section.get("options", {})
    if not isinstance(options, dict):
        raise ConfigError(
            f"{path}: {section_name}.options must be a mapping, got "
            f"{type(options).__name__}"
        )
    return options


def _backend_config(data: dict, section_name: str, path: Path) -> BackendConfig:
    section = _require_section(data, section_name, path)
    backend = _require_str(section, "backend", section_name, path)
    return BackendConfig(backend=backend, options=_options(section, section_name, path))


def _llm_voice_override(data: dict, path: Path) -> dict[str, Any]:
    """`llm.voice_mode_override` may only override `options` — swapping
    the LLM *backend* by mode isn't supported, so any other key is
    rejected rather than silently ignored."""
    override = data["llm"].get("voice_mode_override")
    if override is None:
        return {}
    if not isinstance(override, dict):
        raise ConfigError(
            f"{path}: llm.voice_mode_override must be a mapping, got {type(override).__name__}"
        )
    unknown = set(override) - {"options"}
    if unknown:
        raise ConfigError(
            f"{path}: llm.voice_mode_override only supports 'options' "
            f"(got {sorted(unknown)}); the LLM backend itself can't differ by mode"
        )
    return _options(override, "llm.voice_mode_override", path)


def load_config(path: str | Path) -> AppConfig:
    """Parse and validate one config YAML file (`config.yaml` or
    `config.example.yaml`) into `AppConfig`."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"{path}: config file not found") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML — {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: config file must contain a YAML mapping at the top "
            f"level, got {type(data).__name__}"
        )

    personas_section = _require_section(data, "personas", path)
    personas_dir = _require_str(personas_section, "directory", "personas", path)

    stt = _backend_config(data, "stt", path)
    llm = _backend_config(data, "llm", path)
    vad = _backend_config(data, "vad", path)
    memory = _backend_config(data, "memory", path)
    audio = _backend_config(data, "audio", path)

    tts_section = _require_section(data, "tts", path)
    tts = TTSConfig(
        backend=_require_str(tts_section, "backend", "tts", path),
        voices_dir=_require_str(tts_section, "voices_dir", "tts", path),
        options=_options(tts_section, "tts", path),
    )

    mode = data.get("mode")
    if mode not in _VALID_MODES:
        raise ConfigError(f"{path}: 'mode' must be one of {_VALID_MODES}, got {mode!r}")

    return AppConfig(
        personas_dir=personas_dir,
        stt=stt,
        tts=tts,
        llm=llm,
        vad=vad,
        memory=memory,
        audio=audio,
        mode=mode,
        llm_voice_override=_llm_voice_override(data, path),
        agent=_backend_config(data, "agent", path) if data.get("agent") is not None else None,
    )


def resolve_config_path(
    explicit: str | Path | None, base_dir: str | Path | None = None
) -> Path:
    """Decide which config file to load.

    An explicitly given path must exist. Otherwise: `config.yaml` in
    `base_dir` if the user has copied and edited one (it's git-ignored —
    see config.example.yaml's own header comment), else
    `config.example.yaml` itself, so a first-time clone with zero setup
    still has something runnable to load — see README.md's "Running the
    app" section. `base_dir` defaults to this module's own directory
    (`companion/config/`); overridable so this fallback logic is testable
    against a throwaway directory instead of the real repo config dir.
    """
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            raise ConfigError(f"{explicit}: config file not found")
        return explicit

    config_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    user_config = config_dir / "config.yaml"
    if user_config.exists():
        return user_config
    return config_dir / "config.example.yaml"
