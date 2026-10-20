"""Config loading. `load_config` parses `config.yaml` (or
`config.example.yaml`) into a frozen `AppConfig`, validating structure
only: unknown keys inside `options:` pass through to backend
constructors. Optional sections: `agent:`, `memory.summarizer`,
`llm.voice_mode_override`. `resolve_config_path` picks the file the app
uses. `config.example.yaml` documents every setting.
"""

from companion.config.loader import (
    AppConfig,
    BackendConfig,
    ConfigError,
    TTSConfig,
    load_config,
    resolve_config_path,
)

__all__ = [
    "AppConfig",
    "BackendConfig",
    "ConfigError",
    "TTSConfig",
    "load_config",
    "resolve_config_path",
]
