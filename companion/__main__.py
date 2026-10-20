"""CLI entrypoint: `python -m companion`.

Reads config, builds a real `Orchestrator` + selected persona, and runs
either the text-mode loop (`companion/app.py`'s `run_text_loop`) or the
voice-mode loop (`companion/voice_app.py`'s `run_voice_session`,
real mic capture + real speaker playback), dispatched on `config.mode` —
`config.yaml` is the one place backend/mode selection happens, per
config.example.yaml's own header, so there's no separate `--mode` CLI
flag duplicating that switch. See README.md's "Running the app" section
for the exact commands for either mode.

Deliberately thin: argument parsing, path resolution, and top-level
error/interrupt presentation only. Everything else (config validation,
backend construction, both loops themselves) lives in `companion/app.py`
and `companion/voice_app.py`, where it's directly testable without going
through argparse, real stdin, or real audio hardware.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from companion.app import (
    build_orchestrator,
    load_personas,
    run_text_loop,
    select_persona,
    take_due_reminders,
)
from companion.config.loader import load_config, resolve_config_path
from companion.voice_app import run_voice_session


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m companion",
        description="Local-first AI companion — text or voice conversation loop.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a config YAML file (default: companion/config/config.yaml "
            "if it exists, else companion/config/config.example.yaml). Its "
            "'mode: text|voice' selects which loop runs."
        ),
    )
    parser.add_argument(
        "--persona",
        default=None,
        help="Persona id to talk to (default: the first persona found).",
    )
    parser.add_argument(
        "--user-id",
        default="local",
        help=(
            "Memory-scoping user id (default: 'local' — this is a single-user "
            "local app; override to test separate memory profiles)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Show a full traceback instead of a one-line error message, and "
            "log each step (in voice mode: listening, heard, responding, "
            "speaking, playback finished) to stderr."
        ),
    )
    return parser.parse_args(argv)


def _configure_logging(debug: bool) -> None:
    """Warnings from this app always show (e.g. a stalled speaker). With
    --debug, every step does too. Third-party libraries stay at WARNING
    either way, since they're very chatty at DEBUG."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("companion").setLevel(logging.DEBUG if debug else logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.debug)
    try:
        config_path = resolve_config_path(args.config)
        config = load_config(config_path)

        personas = load_personas(config)
        persona = select_persona(personas, args.persona)
        orchestrator = build_orchestrator(config, persona)

        check_reminders = None
        if config.agent is not None:
            def check_reminders():
                return take_due_reminders(config, args.user_id, personas)

        if config.mode == "voice":
            asyncio.run(run_voice_session(config, orchestrator, persona, args.user_id,
                                          check_reminders=check_reminders))
        else:
            asyncio.run(run_text_loop(orchestrator, persona, args.user_id,
                                      check_reminders=check_reminders))
        return 0
    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
