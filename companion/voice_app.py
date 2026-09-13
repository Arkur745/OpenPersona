"""Voice-mode bootstrap: real mic capture -> `Orchestrator.handle_voice_turn`
-> real speaker playback, wired together for a live voice conversation.
Mirrors `companion/app.py`'s split (testable business logic here, thin
CLI dispatch in `companion/__main__.py`) for the same reason:
`run_voice_loop` takes already-constructed audio I/O objects so its
wiring is testable with mocks instead of requiring real hardware — see
tests/test_voice_app.py, and this module's docstring on `run_voice_loop`
for exactly what that does and doesn't verify.

Backend/persona/`Orchestrator` construction is *not* duplicated here —
`companion/app.py`'s `build_orchestrator`/`load_personas`/
`select_persona` are mode-agnostic (voice mode uses the exact same
STT/TTS/LLM/VAD/MemoryStore construction path as text mode; only the
audio I/O and the loop differ), so `companion/__main__.py` calls those
directly and only reaches into this module for the voice-specific
pieces.

## Sample rate: config.audio must match config.vad, checked here

Silero VAD's model has a hard, non-configurable frame-size requirement
tied to its sample rate (see silero_vad.py's module docstring) — mic
capture must produce audio at exactly that rate or VAD's frame-size math
is simply wrong. `companion/config/loader.py` deliberately doesn't
cross-check option *contents* between sections (see that module's
docstring for why), so this check happens here, before any device is
opened, with a clear, specific message — not a confusing downstream
frame-size `ValueError` from `SileroVAD.is_speech()`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from companion.audio.capture import MicCapture
from companion.audio.playback import SpeakerPlayback
from companion.config.loader import AppConfig
from companion.core.interfaces.types import Persona
from companion.core.orchestrator import Orchestrator


class VoiceSetupError(RuntimeError):
    """Something about constructing voice-mode audio I/O from config
    failed in a way that isn't already `AudioDeviceError` (a real device
    problem raised by companion/audio/) — e.g. a sample-rate mismatch
    between `audio` and `vad` config sections, or an unimplemented
    `audio.backend`."""


def _matched_sample_rate(config: AppConfig) -> int:
    audio_options: dict[str, Any] = config.audio.options
    vad_options: dict[str, Any] = config.vad.options
    audio_rate = audio_options.get("sample_rate")
    vad_rate = vad_options.get("sample_rate", 16000)  # SileroVAD's own default
    if audio_rate is None:
        raise VoiceSetupError("audio.options.sample_rate is required for voice mode.")
    if audio_rate != vad_rate:
        raise VoiceSetupError(
            f"audio.options.sample_rate ({audio_rate}) must match "
            f"vad.options.sample_rate ({vad_rate}) — Silero VAD's frame "
            "size is a hard function of this rate (see silero_vad.py's "
            "module docstring). Fix one of the two in your config."
        )
    return audio_rate


def build_mic_capture(config: AppConfig) -> MicCapture:
    if config.audio.backend != "sounddevice":
        raise VoiceSetupError(
            f"audio.backend {config.audio.backend!r} is not implemented — "
            f"only 'sounddevice' is."
        )
    sample_rate = _matched_sample_rate(config)
    options = config.audio.options
    return MicCapture(
        sample_rate=sample_rate,
        device=options.get("input_device"),
        channels=options.get("channels", 1),
    )


def build_speaker_playback(config: AppConfig) -> SpeakerPlayback:
    if config.audio.backend != "sounddevice":
        raise VoiceSetupError(
            f"audio.backend {config.audio.backend!r} is not implemented — "
            f"only 'sounddevice' is."
        )
    sample_rate = _matched_sample_rate(config)
    options = config.audio.options
    return SpeakerPlayback(
        sample_rate=sample_rate,
        device=options.get("output_device"),
        channels=options.get("channels", 1),
    )


async def run_voice_loop(
    orchestrator: Orchestrator,
    persona: Persona,
    user_id: str,
    mic: MicCapture,
    playback: SpeakerPlayback,
    *,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Drives one live voice session: real mic audio in, through
    `Orchestrator.handle_voice_turn`, real speaker playback out —
    wiring `playback.stop`/`playback.wait_drained` into
    `handle_voice_turn`'s `on_playback_interrupt`/
    `wait_for_playback_drained` hooks so barge-in detection stays
    accurate through real playback latency, not just synthesis latency
    (see orchestrator.py's `handle_voice_turn` docstring and
    `companion/audio/playback.py` for the full mechanism).

    Runs until the caller cancels this coroutine (e.g. Ctrl+C) — a live
    mic has no natural end, unlike `handle_voice_turn`'s test fixtures.

    `mic`/`playback` are accepted pre-constructed (not built internally)
    specifically so this function's *wiring* — hooking playback's
    stop/wait_drained into the right handle_voice_turn parameters, and
    actually calling `playback.play()` on every yielded chunk — is
    testable with a mock/fake `MicCapture`/`SpeakerPlayback` instead of
    requiring real hardware. What this does *not* verify, automated or
    otherwise: that real audio sounds correct or intelligible — see
    tests/test_voice_app.py's module docstring and this project's PR/
    session notes for what was manually verified against real hardware
    instead.
    """
    print_fn(f"Talking to {persona.name} ({persona.id}). Press Ctrl+C to quit.\n")
    async for chunk in orchestrator.handle_voice_turn(
        user_id,
        persona,
        mic,
        on_playback_interrupt=playback.stop,
        wait_for_playback_drained=playback.wait_drained,
    ):
        await playback.play(chunk)


async def run_voice_session(
    config: AppConfig,
    orchestrator: Orchestrator,
    persona: Persona,
    user_id: str,
    *,
    print_fn: Callable[[str], None] = print,
) -> None:
    """Top-level voice-mode entrypoint: construct real mic/speaker I/O
    from config, run the session, and guarantee both devices are closed
    afterward (including if construction or the session itself fails
    partway through)."""
    mic = build_mic_capture(config)
    playback: SpeakerPlayback | None = None
    try:
        playback = build_speaker_playback(config)
        await run_voice_loop(orchestrator, persona, user_id, mic, playback, print_fn=print_fn)
    finally:
        mic.close()
        if playback is not None:
            playback.close()
