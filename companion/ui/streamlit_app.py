"""Streamlit chat UI for companion — the first real UI.

`companion/ui/README.md` deliberately left the UI framework undecided
until a real one was picked; this is that decision (Streamlit), not the
only answer for every future UI need — a fast, local-first-friendly way
to get a real chat window running against the same `Orchestrator`/
`build_*` functions `companion/app.py`'s CLI text loop already uses, not
a second, parallel app-construction path.

**Push-to-talk voice, not the CLI's real-time hands-free voice mode.**
`handle_voice_turn` needs a continuous mic-audio source and speaker
playback sink built around a persistent asyncio event loop and a
real-time barge-in state machine — none of which fits Streamlit's
per-interaction script-rerun model without substantially more plumbing
than this UI needs. Instead: `st.audio_input` records one clip in the
browser, it's transcribed with the same `stt: SpeechToText` the CLI uses
(one-shot, not streaming — see `_transcribe_clip`), the transcript goes
through the exact same `Orchestrator.handle_text_turn` path a typed
message does, and (if "Speak replies" is on) the reply is synthesized
with the same `tts: TextToSpeech` and played back via `st.audio`. No new
abstract-interface surface — this composes the existing STT/TTS/
Orchestrator contracts from UI code, the same way `companion/app.py` and
`companion/voice_app.py` do.

Install: `pip install -e ".[dev,memory-chroma,ui-streamlit]"` for text
only; add `stt-whisper,tts-piper` for voice (swap `memory-chroma` for
`memory-memark` if that's your configured backend). Voice input works
the moment `stt-whisper` is installed; hearing spoken replies also needs
a real Piper voice file for your persona (see README.md's "Voice mode"
section — the starter personas' `voice_id`s are placeholders) — without
one, "Speak replies" surfaces a clear inline error per attempt, not a
crash.

Run: `streamlit run companion/ui/streamlit_app.py`

Uses whichever `config.yaml` (or `config.example.yaml` if you haven't
copied one) `companion/config/loader.py`'s `resolve_config_path` would
pick for a plain `python -m companion` run — same config, same personas,
same memory backend, no separate UI-only config surface.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave

import numpy as np
import streamlit as st

from companion.app import (
    REMINDER_POLL_S,
    AppSetupError,
    build_agent,
    build_llm,
    build_memory,
    build_stt,
    build_summarizer,
    build_tts_for_persona,
    build_vad,
    load_personas,
    select_persona,
    take_due_reminders,
)
from companion.config.loader import (
    AppConfig,
    ConfigError,
    load_config,
    resolve_config_path,
)
from companion.core.agent import AgentEvent, ToolFinished, ToolStarted
from companion.core.interfaces.types import Persona, Turn
from companion.core.orchestrator import Orchestrator
from companion.core.persona_loader import PersonaError
from companion.core.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

_STT_SAMPLE_RATE = 16_000  # must match companion/backends/stt/whisper_local.py's SAMPLE_RATE_HZ

st.set_page_config(page_title="Companion", page_icon="💬")


@st.cache_resource
def _load_config() -> AppConfig:
    return load_config(resolve_config_path(None))


@st.cache_resource
def _load_personas(_config: AppConfig) -> dict[str, Persona]:
    # Leading underscore on the param tells Streamlit not to try to hash
    # AppConfig for the cache key (it holds plain dicts in its `options`
    # fields, which aren't hashable) — this becomes a process-wide
    # singleton, same as _load_config, which is what's wanted: personas
    # don't change without a server restart.
    return load_personas(_config)


@st.cache_resource
def _shared_backends(_config: AppConfig) -> dict[str, object]:
    # Everything except TTS, which is the one persona-bound piece (see
    # docs/ARCHITECTURE.md's "Persona voice binding" note) — built once
    # and reused across persona switches in the sidebar, so switching
    # personas doesn't reconstruct/reload STT/LLM/VAD/Memory needlessly.
    memory = build_memory(_config)
    return {
        "stt": build_stt(_config),
        "llm": build_llm(_config),
        "vad": build_vad(_config),
        "memory": memory,
        # None without an `agent:` config section; then every persona
        # chats through the plain LLM.
        "agent": build_agent(_config, memory),
        # None when memory.summarizer is off. Not handed to Orchestrator:
        # each turn here runs under its own asyncio.run, which would cancel
        # a background task the moment the turn returns. _run_turn calls it
        # itself after the reply has rendered (docs/ROLLING_SUMMARIZER.md
        # section 5).
        "summarizer": build_summarizer(_config, memory),
    }


def _build_orchestrator(config: AppConfig, persona: Persona) -> Orchestrator:
    shared = _shared_backends(config)
    return Orchestrator(
        stt=shared["stt"],
        tts=build_tts_for_persona(config, persona),
        llm=shared["llm"],
        vad=shared["vad"],
        memory=shared["memory"],
        prompt_builder=PromptBuilder(),
        agent=shared["agent"],
    )


def _wav_bytes_to_pcm16_mono(wav_bytes: bytes, target_rate: int = _STT_SAMPLE_RATE) -> bytes:
    """Decodes a browser-recorded WAV blob (`st.audio_input`'s return
    value) into raw, headerless PCM16LE mono at `target_rate` —
    `FasterWhisperSTT.transcribe_stream`'s exact required input shape
    (`SAMPLE_RATE_HZ` in whisper_local.py). `st.audio_input(sample_rate=
    16000)` below already asks the browser to record at that rate, but
    this still reads the real WAV header and converts defensively rather
    than assuming the hint was honored exactly — verified by actually
    round-tripping real Piper-synthesized audio through this exact
    function (resampled down from Piper's 22050Hz) and back through real
    FasterWhisperSTT, getting back the original sentence verbatim."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        src_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"unsupported WAV sample width: {sample_width * 8}-bit "
            "(expected 16-bit — browser mic recordings always are in "
            "practice, so this means something upstream changed)"
        )

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if src_rate != target_rate:
        x_src = np.linspace(0, 1, num=len(samples), endpoint=False)
        x_dst = np.linspace(0, 1, num=int(len(samples) * target_rate / src_rate), endpoint=False)
        samples = np.interp(x_dst, x_src, samples)

    return samples.astype(np.int16).tobytes()


async def _transcribe_clip(orchestrator: Orchestrator, pcm16_mono: bytes) -> str | None:
    # reset() first: transcribe_stream is designed for incremental
    # buffered re-transcription across many calls during a live
    # utterance (see whisper_local.py's LocalAgreement-2 stabilization) —
    # a fresh reset() + exactly one call with the whole clip means
    # nothing is "committed" yet when this call runs, so the returned
    # tail *is* the complete transcript in one shot. Confirmed directly,
    # not assumed: fed real Piper-synthesized audio through this exact
    # path and got the source sentence back verbatim.
    orchestrator.stt.reset()
    return await orchestrator.transcribe_chunk(pcm16_mono)


async def _synthesize_reply(orchestrator: Orchestrator, text: str) -> tuple[np.ndarray, int]:
    chunks = [chunk async for chunk in orchestrator.synthesize_reply(text)]
    samples = np.frombuffer(b"".join(chunks), dtype="<i2")
    # sample_rate isn't part of the abstract TextToSpeech interface (it's
    # backend-specific, like SileroVAD's frame_bytes — see CLAUDE.md) but
    # both real backends (PiperTTS, CoquiTTS) expose it, and build_* only
    # ever constructs one of those two.
    return samples, orchestrator.tts.sample_rate


def _process_message(
    orchestrator: Orchestrator,
    persona: Persona,
    user_id: str,
    recent_turns: list[Turn],
    message: str,
    *,
    speak_reply: bool,
    play_after_rerun: bool = False,
) -> None:
    """`play_after_rerun`: the push-to-talk path calls `st.rerun()` right
    after this returns, which would wipe an audio player rendered here
    before it could play. In that case the audio is kept in session state
    and played once, after the rerun, by `_play_pending_audio`."""
    with st.chat_message("assistant"):
        # Tool activity (agentic personas) shows up in a collapsible status
        # box as it happens; it only appears once a tool is actually used.
        status_box = None
        tool_failed = False

        def show_agent_event(event: AgentEvent) -> None:
            nonlocal status_box, tool_failed
            if isinstance(event, ToolStarted):
                if status_box is None:
                    status_box = st.status("Using tools...", expanded=False)
                status_box.write(f"🔧 {event.name}")
            elif isinstance(event, ToolFinished) and not event.ok and status_box is not None:
                tool_failed = True
                status_box.write(f"⚠️ {event.name} reported a problem: {event.detail}")

        with st.spinner("Thinking..."):
            try:
                reply = asyncio.run(
                    orchestrator.handle_text_turn(
                        user_id=user_id,
                        persona=persona,
                        recent_turns=recent_turns,
                        message=message,
                        on_agent_event=show_agent_event,
                    )
                )
                if status_box is not None:
                    if tool_failed:
                        # Expanded, so a failure is visible even if the
                        # reply claims the action worked.
                        status_box.update(label="A tool reported a problem",
                                          state="error", expanded=True)
                    else:
                        status_box.update(label="Used tools", state="complete")
            except Exception as exc:  # noqa: BLE001 — see below
                # Same "fail loudly, not silently" posture as every other
                # entry point in this project (__main__.py's top-level
                # except Exception, which ruff doesn't flag there only
                # because that one has a conditional `raise` inside it —
                # st.stop() below is this project's Streamlit-native
                # equivalent, it just isn't syntactically a `raise` ruff's
                # checker recognizes). Shown inline rather than crashing
                # the whole page into Streamlit's raw traceback view, so
                # the chat stays usable for a retry.
                st.error(str(exc))
                st.stop()
        st.write(reply)

        if speak_reply and reply.strip():
            with st.spinner("Synthesizing speech..."):
                try:
                    samples, sample_rate = asyncio.run(_synthesize_reply(orchestrator, reply))
                except Exception as exc:  # noqa: BLE001 — same reasoning as above
                    # Deliberately not st.stop() here: a TTS failure
                    # (e.g. no real voice file downloaded for this
                    # persona yet — see module docstring) shouldn't erase
                    # the reply that already rendered successfully above.
                    st.warning(f"Couldn't synthesize speech: {exc}")
                else:
                    if play_after_rerun:
                        st.session_state["_pending_audio"] = (samples, sample_rate)
                    else:
                        # autoplay: "Speak replies" should speak without a
                        # click (the browser allows it, since sending the
                        # message was a user action).
                        st.audio(samples, sample_rate=sample_rate, autoplay=True)

    recent_turns.append(Turn(role="user", text=message))
    if reply.strip():
        recent_turns.append(Turn(role="assistant", text=reply))

    # After the reply (and speech) has rendered, so it never delays one.
    # Cheap when nothing is due (one list_turns call).
    summarizer = _shared_backends(config)["summarizer"]
    if summarizer is not None:
        try:
            asyncio.run(summarizer.summarize_if_due(user_id, persona.id))
        except Exception:  # memory upkeep must never break the chat
            logger.exception("RollingSummarizer failed; memory left unsummarized")


try:
    config = _load_config()
    personas = _load_personas(config)
except (ConfigError, PersonaError, AppSetupError) as exc:
    st.error(f"Couldn't start: {exc}")
    st.stop()

if not personas:
    st.error("No personas found — check personas.directory in your config.")
    st.stop()

with st.sidebar:
    st.header("Companion")
    persona_id = st.selectbox("Persona", sorted(personas), index=0)
    user_id = st.text_input("User ID", value="local").strip() or "local"
    speak_reply = st.checkbox(
        "🔊 Speak replies",
        value=False,
        help="Needs a real downloaded Piper voice file for this persona "
        "— see README.md's Voice mode section. Off by default since it "
        "adds synthesis latency to every reply.",
    )
    if st.button("Clear chat window"):
        # Reset directly, not pop() — popping relied on the state_key-
        # change guard below to recreate the key on rerun, but that guard
        # only fires when persona/user_id actually change, so a plain
        # clear-button rerun left "recent_turns" missing entirely and
        # crashed with a KeyError the next line down (caught by actually
        # running this through streamlit.testing.v1.AppTest, not assumed).
        st.session_state["recent_turns"] = []
        st.rerun()
    st.caption(
        "Clearing the chat window only resets what's shown here — it "
        "does not delete your companion's long-term memory."
    )
    st.divider()
    st.caption(f"Memory backend: `{config.memory.backend}`")
    st.caption(f"LLM: `{config.llm.options.get('model', '?')}` via {config.llm.backend}")

persona = select_persona(personas, persona_id)

state_key = (persona.id, user_id)
if st.session_state.get("_state_key") != state_key:
    st.session_state["_state_key"] = state_key
    st.session_state["recent_turns"] = []

if st.session_state.get("_orchestrator_persona") != persona.id:
    try:
        st.session_state["orchestrator"] = _build_orchestrator(config, persona)
        st.session_state["_orchestrator_persona"] = persona.id
    except AppSetupError as exc:
        st.error(f"Couldn't build {persona.name}: {exc}")
        st.stop()

orchestrator: Orchestrator = st.session_state["orchestrator"]
recent_turns: list[Turn] = st.session_state["recent_turns"]

st.title(persona.name)
if orchestrator.agent is not None and orchestrator.agent.handles(persona):
    allow_network = bool(config.agent and config.agent.options.get("allow_network", False))
    from companion.tools.registry import enabled_tool_names

    st.caption("Tools: " + ", ".join(enabled_tool_names(persona, allow_network)))


@st.fragment(run_every=REMINDER_POLL_S)
def _reminders() -> None:
    # Reruns on its own every REMINDER_POLL_S seconds, so a due reminder
    # appears without the user typing anything. Each is shown once (as a
    # toast) and stays listed until the page is reloaded.
    shown = st.session_state.setdefault("_reminders_shown", [])
    for line in take_due_reminders(config, user_id, personas):
        shown.append(line)
        st.toast(line)
    for line in shown[-3:]:
        st.info(line)


_reminders()

for turn in recent_turns:
    with st.chat_message(turn.role):
        st.write(turn.text)


def _play_pending_audio() -> None:
    """Plays, once, a spoken reply saved across the push-to-talk rerun."""
    pending = st.session_state.pop("_pending_audio", None)
    if pending is not None:
        samples, sample_rate = pending
        st.audio(samples, sample_rate=sample_rate, autoplay=True)


_play_pending_audio()

# Push-to-talk: st.session_state["_voice_gen"] is bumped after every
# processed recording specifically to change this widget's `key`, which
# forces Streamlit to mount a *new*, empty audio_input for the next
# recording — without that, the same recorded clip would still be sitting
# in this widget's value on every later rerun (e.g. switching personas),
# and get re-transcribed/re-sent every time instead of exactly once.
if "_voice_gen" not in st.session_state:
    st.session_state["_voice_gen"] = 0

audio_value = st.audio_input(
    f"🎤 Or record a message for {persona.name}",
    sample_rate=_STT_SAMPLE_RATE,
    key=f"voice_recorder_{st.session_state['_voice_gen']}",
)
if audio_value is not None:
    with st.spinner("Transcribing..."):
        try:
            pcm = _wav_bytes_to_pcm16_mono(audio_value.getvalue())
            transcript = asyncio.run(_transcribe_clip(orchestrator, pcm))
        except Exception as exc:  # noqa: BLE001 — same fail-loudly posture as _process_message
            st.error(f"Transcription failed: {exc}")
            transcript = None

    st.session_state["_voice_gen"] += 1  # always advance — don't leave a stale/error clip sitting in the widget

    if transcript and transcript.strip():
        with st.chat_message("user"):
            st.write(transcript)
        _process_message(
            orchestrator, persona, user_id, recent_turns, transcript.strip(),
            speak_reply=speak_reply, play_after_rerun=True,
        )
    elif transcript is not None:
        st.warning("Didn't catch any speech in that recording — try again.")
    st.rerun()

if message := st.chat_input(f"Message {persona.name}..."):
    with st.chat_message("user"):
        st.write(message)
    _process_message(orchestrator, persona, user_id, recent_turns, message, speak_reply=speak_reply)
