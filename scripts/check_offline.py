"""Prove the companion runs with no network access beyond localhost.

Usage (after models/voices have been downloaded once, see README.md):

    python scripts/check_offline.py [--config PATH] [--persona ID]

Installs a guard on Python's socket layer that refuses any connection or
DNS lookup to a non-loopback host (localhost stays allowed, because Ollama
is a local server). It then runs every backend the config names, for
real, through the same `companion.app.build_*` functions
`python -m companion` uses:

- LLM: one short reply from the configured Ollama model;
- memory: write + query on both Chroma and MemArk, in throwaway dirs;
- TTS: the persona's voice synthesizes a test sentence;
- VAD: Silero detects speech in that synthesized audio;
- STT: faster-whisper transcribes that same audio back to text;
- agent: if the config has an `agent:` section, the agent model makes one
  real tool call (the calculator) and answers with its result.

So TTS -> VAD -> STT is a real offline audio round trip, not three
isolated smoke calls. Any blocked connection attempt is listed at the end
(some libraries try to check for updates or send telemetry, then fall back
to their local cache when the network is unavailable — that is still
offline-capable, but worth knowing about).

Limit: the guard only covers networking done through Python's `socket`
module. A native library opening its own sockets in C/C++/Rust would not
be caught. For an airtight check, also run this with the machine's
network physically disabled.

Exit code 0 only if no check failed (a check whose backend extra isn't
installed is reported as SKIP, not a failure).
"""

from __future__ import annotations

import argparse
import asyncio
import errno
import ipaddress
import socket
import sys
import tempfile
import time
import traceback
from collections.abc import Awaitable, Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_SENTENCE = "Hello, this is an offline test of the companion app."

# --- network guard -------------------------------------------------------

blocked_attempts: list[str] = []


def is_loopback(host: object) -> bool:
    if host is None:
        return True  # getaddrinfo(None, ...) is a local bind lookup
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").split("%", 1)[0]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        name = host.lower().rstrip(".")
        return name == "localhost" or name.endswith(".localhost")


def install_network_guard() -> Callable[[], None]:
    """Returns a function that removes the guard again (used by tests)."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def _check(sock: socket.socket, address) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6) and not is_loopback(address[0]):
            blocked_attempts.append(f"connect {address[0]}:{address[1]}")
            raise OSError(errno.ENETUNREACH, f"offline check: blocked connection to {address[0]}")

    def connect(self, address):
        _check(self, address)
        return real_connect(self, address)

    def connect_ex(self, address):
        _check(self, address)
        return real_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if not is_loopback(host):
            blocked_attempts.append(f"dns {host}")
            raise socket.gaierror(socket.EAI_NONAME, f"offline check: blocked DNS lookup of {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo

    def uninstall() -> None:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.getaddrinfo = real_getaddrinfo

    return uninstall


# --- checks --------------------------------------------------------------


def _resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if src_rate == dst_rate or len(samples) == 0:
        return pcm
    n_out = int(len(samples) * dst_rate / src_rate)
    positions = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.int16).tobytes()


class _Skip(Exception):
    """A check that doesn't apply to this config."""


class Checks:
    def __init__(self, config, persona) -> None:
        self.config = config
        self.persona = persona
        self.speech_16k: bytes | None = None

    async def llm(self) -> str:
        from companion.app import build_llm
        from companion.core.interfaces import Turn

        llm = build_llm(self.config)
        prompt = "Reply with exactly one short sentence."
        tokens = [t async for t in llm.generate_stream(prompt, [Turn(role="user", text="Hi!")])]
        reply = "".join(tokens).strip()
        assert reply, "empty reply"
        return f"{self.config.llm.options.get('model')}: {reply[:60]!r}"

    async def _memory(self, backend: str) -> str:
        from types import SimpleNamespace

        from companion.app import build_memory
        from companion.config.loader import BackendConfig
        from companion.core.interfaces import Turn

        # ignore_cleanup_errors: on Windows the stores' open file handles can
        # outlive this block; leaving a temp dir behind isn't a failure.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            config = SimpleNamespace(memory=BackendConfig(backend=backend, options={"persist_dir": tmp}))
            store = build_memory(config)
            turn = Turn(role="user", text="I am allergic to peanuts")
            await store.write("offline", "check", turn)
            results = await store.query("offline", "check", "peanut allergy", k=3)
            assert [t.text for t in results] == [turn.text], results
        return "write + query OK"

    async def memory_chroma(self) -> str:
        return await self._memory("chroma")

    async def memory_memark(self) -> str:
        return await self._memory("memark")

    async def tts(self) -> str:
        from companion.app import build_tts_for_persona

        tts = build_tts_for_persona(self.config, self.persona)
        pcm = b"".join([chunk async for chunk in tts.synthesize_stream(TEST_SENTENCE)])
        assert pcm, "no audio produced"
        rate = tts.sample_rate
        self.speech_16k = _resample_pcm16(pcm, rate, 16000)
        return f"{self.persona.voice_id}: {len(pcm) / 2 / rate:.1f}s of audio at {rate} Hz"

    async def vad(self) -> str:
        from companion.app import build_vad

        assert self.speech_16k, "needs the TTS check's audio"
        vad = build_vad(self.config)
        if getattr(vad, "sample_rate", 16000) != 16000:
            raise AssertionError("this check feeds 16 kHz audio; set vad sample_rate 16000")
        frame = vad.frame_bytes
        audio = self.speech_16k + b"\x00\x00" * 16000  # speech, then 1s of silence
        decisions = [vad.is_speech(audio[i:i + frame])
                     for i in range(0, len(audio) - frame + 1, frame)]
        speech_frames = sum(decisions)
        assert speech_frames, "no speech detected in synthesized speech"
        assert not decisions[-1], "still reporting speech after 1s of silence"
        return f"speech in {speech_frames}/{len(decisions)} frames, silence at the end"

    async def stt(self) -> str:
        from companion.app import build_stt

        assert self.speech_16k, "needs the TTS check's audio"
        stt = build_stt(self.config)
        audio = self.speech_16k + b"\x00\x00" * 32000  # trailing silence forces a final pass
        chunk = 3200  # 100 ms at 16 kHz, 16-bit mono — like a mic callback
        text = ""
        for i in range(0, len(audio), chunk):
            partial = await stt.transcribe_stream(audio[i:i + chunk])
            if partial is not None:
                text = partial
        expected = {"offline", "test", "companion"}
        heard = {w.strip(".,!?").lower() for w in text.split()}
        assert heard & expected, f"transcript doesn't match the spoken sentence: {text!r}"
        return repr(text)

    async def agent(self) -> str:
        """One real tool round trip: the agent model calls the calculator."""
        from dataclasses import replace

        from companion.app import build_agent
        from companion.config.loader import BackendConfig
        from companion.core.agent import TextDelta, ToolStarted
        from companion.core.interfaces.types import Persona
        from companion.core.prompt_builder import PromptBuilder

        if self.config.agent is None:
            raise _Skip("no agent: section in this config")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            options = {**self.config.agent.options, "data_dir": tmp, "temperature": 0}
            config = replace(self.config, agent=BackendConfig(self.config.agent.backend, options))
            agent = build_agent(config, memory=None)
            persona = Persona(id="offline_check", name="Checker", tone="Brief.",
                              boundaries="None.", voice_id="piper:unused", tools=["calculator"])
            built = PromptBuilder().build(persona=persona, recent_turns=[], retrieved_memory=[],
                                          user_message="Use the calculator: what is 12.5 * 8?")
            events = [e async for e in agent.run_stream(persona, "offline", built)]
        tools = [e.name for e in events if isinstance(e, ToolStarted)]
        answer = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert "calculate" in tools, f"model didn't call the calculator; it said {answer!r}"
        assert "100" in answer, f"unexpected answer: {answer!r}"
        return f"{options['model']} called {tools} -> {answer.strip()[:60]!r}"


async def run(config_path: str | None, persona_id: str | None) -> int:
    from companion.app import load_personas, select_persona
    from companion.config.loader import load_config, resolve_config_path

    path = resolve_config_path(config_path)
    config = load_config(path)
    persona = select_persona(load_personas(config), persona_id)
    print(f"Config: {path}\nPersona: {persona.id}\n")

    checks = Checks(config, persona)
    steps: list[tuple[str, Callable[[], Awaitable[str]]]] = [
        ("LLM (Ollama, localhost)", checks.llm),
        ("Memory: Chroma", checks.memory_chroma),
        ("Memory: MemArk", checks.memory_memark),
        ("TTS", checks.tts),
        ("VAD (on TTS audio)", checks.vad),
        ("STT (on TTS audio)", checks.stt),
        ("Agent tool call (calculator)", checks.agent),
    ]
    failures = skipped = 0
    for name, step in steps:
        start = time.perf_counter()
        try:
            detail = await step()
            status = "PASS"
        except ImportError as exc:
            # A backend extra that isn't installed isn't an offline failure.
            detail, status = f"extra not installed ({exc.name}), not checked", "SKIP"
            skipped += 1
        except _Skip as exc:
            detail, status = f"{exc}, not checked", "SKIP"
            skipped += 1
        except Exception as exc:  # noqa: BLE001 — any failure is reported, not raised
            detail, status = f"{type(exc).__name__}: {exc}", "FAIL"
            if "--debug" in sys.argv:
                traceback.print_exc()
        failures += status == "FAIL"
        print(f"[{status}] {name} ({time.perf_counter() - start:.1f}s): {detail}")

    print()
    if blocked_attempts:
        print("Blocked network attempts (the check still passed if marked PASS above):")
        for attempt in sorted(set(blocked_attempts)):
            print(f"  - {attempt}")
    else:
        print("No non-localhost network attempts were made.")
    if failures:
        print(f"\n{failures} check(s) failed.")
    else:
        note = f" ({skipped} skipped, see SKIP lines above)" if skipped else ""
        print(f"\nAll checks that ran passed offline{note}.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--persona", default=None)
    parser.add_argument("--debug", action="store_true", help="print tracebacks for failures")
    args = parser.parse_args()
    install_network_guard()
    return asyncio.run(run(args.config, args.persona))


if __name__ == "__main__":
    sys.exit(main())
