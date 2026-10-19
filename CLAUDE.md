# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**OpenPersona** (repo `Arkur745/OpenPersona`; Python package/CLI name
`companion`) is an open-source, local-first AI companion app: voice/text
chat with a swappable persona (mentor, friend, romantic companion, or
community-contributed), running on local models by default. The
deployed, cloud-hosted version is a **separate project in a separate
repo**, not a phase of this one. **Do not build cloud backends/adapters,
hosted infra, or auth/billing in this repo**, even if a task seems to
gesture toward it. This repo's claim is "fully offline-capable"
(`scripts/check_offline.py` checks it), and a cloud dependency here would
break that.

**Current state: both text and voice mode are real, runnable apps
(`python -m companion`, mode picked by `config.yaml`).** Directory
structure, the five abstract interfaces, the persona format + 3 starter
personas, config format, and docs exist.
`companion/backends/stt/whisper_local.py` (`FasterWhisperSTT`),
`companion/backends/tts/{piper_local,coqui_local}.py` (`PiperTTS`,
`CoquiTTS`), `companion/backends/llm/ollama_local.py` (`OllamaLLM`),
`companion/backends/vad/silero_vad.py` (`SileroVAD`),
`companion/core/prompt_builder/prompt_builder.py` (`PromptBuilder`),
`companion/core/persona_loader/persona_loader.py` (`PersonaLoader`,
`load_persona`), `companion/memory/store.py` (`VectorMemoryStore`,
Chroma-backed), `companion/config/loader.py` (`load_config`,
`AppConfig`), `companion/app.py` (backend construction + the text loop,
shared by both modes), `companion/audio/` (`MicCapture`,
`SpeakerPlayback`, sounddevice-backed real mic/speaker I/O), and
`companion/voice_app.py` (voice-mode bootstrap + loop) are all
implemented. Both `Orchestrator.handle_text_turn` and `.handle_voice_turn`
are real — `handle_voice_turn` composes all of the above into a full
listening/processing/speaking-response state machine with barge-in,
looping across an entire session (see `docs/ARCHITECTURE.md`'s
"Voice-turn state machine" section), querying and writing real memory
each turn; `handle_text_turn` is request/response (caller-owned
`recent_turns`) but otherwise does the same memory query/dedup/write.
`handle_voice_turn` also now takes two optional hooks
(`on_playback_interrupt`/`wait_for_playback_drained`) that
`companion/audio/playback.py`'s `SpeakerPlayback` drives, closing a
previously-documented known limitation for real (barge-in detection used
to stop as soon as TTS finished *yielding*, not once audio actually
finished *playing* — see `docs/ARCHITECTURE.md`'s "Voice mode audio I/O"
section for the full mechanism and the regression test proving it).
`python -m companion` genuinely runs a real conversation end to end in
either mode — text mode verified against a real local Ollama server and
real Chroma storage; voice mode verified against real Silero VAD, real
faster-whisper STT, real Ollama, real Piper TTS, real Chroma, and real
mic/speaker hardware together in one round trip, not just tests (see
README.md's "Running the app" and `docs/ARCHITECTURE.md`'s "Voice mode
audio I/O" testing-reality-check subsection for exactly what was
automated vs. manually run). That first real round trip — a real,
continuous mic source, not a hand-scripted frame list — also surfaced and
fixed a real deadlock in `_pump_mic`/`handle_voice_turn`'s shared-queue
design (a one-shot "mic stream ended" signal could be consumed by the
wrong one of two alternating readers, permanently hanging the next
listening iteration); see `docs/ARCHITECTURE.md`'s "A real deadlock found
via the first real, continuous mic round trip" section and
`Orchestrator._pump_mic`'s docstring. What's *not* done yet:
`RollingSummarizer`
(`companion/memory/summarizer.py`, unimplemented/unwired).
(`llm.voice_mode_override` used to be listed here too; it's now real:
`companion.app.build_llm` merges its options over `llm.options` in voice
mode.) Check whether the
specific method/module you're touching is one of those gaps before
assuming it's live. Project license is MIT (this repo's own code only —
see `THIRD_PARTY_LICENSES.md` for bundled/downloaded model weights,
which do not inherit that license).

## `legacy/`

`legacy/` contains an **unrelated, archived prior project**: a Node/Express
+ React text-chat app (Gemini + MongoDB + Pinecone + ElevenLabs TTS-only),
moved here wholesale from what used to be this repo's root. It shares no
code, stack, or architecture with the Python companion app and should be
treated as read-only reference at most — do not import from it, extend it,
or assume its patterns apply here. If you're asked to work on "the
backend" or "the frontend" without qualification, that almost certainly
means `companion/`, not `legacy/`.

## Commands

### Running the app

Both modes are real — see README.md's "Running the app" for the full
walkthrough (voice mode also needs a real voice file for whichever
persona you pick — starter personas' `voice_id`s are placeholders, see
`personas/schema.md`). Short version:

```
# text mode
pip install -e ".[dev,memory-chroma]"   # STT/TTS/VAD extras not needed —
                                          # text mode never touches them,
                                          # see companion/app.py
cp companion/config/config.example.yaml companion/config/config.yaml
# edit config.yaml's llm.options.model to a model you've actually
# `ollama pull`ed — the shipped default isn't pre-pulled for you
python -m companion

# voice mode: also install stt-whisper,tts-piper,vad-silero,audio-sounddevice,
# set mode: voice in config.yaml, and point a persona's voice_id at a real
# downloaded .onnx voice file — see README.md for the exact steps
```

### Tests

```
pip install -e ".[dev]"
pytest                    # interface-conformance + STT wrapper-logic tests, no optional deps needed
ruff check .               # lint
```

`pytest` alone (just the `dev` extra, which includes `numpy`) covers
`FasterWhisperSTT`'s, `PiperTTS`'s, and `CoquiTTS`'s own logic against
fake models/voices — deterministic, no network. Add the matching extra to
also run each backend's real end-to-end smoke test, which downloads a
small model/voice on first run and skips itself (not fails) if that's not
possible in the current environment:

```
pip install -e ".[dev,stt-whisper]"   # test_real_faster_whisper_backend_smoke
pip install -e ".[dev,tts-piper]"     # test_real_piper_backend_smoke
pip install -e ".[dev,tts-coqui]"     # test_real_coqui_backend_smoke — some
                                       # stock Coqui models additionally need
                                       # the espeak-ng system binary; see
                                       # THIRD_PARTY_LICENSES.md
pip install -e ".[dev,vad-silero]"    # test_real_silero_backend_smoke — torch
                                       # is deliberately NOT in the dev extra
                                       # (unlike numpy/ollama, it's genuinely
                                       # heavy); offline VAD tests mock at
                                       # _speech_probability instead of _model
                                       # so they never need it
pip install -e ".[dev,memory-chroma]" # tests/test_memory_store.py's real-DB
                                       # tests (write, query ordering,
                                       # scoping, delete) — chromadb pulls in
                                       # onnxruntime for its own default
                                       # local embedding model, no torch;
                                       # also gates
                                       # test_app.py::test_real_end_to_end_text_conversation_smoke,
                                       # which additionally needs a real
                                       # Ollama server with a model pulled
                                       # (same skip conditions as the
                                       # OllamaLLM real tests below)
pip install -e ".[dev,audio-sounddevice]" # test_real_mic_capture_smoke /
                                       # test_real_speaker_playback_smoke —
                                       # opens the real default input/output
                                       # device; skips cleanly if none is
                                       # available (e.g. headless/CI). See
                                       # docs/ARCHITECTURE.md's "Voice mode
                                       # audio I/O" testing-reality-check
                                       # subsection for exactly what these
                                       # do and don't verify.
pip install -e ".[dev,memory-chroma,memory-memark]"
                                       # tests/test_memory_persistence_across_processes.py —
                                       # every step in its own subprocess
                                       # (tests/_persistence_child.py), both
                                       # backends; ~1 min. Its real-app test
                                       # also needs local Ollama (skips otherwise).
```

`scripts/check_offline.py` is the offline-capability check: it blocks
non-localhost sockets/DNS, then runs every configured backend for real
(TTS → VAD → STT audio round trip included). It needs downloaded
models and a running Ollama server, so it's run by hand, not by
`pytest`. Its network guard is unit-tested in `tests/test_check_offline.py`.
If you add a backend, add it to that script too.

`tests/test_orchestrator_voice_turn.py` tests the full voice-turn state
machine (including barge-in) entirely with fakes — no optional deps
needed. If you touch `handle_voice_turn`'s concurrency (the `_pump_mic`
task / `asyncio.Queue` design), read that test file's module docstring
first — there's a real, previously-shipped concurrency bug documented
there (cancelling a task mid-`await` on a *shared* async generator kills
it silently) that the current design specifically avoids; don't
reintroduce a second call site that reads `mic_audio` directly.

`OllamaLLM`'s real end-to-end tests (`test_real_ollama_*`) need a running
local Ollama server — `ollama` itself is already in the `dev` extra (see
`pyproject.toml`'s comment on why), so no separate install step is needed,
but the tests skip themselves if no server is reachable at
`localhost:11434` or if the server has no models pulled at all (unlike the
STT/TTS smoke tests, they don't auto-pull one — an LLM is too large to
fetch automatically in a test).

For any other backend, install its extra — see `pyproject.toml` for the
full list of optional-dependency groups (`stt-whisper`, `tts-piper`,
`tts-coqui`, `llm-ollama`, `vad-silero`, `memory-chroma`, `memory-lancedb`,
`audio-sounddevice`).
Running the LLM backend also requires a local Ollama server
(`ollama pull <model>` for whatever `companion/config/config.yaml` names).

Config: copy `companion/config/config.example.yaml` →
`companion/config/config.yaml` (git-ignored) and edit — this is the only
place backend selection happens, see below.

## Architecture (see `docs/ARCHITECTURE.md` for the full version)

- **`companion/core/interfaces/`** — five abstract contracts
  (`SpeechToText`, `TextToSpeech`, `LLMBackend`, `VoiceActivityDetector`,
  `MemoryStore`) plus shared value types (`Turn`, `Persona`). Nothing
  outside `companion/backends/` and `companion/memory/` may import a
  concrete backend class directly — only these interfaces. This is the
  rule most likely to get violated by a well-intentioned shortcut; don't.
- **`companion/backends/{stt,tts,llm,vad}/`** — concrete implementations,
  one file per backend, all implemented (`whisper_local.py`,
  `piper_local.py`, `coqui_local.py`, `ollama_local.py`, `silero_vad.py`).
  Adding a new backend = new file implementing the interface + a new
  optional-dependency group in `pyproject.toml` + a new `backend:` value
  documented in `config.example.yaml`. Never branch on backend name
  outside this layer. `companion/backends/tts/_sync_stream.py` is a shared
  (not interface-defined) helper both TTS backends use to bridge their
  blocking synthesis calls onto a background thread — reuse it rather than
  reimplementing thread/queue plumbing if you add a third TTS backend.
  `SileroVAD` requires exactly-sized audio frames (`frame_bytes`,
  discoverable but not part of the abstract `VoiceActivityDetector`
  interface) — see `docs/ARCHITECTURE.md`'s "VAD" section.
- **`companion/core/orchestrator/orchestrator.py`** — composes the
  interfaces into the actual conversation loop. Both `handle_text_turn`
  (request/response, caller-owned `recent_turns`) and `handle_voice_turn`
  (a full listening/processing/speaking-response/barge-in state machine,
  its own session-scoped `recent_turns`) are real — see
  `docs/ARCHITECTURE.md`'s "Voice-turn state machine" section, including a
  real concurrency bug found and fixed while building it (don't let two
  call sites read the same `mic_audio` async generator directly — see
  `_pump_mic`'s docstring). Both query `MemoryStore` for relevant past
  turns, de-duplicate against the live conversation's own history via the
  module-level `_dedup_retrieved_memory` (see docs/MEMORY_STORE.md
  section 4 for why that's resolved here, not inside `MemoryStore`), and
  write both sides of each exchange back to memory. `transcribe_chunk`,
  `synthesize_reply`/`stop_speaking`, and `generate_reply` remain as real,
  standalone single-backend delegation methods, reused by
  `handle_voice_turn` rather than duplicated. `handle_voice_turn` also
  takes two optional keyword-only hooks, `on_playback_interrupt` and
  `wait_for_playback_drained` (both default `None`, so every existing
  caller/test is unaffected) — real callers (`companion/voice_app.py`)
  wire these to `SpeakerPlayback.stop`/`.wait_drained` so the barge-in
  watcher stays alive through real playback latency, not just
  synthesis+yield latency. Not an abstract-interface change (neither
  parameter touches `core/interfaces/`) — see docs/ARCHITECTURE.md's
  "Voice mode audio I/O" section for the full mechanism and why it was
  needed.
- **`companion/core/prompt_builder/prompt_builder.py`** — the *only* place
  persona + retrieved memory + recent conversation window get assembled.
  If you find prompt-assembly logic anywhere else, that's a bug to fix,
  not a pattern to follow. Implemented: `build()` returns
  `BuiltPrompt(prompt, context)`, not a bare string — see
  `docs/PROMPT_BUILDER.md` for the full design, including two refinements
  found during implementation (`user_message` ends up in `context`, not
  `prompt`; `recent_turns` bounding is enforced here, which is *why*
  `build()` returns `context` at all). It's a pure function — no
  `MemoryStore` dependency; the caller is expected to fetch memory and
  pass it in as `retrieved_memory`.
- **`companion/personas/*.yaml`** — persona configs (see `schema.md` in
  that directory and `docs/PERSONA_AUTHORING.md`). Adding a persona must
  never require a code change. Loaded and validated by
  `companion/core/persona_loader/persona_loader.py` — `PersonaLoader`
  scans a directory once (`load_all()`) and caches for `get(id)` lookups;
  validation fails loudly and specifically (file + field + reason) on a
  malformed persona. The loader only validates that `voice_id` is
  *shaped* `<namespace>:<value>` (e.g. `piper:...` / `coqui:...`) —
  actually resolving it via a backend's `from_voice_id()` classmethod
  (`PiperTTS.from_voice_id`, `CoquiTTS.from_voice_id`) into a bound
  `TextToSpeech` instance happens once per persona, in
  `companion/app.py`'s `build_tts_for_persona`, not inside the loader
  itself. A `TextToSpeech` instance is bound to one voice for its
  lifetime; there's no per-call voice parameter on `synthesize_stream`.
- **`companion/memory/store.py`** — `VectorMemoryStore`, `MemoryStore`
  implemented over Chroma (`backend="lancedb"` named but not implemented,
  same pattern as STT's `whisper_cpp`) — see docs/MEMORY_STORE.md for the
  full design. One Chroma collection per `(user_id, persona_id)` (hashed
  `user_id` — collection names are charset-restricted), a structural
  no-cross-persona-leak guarantee, not just a filter-correctness one.
  Embeddings use Chroma's own default (ONNX MiniLM-L6, local, no torch).
  `query()`'s ordering is Chroma's own native most-similar-first, not
  re-sorted. `summarizer.py`'s `RollingSummarizer` is unimplemented and
  unwired — a background rolling-summarization job for later, not
  something called inline in the request path.
- **`companion/config/config.example.yaml`** — the single source of truth
  for which concrete backend class gets instantiated for each interface.
  If you're about to write `if backend_name == "piper":` anywhere outside
  the code that reads this file, stop — that branching belongs here, not
  scattered through the app. "The code that reads this file" is now a
  real, specific place: `companion/config/loader.py` (`load_config`
  parses+validates it into `AppConfig` — structural validation only, see
  that module's docstring for why it deliberately doesn't reject unknown
  `options:` keys) and `companion/app.py` (`build_stt`/`build_llm`/
  `build_vad`/`build_tts_for_persona`/`build_memory` construct the real
  backend classes from it — this is where backend-name branching
  actually lives now).
- **`companion/app.py` + `companion/__main__.py`** — the app bootstrap:
  config -> real backends -> a selected persona -> `Orchestrator`, shared
  by both modes, plus text mode's own loop (`run_text_loop`).
  `__main__.py` is a thin CLI shim (argparse, path resolution, dispatch on
  `config.mode` to `run_text_loop` or `companion/voice_app.py`'s
  `run_voice_session`, top-level error/`KeyboardInterrupt` presentation
  via `except Exception: print(f"Error: {exc}")` — works because every
  lower layer already raises actionable, specific messages, not because
  `__main__.py` special-cases each error type); everything else is in
  `app.py`/`voice_app.py`, directly testable without argparse, real
  stdin, or real audio hardware (see `run_text_loop`'s injectable
  `read_input`/`print_fn` params and `run_voice_loop`'s
  pre-constructed-I/O-object design). Persona selection
  (`select_persona`) and the memory-scoping `user_id` both had no prior
  definition of how they should work — see `app.py`'s module docstring
  for the reasonable defaults chosen (first-loaded persona / `"local"`)
  and the CLI overrides (`--persona`, `--user-id`).
- **`companion/audio/` + `companion/voice_app.py`** — real mic capture
  (`capture.py`'s `MicCapture`, directly usable as `handle_voice_turn`'s
  `mic_audio`) and speaker playback (`playback.py`'s `SpeakerPlayback`),
  sounddevice-backed; `voice_app.py` wires them into `handle_voice_turn`
  (including the `on_playback_interrupt`/`wait_for_playback_drained`
  hooks above) and is voice mode's counterpart to `app.py`'s text loop.
  Any `sounddevice.PortAudioError` (no device, invalid device, permission
  denied, unsupported sample rate) is wrapped as
  `companion.audio.errors.AudioDeviceError` with an actionable message —
  never let a raw `PortAudioError` reach the user. `audio.options
  .sample_rate` must equal `vad.options.sample_rate`; `voice_app.py`
  checks this explicitly before opening any device (the config loader
  itself doesn't cross-check option *contents* between sections — see its
  docstring). See docs/ARCHITECTURE.md's "Voice mode audio I/O" section,
  including its testing-reality-check subsection — real device behavior
  is deliberately not something the automated test suite claims to
  verify beyond "the real API runs without erroring."

## Build order

Tracking the phase-1 sequence from the project brief (do not skip ahead
without being asked):

1. ~~Audit + cleanup~~ — done; see `legacy/` above.
2. ~~Scaffold repo structure~~ — done (this state).
3. ~~Implement local STT backend + streaming wrapper~~ — done:
   `FasterWhisperSTT` (buffered re-transcription + LocalAgreement-2
   stabilization, not true incremental decoding — see
   `docs/ARCHITECTURE.md`'s "STT" section for why), wired via
   `Orchestrator.transcribe_chunk`. `whisper_cpp` CPU-fallback variant
   named in config is still unimplemented.
4. ~~Implement local TTS backend~~ — done: `PiperTTS` (default; native
   per-sentence streaming) and `CoquiTTS` (optional, higher fidelity;
   sentence-granularity via a blocking call + output chunking — see
   `docs/ARCHITECTURE.md`'s "TTS" section for why they differ there), both
   wired via `Orchestrator.synthesize_reply` / `.stop_speaking`. Voice
   cloning (Coqui/XTTS `speaker_wav`) is documented but not wired up —
   same section.
5. ~~Implement Ollama LLM backend + PromptBuilder~~ — done. `OllamaLLM`,
   wired via `Orchestrator.generate_reply`; see `docs/ARCHITECTURE.md`'s
   "LLM" section for a real gotcha in how streaming errors surface, and
   for the `qwen2.5:7b-instruct` default's license-driven justification
   (full reasoning in `ollama_local.py`'s docstring). `PromptBuilder`
   assembles `BuiltPrompt(prompt, context)` from persona + memory +
   recent turns — pure function, no `MemoryStore` dependency (Option A in
   `docs/PROMPT_BUILDER.md`, confirmed).
6. ~~Implement Silero VAD + barge-in handling in the orchestrator~~ —
   done. `SileroVAD` (fixed-frame requirement, internal debouncing — see
   `docs/ARCHITECTURE.md`'s "VAD" section) and a full
   `Orchestrator.handle_voice_turn` state machine with barge-in (see that
   doc's "Voice-turn state machine" section). This is the first pass
   where a whole voice turn runs for real, end-to-end — with
   `recent_turns=[]`/`retrieved_memory=[]` hardcoded into
   `PromptBuilder.build()` calls, clearly flagged as temporary. Same gap
   blocks `handle_text_turn`.
7. ~~Implement persona config loader (YAML → `Persona`)~~ — done.
   `PersonaLoader`/`load_persona` in
   `companion/core/persona_loader/persona_loader.py`; see
   `docs/ARCHITECTURE.md`'s "Persona loader" section. Validates shape and
   required fields per `personas/schema.md`; does not resolve `voice_id`
   to a live TTS backend instance (that's step 9's app-wiring job).
8. ~~Implement `MemoryStore`~~ — done (Chroma; LanceDB named but
   deferred, see `docs/MEMORY_STORE.md`). `VectorMemoryStore` in
   `companion/memory/store.py`; `Orchestrator.handle_text_turn` (now real)
   and `.handle_voice_turn` both query/dedup/write real memory each turn
   — see `docs/ARCHITECTURE.md`'s "MemoryStore" section. The summarizer
   job (`companion/memory/summarizer.py`'s `RollingSummarizer`) stays
   unimplemented/unwired — out of scope for this step, per that doc.
9. ~~Wire the full app end-to-end~~ — done, for text mode.
   `companion/config/loader.py` reads/validates `config.yaml`;
   `companion/app.py` constructs the real concrete backend instances it
   names and resolves the selected persona's `voice_id` via
   `from_voice_id()`; `companion/__main__.py` (`python -m companion`) is
   a real, runnable text conversation loop — verified against a real
   local Ollama server and real Chroma storage, not just tests. See
   `docs/ARCHITECTURE.md`'s "App bootstrap" section, including a real
   pre-existing gap found while wiring real backends together
   (`llm.voice_mode_override` wasn't implemented; it is now, in
   `app.build_llm`, applied once per voice session).
10. ~~Implement a voice-mode CLI (real mic capture + real speaker
    playback)~~ — done. `companion/audio/` (`MicCapture`,
    `SpeakerPlayback`, sounddevice-backed) + `companion/voice_app.py`
    wire real audio I/O into `Orchestrator.handle_voice_turn`;
    `config.yaml`'s `mode: voice` drives `python -m companion` into this
    path instead of text mode's loop. Closed a previously-documented
    known limitation for real in the process: `handle_voice_turn` gained
    two optional hooks (`on_playback_interrupt`/
    `wait_for_playback_drained`) so barge-in detection stays accurate
    through real playback latency, not just synthesis+yield latency —
    see `docs/ARCHITECTURE.md`'s "Voice mode audio I/O" section for the
    full mechanism, the regression test proving it, and — importantly —
    its testing-reality-check subsection spelling out exactly what's
    automated-tested vs. only verified by manually running a real
    round trip (real VAD/STT/LLM/TTS/MemoryStore/audio hardware
    together) by hand.
11. Write up real numbers in `docs/HARDWARE_REQUIREMENTS.md` (currently a
    draft based on published specs, not measured).
12. Cloud backend adapters — **do not start until explicitly instructed.**

## Open decisions (don't resolve unilaterally)

- **License** — resolved: this repo's own code is MIT (`LICENSE`,
  `pyproject.toml`). Bundled/downloaded model weights are a separate,
  ongoing concern — see `THIRD_PARTY_LICENSES.md`, and update it whenever
  a new backend or new default model is integrated; don't assume MIT
  extends to a model just because the code loading it does.
- **UI framework** — resolved for a first UI: Streamlit
  (`companion/ui/streamlit_app.py`), built against the same
  `build_*`/`Orchestrator` construction the CLI text loop uses — text
  chat plus optional push-to-talk voice (record a clip, transcribed
  one-shot, spoken replies optional; NOT the CLI's real-time hands-free
  voice mode, which doesn't fit Streamlit's rerun model) — see
  `docs/ARCHITECTURE.md`'s "Open decisions" entry and
  `companion/ui/README.md`. Not treated as the final or only UI this
  project will ever have (real-time hands-free voice in a browser, a
  richer web frontend, a TUI, a desktop app are all still open) — don't
  assume Streamlit is the answer for a future UI need without checking
  whether it still fits.
