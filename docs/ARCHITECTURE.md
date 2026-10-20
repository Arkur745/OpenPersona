# Architecture

## Status

The local-first core is built and runs end to end: text, voice (with
barge-in) and a Streamlit UI, five agentic personas with tools, Chroma or
MemArk memory, and rolling summarization. This document describes how it
works as built (last reviewed 2026-10-19). Cloud backends, hosted infra
and auth/billing don't belong in this repo: the deployed version is a
separate project. See "Build order" below. The reasons behind the
main choices are collected in `DECISIONS.md` at the repo root (D1–D31);
this document describes the mechanics.

## Principles

1. **Backend-agnostic interfaces.** `companion/core/interfaces/` defines
   five abstract contracts: `SpeechToText`, `TextToSpeech`, `LLMBackend`,
   `VoiceActivityDetector`, `MemoryStore`. Everything outside
   `companion/backends/` and `companion/memory/` (orchestrator,
   prompt_builder, UI) depends only on these interfaces, never on a
   concrete implementation. That's what makes backends swappable through
   config alone (and what will let the separate deployed version plug in
   other backends without forking the orchestrator).
2. **Config-driven backend selection.** `companion/config/config.example.yaml`
   is the *only* place a backend is chosen by name. There is no `if
   config.stt == "whisper"` branching anywhere else — the app reads config
   once at startup, constructs the named class from `companion/backends/`,
   and passes it into the `Orchestrator`. Adding a new backend means
   adding a new class that satisfies an existing interface plus one new
   config option — not editing the orchestrator.
3. **Persona and memory are backend-agnostic.** A `Persona`
   (`companion/personas/*.yaml`, loaded into
   `companion.core.interfaces.types.Persona`) never references a specific
   STT/LLM/TTS backend by name — only a `voice_id` namespaced by *TTS
   backend kind* (`piper:...` / `coqui:...`), resolved at runtime against
   whichever TTS backend config selects. Memory (`companion/memory/`) is
   scoped by `(user_id, persona_id)` and knows nothing about which LLM
   generated the turns it stores.
4. **No mandatory external services for the local-first path.** STT/TTS/VAD
   run in-process; the LLM talks to a local Ollama server (which the user
   runs, but which is not a *cloud* dependency); memory is embedded (Chroma,
   or MemArk's SQLite graph) with no separate DB server to stand up. The
   only network-using features are the agent tools `web_search` and
   `weather`, off unless `agent.options.allow_network: true`.
   `scripts/check_offline.py` checks this by blocking non-localhost
   connections and running every configured backend for real.
5. **Tools are opt-in per persona.** A persona becomes an agent only by
   listing `tools:` in its YAML; no persona needs code, and personas
   without tools never touch the agent stack. See "Agentic personas".

## Data flow (one turn)

```
                         ┌─────────────────┐
   mic audio ──chunks──▶ │ VoiceActivity-   │──speech?──▶ buffer
                         │ Detector         │
                         └─────────────────┘
                                  │ end-of-utterance
                                  ▼
                         ┌─────────────────┐
                         │ SpeechToText     │──partial/final transcript
                         └─────────────────┘
                                  │
                                  ▼
   user_id, persona ──▶  MemoryStore.query() ──▶ retrieved_memory: list[Turn]
                                  │
                                  ▼
                         ┌─────────────────┐
   persona + recent_turns +      │ PromptBuilder    │──▶ BuiltPrompt(prompt: str,
   retrieved_memory + ──▶│ .build()         │              context: list[Turn])
   user_message           └─────────────────┘        (user_message becomes the
                                  │                    final turn in context —
                                  ▼                    see docs/PROMPT_BUILDER.md)
                         ┌─────────────────┐
   prompt, context ────▶ │ LLMBackend       │──token stream──▶ sentence buffer
                         │ .generate_stream │
                         └─────────────────┘
                                  │ completed sentence
                                  ▼
                         ┌─────────────────┐
                         │ TextToSpeech     │──audio chunks──▶ playback
                         │ .synthesize_stream│
                         └─────────────────┘
                                  │
                                  ▼
                MemoryStore.write() for both the user turn and the
                assistant turn (see companion/core/orchestrator/orchestrator.py)
```

This diagram shows the data shapes flowing through one exchange; it's not
the actual control flow for voice mode, which loops through many
exchanges and runs VAD/STT concurrently with TTS for barge-in — see
"Voice-turn state machine" below for the real state diagram and how
barge-in (`VoiceActivityDetector.is_speech()` firing during playback ->
`TextToSpeech.stop()`) actually works.

Text mode skips VAD/STT/TTS entirely; the same `MemoryStore` →
`PromptBuilder` → `LLMBackend` path is shared. Both `handle_text_turn`
and `handle_voice_turn` are real now (build-order step 8, `MemoryStore` —
see "MemoryStore (implemented)" below and docs/MEMORY_STORE.md for the
design). The one difference: `handle_text_turn` is request/response (the
caller passes `recent_turns` fresh each call), while `handle_voice_turn`
tracks its own `recent_turns` across a whole session — see "Voice-turn
state machine" below.

Two additions to this flow:
- **Personas with tools:** the `LLMBackend.generate_stream` box is
  replaced by the agent (`AgentRunner.run_stream`), which may call tools
  before answering. Everything before and after is identical, and memory
  still receives only the user message and the final answer. See "Agentic
  personas".
- **After the memory writes:** `RollingSummarizer.summarize_if_due` is
  scheduled as a background task, never awaited by the turn. See "Rolling
  summarization" under MemoryStore.

## Directory layout

```
companion/
  __main__.py            # `python -m companion`: thin CLI shim, dispatches on config.mode
  app.py                 # shared bootstrap: config -> real backends -> Orchestrator; text loop; reminders
  voice_app.py           # voice-mode bootstrap: real audio I/O -> Orchestrator; voice loop
  audio/                 # MicCapture (capture.py) + SpeakerPlayback (playback.py), sounddevice-backed
  core/
    interfaces/          # the five abstract contracts + shared value types (Turn, Persona)
    orchestrator/        # composes the interfaces into the text and voice turn loops
    prompt_builder/      # the one place persona + memory + context get assembled into a prompt
    persona_loader/      # loads + validates personas/*.yaml into Persona
    agent/               # AgentRunner protocol + events (TextDelta, ToolStarted, ToolFinished)
  backends/
    stt/, tts/, llm/, vad/   # concrete implementations of the interfaces
    agent/               # LangGraphOllamaAgent: the only LangGraph/LangChain code
  tools/                 # persona tools (plain async functions), registry, SQLite store (notes, reminders)
  memory/                # Chroma store (store.py), MemArk SQLite graph store, RollingSummarizer
  memark/                # MemArk retrieval adapter + vendored engine (_vendor/, unmodified)
  personas/              # *.yaml persona configs + schema.md
  config/                # config.example.yaml + loader.py (config.yaml -> AppConfig)
  ui/                    # streamlit_app.py: browser UI (text + push-to-talk voice)
scripts/                 # check_offline.py, benchmark_latency.py, benchmark_memory_backends.py
docs/
tests/
legacy/                  # archived prior Node/React project; not part of this codebase, do not build on it
```

## Build order

See the repository root `CLAUDE.md` for the current phase and the full
numbered build order. Short version: interfaces and scaffold exist now;
implementations land backend-by-backend — STT, TTS, the LLM backend,
PromptBuilder, VAD, the persona loader, and `MemoryStore` are all done.
Both `handle_text_turn` and `handle_voice_turn` run the full pipeline
end-to-end for real, memory included. App bootstrap is done for *both*
modes — `python -m companion` is a real, runnable conversation loop in
text or voice mode (see "App bootstrap" and "Voice mode audio I/O"
below). Agentic personas (personas with tools, run through a LangGraph
agent loop) are implemented too; see "Agentic personas" below. The
deployed, cloud-hosted version is a separate project, not a phase of this
repo.

## STT (implemented)

`companion/backends/stt/whisper_local.py`'s `FasterWhisperSTT` is the first
non-stub backend. Two things worth knowing before touching it or writing a
second `SpeechToText` implementation:

- **Audio format contract.** `transcribe_stream(audio_chunk: bytes)` assumes
  mono 16 kHz 16-bit signed little-endian PCM. This isn't stated in the
  abstract interface itself (which stays format-agnostic on purpose), but
  every concrete implementation in this codebase — STT and VAD — agrees on
  one format, so the mic capture layer (`companion/audio/capture.py`)
  doesn't have to convert per backend. If you add a backend
  that needs a different format, that's a real conflict to resolve, not
  something to silently paper over with a per-backend conversion.
- **"Streaming" here means buffered re-transcription, not incremental
  decoding.** faster-whisper has no API to cheaply extend a previous
  decode with new audio — nothing local-Whisper-based does. The backend
  buffers audio and periodically re-transcribes the whole buffer,
  stabilizing output with a LocalAgreement-2 policy (a word is only
  "committed" — and therefore guaranteed not to change in what's
  returned — once it agrees across two consecutive passes). Full
  reasoning and the explicit non-goal (word-timestamp-aligned buffer
  trimming, not implemented) are in the module's docstring. The output
  must be built from `committed_words` + the tentative tail, never from
  the raw per-pass hypothesis directly — that's what makes committed text
  actually stable instead of just tracked-but-ignored bookkeeping.

## TTS (implemented)

`companion/backends/tts/piper_local.py` (`PiperTTS`, default) and
`coqui_local.py` (`CoquiTTS`, optional higher-fidelity) both implement
`TextToSpeech`. Worth knowing before touching either or adding a third:

- **Neither library is asyncio-native.** Both expose synthesis as a
  blocking, CPU/GPU-bound call. `companion/backends/tts/_sync_stream.py`
  is the one shared place that bridges a blocking generator onto a
  background thread without blocking the event loop — both backends use
  it rather than each reimplementing thread/queue plumbing. It's also
  what makes `TextToSpeech.stop()` actually interrupt in-flight synthesis
  (checked on both the producer thread and the consumer side), not just
  get ignored.
- **Streaming granularity is per text_chunk (typically one sentence), not
  sub-sentence.** Piper's own `PiperVoice.synthesize()` happens to yield
  one `AudioChunk` per sentence natively, so a multi-sentence `text_chunk`
  streams multiple pieces there for free. Coqui's high-level `TTS.api.TTS`
  has no native streaming API at all — `.tts()` returns one complete,
  already-concatenated waveform per call — so `CoquiTTS` synthesizes the
  whole `text_chunk` in one blocking call and then slices the *result*
  into fixed-size byte chunks (`chunk_frames`) before yielding, purely so
  a downstream player still gets a stream of chunks rather than one large
  blob. That slicing doesn't reduce that call's synthesis latency; the
  actual latency win (not waiting for the full LLM response before
  starting synthesis) comes from the orchestrator calling
  `synthesize_stream()` per sentence in the first place, which both
  backends benefit from equally.
- **Persona voice binding happens outside the interface**, via a
  `from_voice_id(voice_id, ...)` classmethod on each backend
  (`PiperTTS.from_voice_id`, `CoquiTTS.from_voice_id`) — not a parameter
  on `synthesize_stream` itself. A `TextToSpeech` instance is bound to one
  voice for its lifetime. The persona loader (see "Persona loader" below)
  validates that `voice_id` is *shaped* `<namespace>:<value>` but does not
  itself call `from_voice_id()` — actually resolving a persona's
  `voice_id` into a bound `TextToSpeech` instance needs a live backend
  constructed from `config.yaml`, which is `companion/app.py`'s
  `build_tts_for_persona` (build order step 9, done — see "App bootstrap"
  below), not the loader. See personas/schema.md's `voice_id` field
  notes.
- **Voice cloning (Coqui/XTTS only) is deliberately not wired up.** The
  constructor parameter (`speaker_wav`) and the natural extension point in
  `from_voice_id` are documented in `coqui_local.py`'s module docstring —
  adding it needs no change to `TextToSpeech`, `Persona`, or
  personas/schema.md's shape, only a policy decision about voice-cloning
  consent tracking, which is a product question, not a technical one.
- **Single-speaker Coqui models** (e.g. `tts_models/en/ljspeech/vits`)
  reject any speaker name, so a persona selects their only voice with
  `voice_id: coqui:default`; `CoquiTTS.from_voice_id` maps it to no
  speaker. Found by running `scripts/check_offline.py` with Coqui, which
  now passes with that model.
- **`espeak-ng`** (a system binary, not a Python package) is required at
  runtime by some stock Coqui models' phonemizers — not by Piper, and not
  by XTTS. See THIRD_PARTY_LICENSES.md.

## VAD (implemented)

`companion/backends/vad/silero_vad.py`'s `SileroVAD` implements
`VoiceActivityDetector`. Two things worth knowing:

- **Fixed-size frames only — verified against the real model, not
  assumed.** Silero's model is a TorchScript module compiled for an exact
  input length: 512 samples at 16 kHz (256 at 8 kHz). Calling it with any
  other length raises a TorchScript runtime error directly — confirmed by
  actually calling it with wrong sizes and watching it fail.
  `get_speech_timestamps`'s own `window_size_samples` parameter is marked
  `!!! DEPRECATED, DOES NOTHING !!!` in Silero's own docstring for the
  same reason: the reference implementation hardcodes 512/256 regardless
  of what's passed. This is a real divergence from `SpeechToText`, which
  explicitly tolerates arbitrary chunk sizes (it buffers internally) — VAD
  can't do the equivalent buffering *inside* `is_speech()` without either
  delaying its answer (stale bool while a partial frame accumulates) or
  silently re-chunking behind the caller's back. Resolution:
  `SileroVAD.is_speech()` validates its input is exactly `frame_bytes`
  (a documented, discoverable attribute) and raises `ValueError`
  otherwise — the caller re-chunks. That caller is
  `Orchestrator.handle_voice_turn` (specifically its `_FrameBuffer` helper
  and `_pump_mic`), not the VAD backend — see "Voice-turn state machine"
  below. 512 samples (32ms) at 16 kHz also happens to be a perfectly fine
  chunk size for `FasterWhisperSTT` (which accepts anything), so
  standardizing mic-capture chunking on that value (as
  `companion/audio/capture.py`'s `MicCapture` does) satisfies both
  backends without either dictating the other's format.
- **`is_speech()` debounces internally**, close to Silero's own
  `VADIterator` reference implementation (verified against its source):
  raw per-frame `probability >= threshold` flickers on brief sub-threshold
  dips (e.g. between syllables), so `is_speech()` requires
  `min_speech_duration_ms` of *consecutive* speech frames before onset
  registers, and once speaking, requires `min_silence_duration_ms` of
  *sustained* silence before offset registers — a single frame of
  recovered speech cancels a pending silence countdown outright rather
  than just pausing it, matching `VADIterator`'s asymmetric behavior. This
  means the orchestrator can react to raw transitions in `is_speech()`'s
  returned bool without re-implementing debouncing itself — see below.
  `VADIterator` itself wasn't reused directly because its API returns
  start/end event dicts, which doesn't fit `is_speech() -> bool`'s
  contract; the debounce *logic* was reimplemented to match its behavior
  instead of inventing a different policy.

## Voice-turn state machine (implemented)

`Orchestrator.handle_voice_turn` (`companion/core/orchestrator/orchestrator.py`)
is a long-running async generator: given a continuous stream of mic audio
chunks, it runs an entire voice *session* — potentially many turns — not
just one exchange, yielding TTS playback audio chunks as they're
produced. States:

```
                    ┌─────────────┐
        ┌──────────▶│  listening  │◀────────────────────────┐
        │           └─────────────┘                          │
        │                  │ VAD: speech frame                │
        │                  ▼                                  │
        │           ┌─────────────┐                           │
        │           │    user     │                           │
        │           │  speaking   │ (STT accumulating)        │
        │           └─────────────┘                           │
        │                  │ VAD: sustained silence           │
        │                  │ (min_silence_duration_ms)         │
        │                  ▼                                  │
        │           ┌─────────────┐                           │
        │           │ processing  │  PromptBuilder.build()    │
        │           │             │  -> LLMBackend.generate_  │
        │           │             │     stream()               │
        │           └─────────────┘                           │
        │                  │ first sentence ready              │
        │                  ▼                                  │
        │           ┌─────────────┐                           │
        │           │  speaking   │  TTS.synthesize_stream()  │
        │           │  response   │  per sentence, yielding    │
        │           │             │  audio chunks               │
        │           └─────────────┘                           │
        │            │           │                              │
        │  VAD: speech            all sentences                │
        │  during playback        synthesized & yielded         │
        │  -> TTS.stop()           (no interruption)            │
        │            │           │                              │
        └────[interrupted]  [finished]────────────────────────┘
```

Both exits from "speaking response" go back to "listening" — this is a
loop, not a one-shot call; `handle_voice_turn` only returns for real when
the mic stream itself ends (`StopAsyncIteration`).

### A real concurrency bug found while implementing this, and the fix

Barge-in requires watching for speech *while* TTS is streaming — i.e.
two things happening at once: producing response audio, and consuming
more mic input. The first implementation used a background
`asyncio.create_task` per response that independently called
`mic_audio.__anext__()` to watch for barge-in, cancelled via
`watcher.cancel()` once a response finished normally. This is broken, and
it was caught empirically, not by inspection: **cancelling a task while
it's suspended mid-`await` inside a shared async generator's
`__anext__()` call kills that generator permanently** — confirmed with a
minimal repro (cancel a task awaiting `agen.__anext__()`; every
subsequent read on `agen`, even from an unrelated task, raises
`StopAsyncIteration` forever after, regardless of items conceptually
"left"). Since the *next* listening phase reads from the same
`mic_audio` iterator, every session died silently after its first
response — indistinguishable from "the mic stream ended."

The fix: exactly one long-lived task (`Orchestrator._pump_mic`) owns the
*only* `mic_audio.__anext__()` calls for the whole session, pushing
`(stream_ended, transcript, is_speech)` onto an `asyncio.Queue`. The
listening loop and the per-response barge-in watcher both consume that
queue instead of the raw iterator. `asyncio.Queue.get()` is safe to
cancel — cancelling one waiter doesn't corrupt the queue for the next
one — so the per-response watcher task can be freely created and
cancelled every cycle, exactly as originally intended, without ever
touching `mic_audio` directly. `_pump_mic` itself is only cancelled when
`handle_voice_turn` as a whole is torn down.

### Known limitations (not solved here, deliberately)

- ~~**Barge-in stops being watched for once response audio has finished
  *yielding*, not once it's finished *playing*.**~~ **Fixed since:**
  `handle_voice_turn`'s `on_playback_interrupt` / `wait_for_playback_drained`
  hooks, driven by `SpeakerPlayback`, keep the watcher alive through real
  playback. See "The real playback-latency barge-in fix" under "Voice mode
  audio I/O" below.
- **`VoiceActivityDetector` has no `frame_bytes` in its abstract
  interface** — only concrete `SileroVAD` exposes one. `handle_voice_turn`
  duck-types it (`getattr`) and fails loudly if missing, rather than
  promoting it to the abstract interface on the strength of one
  implementation. If a second VAD backend needs this too, that's the
  point to reconsider.

## LLM (implemented)

`companion/backends/llm/ollama_local.py`'s `OllamaLLM` talks to a local
Ollama server via the `ollama` Python package's `AsyncClient`. Two things
worth knowing:

- **`context: list[Turn]` maps directly onto Ollama's chat message list**
  (each `Turn.role`, already one of "user"/"assistant"/"system", becomes
  `{"role": ..., "content": ...}`), with `prompt` prepended as the system
  message. This is a real constraint on what PromptBuilder may put in
  `context`: every turn's role must be something Ollama's chat API accepts
  as a message role. See `docs/PROMPT_BUILDER.md` for where this bites —
  retrieved memory doesn't get a `context`-list free pass just because
  `Turn` can hold arbitrary text.
- **Streaming errors don't surface where you'd expect.** `await
  client.chat(..., stream=True)` returns immediately without making any
  request — verified against a real server. The actual connection (and
  therefore `ConnectionError`/`ResponseError`) only happens once you
  iterate the result, and connection failures on that path surface as
  `httpx.TransportError`, not the friendlier `ConnectionError` that
  ollama-python's *non-streaming* path wraps things in. `OllamaLLM`
  catches both and re-raises as `OllamaConnectionError` /
  `OllamaModelNotFoundError` with an actionable message (`ollama pull
  <model>` for the latter) — see that module's docstring for the full
  story, including why `httpx`/`ollama` are in the `dev` extra (not just
  `llm-ollama`): so this error-translation logic has real exception types
  to test against without needing a live server.

## PromptBuilder (implemented)

`companion/core/prompt_builder/prompt_builder.py` — see
`docs/PROMPT_BUILDER.md` for the full design and the two refinements found
during implementation. Short version: it's a pure function (no
`MemoryStore` dependency — the caller fetches memory and passes it in),
`build()` returns `BuiltPrompt(prompt, context)`, persona + retrieved
memory render into `prompt`, and `recent_turns` (bounded) + the new
`user_message` (appended as the final turn) become `context` — ready to
pass straight to `LLMBackend.generate_stream(prompt, context)`.

## Persona loader (implemented)

`companion/core/persona_loader/persona_loader.py` — parses and validates
`companion/personas/*.yaml` into `Persona`, per `personas/schema.md`.
`load_persona(path)` handles one file; `PersonaLoader(directory)` scans a
whole directory once (`load_all()`) and caches the result for `get(id)`
lookups, matching how personas are meant to be used (scanned at startup,
not re-read per message).

Validation fails loudly and specifically — naming the file, the field,
and what was wrong — rather than surfacing a bare YAML parse error:
missing/empty required fields, an `id` that isn't filesystem/URL-safe, a
`voice_id` that isn't shaped `<namespace>:<value>`, unknown fields (a
typo'd field name is silently ignored otherwise), a non-list-of-strings
`example_dialogue`, and duplicate `id`s across files in the same
directory (since `get()` looks up by id). The optional agent fields are
checked for shape too: `tools` must be a list of distinct tool-like names
(lowercase, digits, `_`) and `filler_phrases` a list of non-empty strings.
Whether each tool name actually exists is checked at app startup against
`companion/tools/registry.py` (`validate_persona_tools`, called from
`app.load_personas`), which keeps this loader independent of the tools
layer.

**Deliberately not this loader's job**: validating that a `voice_id`'s
namespace (`piper:`, `coqui:`, ...) is one a given deployment's TTS
backend actually recognizes — that's each backend's own
`from_voice_id()` classmethod (`PiperTTS.from_voice_id`,
`CoquiTTS.from_voice_id`), so this loader never has to branch on backend
name. This loader only validates the shape, not the namespace.

## MemoryStore (implemented — two backends now)

Two real, config-selectable `MemoryStore` implementations exist and are
both wired into `companion/app.py`'s `build_memory`. The interface has
five methods: `query`, `write` and `delete`, plus `list_turns` (every
stored turn in a scope, oldest first) and `replace_turns` (all-or-nothing
swap of old turns for one new turn), which were added for rolling
summarization (`docs/ROLLING_SUMMARIZER.md`).

### Chroma (`companion/memory/store.py`'s `VectorMemoryStore`) — default

See docs/MEMORY_STORE.md for the full design note (storage/embedding
choice, ordering, the dedup gap's resolution, write path). Short version:

- **Chroma-backed** (`chromadb.PersistentClient`, no server), scoped one
  collection per `(user_id, persona_id)` pair for a structural
  no-cross-persona-leak guarantee, not just a filter-correctness one.
  `backend="lancedb"` is named but not implemented (same "named, deferred"
  pattern as STT's `whisper_cpp`).
- **Embeddings**: Chroma's own default (ONNX MiniLM-L6) — local,
  no torch, no extra dependency beyond `chromadb` itself.
- **`query()` returns most-relevant-first** — that's Chroma's own native
  ordering, not re-sorted; `PromptBuilder`'s ordering assumption
  (docs/PROMPT_BUILDER.md) holds for real now, not just by convention.
- **The dedup gap is resolved in the orchestrator, not inside
  `MemoryStore`**: `Orchestrator._dedup_retrieved_memory` drops any
  `query()` result already present in the caller's own `recent_turns`,
  by value equality — relies on `write()`/`query()` round-tripping
  `Turn.timestamp` exactly (verified). `MemoryStore.query()`'s abstract
  signature is unchanged.
- **Write path**: per-turn, immediate, not batched — both
  `handle_text_turn` and `handle_voice_turn` call `memory.write()` for
  the user's message and the assistant's full reply as each turn
  finishes (voice mode accumulates the streamed sentences into one
  assistant turn first — see "Voice-turn state machine" below).
  `RollingSummarizer` (`companion/memory/summarizer.py`) works on turns
  after they're already in `MemoryStore`: a background job, triggered
  after turns, that replaces the oldest raw turns with one summary turn
  (via `list_turns`/`replace_turns`). See `docs/ROLLING_SUMMARIZER.md`.
- **`delete()` is implemented for real** (it was already part of the
  abstract interface, not deferred) — per-persona or whole-user wipe.
  Automatic pruning/expiry policy and a UI affordance to call `delete()`
  are real future work, not addressed here.

### MemArk (`companion/memark/retrieval.py`'s `MemArkMemoryStore`) — experimental, wired

The full MemArk effort (`docs/MEMARK_BACKEND.md` → `docs/
MEMARK_COMPANION_EXTENSIONS.md` → `docs/MEMARK_PERSISTENCE.md` → `docs/
MEMARK_RETRIEVAL_ADAPTER.md`) graduates here from a series of design
notes to a real, shipped, `config.yaml`-selectable second `MemoryStore`
backend — `memory.backend: memark` in place of `chroma`, chosen once at
startup exactly like every other backend swap in this project (see "App
bootstrap" below); no per-session picker (`docs/MEMARK_BACKEND.md`
section 3 scoped that out explicitly — it needs a `Conversation` concept
this project doesn't have).

- **Hybrid graph + recency/keyword retrieval** over a local, in-process
  `GraphMemoryManager` (vendored MemEngine source, not a `memark` pip
  dependency — see `companion/memark/_vendor/PROVENANCE.md`), persisted
  to one SQLite file per `(user_id, persona_id)` via
  `companion/memory/memark_store.py`'s `MemArkGraphStore` and hydrated
  back into a live graph once per scope, on first use.
- **Same `MemoryStore` contract, same `Orchestrator`/`PromptBuilder`
  wiring** — `docs/MEMARK_BACKEND.md`'s Option A decision (force-fit
  MemArk's node ranking into `list[Turn]` at the adapter boundary, touch
  nothing upstream) holds; neither interface nor orchestrator changed to
  accommodate this backend, only `Orchestrator._write_turn_safely`'s
  generic `MemoryStoreError` handling (below) benefits it too.
- **Write-failure handling is shared, not MemArk-specific**:
  `MemArkPersistenceError` (raised by `MemArkGraphStore` on a corrupted
  or unreadable SQLite file) subclasses the interface-level
  `MemoryStoreError`, so `Orchestrator._write_turn_safely` catches it
  generically — `Orchestrator` never imports anything MemArk-specific.
  See "Voice-turn state machine" / `docs/MEMARK_RETRIEVAL_ADAPTER.md`'s
  "Resolved in a later pass" note for the full end-the-turn-only decision
  and why (b)/(c) were rejected.
- **Honest, disclosed tradeoffs, not just "an alternative"**:
  `docs/MEMARK_BACKEND.md` section 5 found this backend trades raw
  recall for context cleanliness on MemArk's own published benchmarks,
  and section 4 flagged salience decay (old memories stop surfacing by
  design) as a real behavioral difference from Chroma's "everything
  stays retrievable forever." Neither is a bug; both are why this ships
  labeled experimental in `config.example.yaml`, not as the new default.
- **Import cost stays opt-in**: `companion/memark/retrieval.py` imports
  the vendored `engine.*` modules (→ `networkx`/`faiss`) at module level,
  unlike every other backend's lazy-inside-a-method pattern — so
  `build_memory` imports `MemArkMemoryStore` only inside its `memark`
  branch, not unconditionally alongside `VectorMemoryStore` and the other
  backend classes at `app.py`'s module top. A `chroma`-only install never
  needs the `memory-memark` extra; selecting `memark` without it fails
  right at startup with a plain, unwrapped `ImportError` — the same
  "don't invent a new error style" convention every other optional-extra
  backend in this project already follows (see `build_memory`'s own
  comment in `companion/app.py`).
- **`delete()` is implemented**, matching `VectorMemoryStore.delete()`'s
  contract exactly (`persona_id=None` wipes every scope for that
  `user_id`) — added in a later pass than the original retrieval adapter
  (`docs/MEMARK_RETRIEVAL_ADAPTER.md` shipped query()/write() only).
  Clears two things, not one: the persisted SQLite file(s) via a new
  `MemArkGraphStore.delete()` (closes any cached connection first, then
  removes the `.db` file and its WAL/SHM sidecars) and this adapter's own
  in-memory `_scopes` cache entry — dropping the whole cached
  `GraphMemoryManager`/`FaissVectorStore` object for a deleted scope
  rather than surgically clearing it, since `GraphMemoryManager` has no
  supported single-node removal API either (same finding
  `docs/MEMARK_RETRIEVAL_ADAPTER.md` section 8 made about `write()`'s own
  failure-rollback question).

### Rolling summarization (`companion/memory/summarizer.py`)

Keeps long relationships with a persona from filling memory with raw
turns. Full design: `docs/ROLLING_SUMMARIZER.md`.

- **What it does:** once a `(user, persona)` scope holds
  `memory.summarizer.trigger_turn_count` raw user/assistant turns
  (default 50; 0 or missing = off), the oldest half are summarized by the
  LLM into one third-person summary turn (`role="system"`, timestamped at
  the newest turn it replaced) and swapped in through `replace_turns`.
  Summary turns aren't counted and are never re-summarized.
- **Never on the request path:** `Orchestrator` (CLI text and voice)
  schedules `summarize_if_due` as a background task after each turn's
  memory writes, at most one per scope at a time; failures are logged and
  never reach the conversation. Streamlit runs each turn under its own
  `asyncio.run`, so it builds its `Orchestrator` without a summarizer and
  calls it itself after the reply has rendered (on a due turn, the next
  message can wait a few seconds).
- **Safe against deletes:** `replace_turns` changes nothing and returns
  False if any of the old turns is gone (e.g. `delete()` ran meanwhile),
  so a summary can't resurrect erased memory. MemArk does the swap in one
  SQLite transaction, under a store-wide lock that also covers `write()`
  and `delete()` (the summarizer made concurrent writes possible).
  Chroma has no transactions, so it adds the summary before deleting the
  old turns: a crash in between leaves duplicates, never a gap.
- **Its own LLM instance** (`app.build_summarizer`), always with the base
  `llm.options`, never the voice override, whose short-reply settings
  would truncate a summary.
- **The prompt lives in summarizer.py, not PromptBuilder:** it assembles
  no persona, memory or conversation window, which is what PromptBuilder
  is for.
- **Verified** with a fake LLM against real Chroma and MemArk stores,
  including across processes. Summary *quality* with a real model is not
  measured.

### Persistence across real process restarts (both backends)

`tests/test_memory_persistence_across_processes.py` runs each step in a
separate short-lived process (`tests/_persistence_child.py`) against
one shared directory, so nothing can carry over in process memory. Same-
process "fresh instance" tests (as in `tests/test_memark_retrieval.py`)
can pass because of module-level caches or open SQLite connections;
these can't. It covers, for both `chroma` and `memark` and built through
`companion.app.build_memory` as the app does:
- write in one process, query in the next, with exact `Turn` equality
  including timestamps (the orchestrator's dedup depends on it), persona
  scoping, and deletes staying deleted;
- `Orchestrator.handle_text_turn`: a fact from session one appears in
  session two's LLM prompt, with an empty `recent_turns` (recording fake
  LLM, real memory, real `PromptBuilder`);
- two real `python -m companion` text-mode runs against local Ollama,
  where the second must answer from the first's memory (skips without
  Ollama). Run for real on 2026-10-19 with `dolphin-llama3:8b`: both
  backends answered correctly. With an empty memory directory, the same
  question got "New York City", so the answer really comes from memory.

## App bootstrap (implemented, both modes)

`companion/config/loader.py` (`load_config`, `resolve_config_path`) +
`companion/app.py` (backend construction + the text loop, shared by both
modes) + `companion/voice_app.py` (real audio I/O + the voice loop,
voice-mode-only) + `companion/__main__.py` (thin CLI shim, dispatches on
`config.mode`) — `python -m companion` is a real, runnable conversation
loop in either mode. See README.md's "Running the app" for the commands,
and "Voice mode audio I/O" below for the voice-specific pieces.

- **Config loading is a separate, narrower-scoped validator than the
  persona loader.** `load_config` validates *structure* (required
  sections, `backend`/`options` shape, `tts.voices_dir`, `mode`) but,
  unlike the persona loader, does not reject unrecognized keys — an
  `options:` block is inherently backend-defined and meant to grow, and
  config.example.yaml ships keys the loader only passes through. A bad
  `options:` key surfaces naturally as a `TypeError` from the backend's
  own constructor instead — not re-validated redundantly here.
- **This is the one place backend-name branching is expected**, per
  CLAUDE.md's architecture rule ("the code that reads
  config.example.yaml" is where `if backend_name == "piper":` belongs).
  `build_stt`/`build_llm`/`build_vad` reject any name but the one real
  implementation each has; `build_tts_for_persona` branches for real
  between `piper`/`coqui`, since both exist and their `from_voice_id()`
  signatures differ (`voices_dir` vs. `model_name`); `build_memory` now
  branches for real too, between `chroma` (`VectorMemoryStore`) and
  `memark` (`MemArkMemoryStore`) — see "MemoryStore" above for why the
  `memark` import specifically stays inside that one branch rather than
  living alongside the other backend imports at module top.
- **TTS is resolved per-persona, not globally** — `build_tts_for_persona`
  calls the selected backend's `from_voice_id(persona.voice_id, ...)`
  once, after a persona is selected, per the "Persona voice binding" note
  above.
- **Persona selection**: in the CLI, `select_persona()` takes an
  explicit `--persona <id>` if given, else the first persona
  `PersonaLoader.load_all()` returns (alphabetical by filename); the
  Streamlit UI has a sidebar picker. Memory's
  `user_id` similarly has no auth system behind it (out of scope per this
  project's brief) — `--user-id` defaults to `"local"`.
- **`llm.voice_mode_override`** (a gap found while wiring real backends,
  since closed): the config loader parses `llm.voice_mode_override.options`
  into `AppConfig.llm_voice_override`, and `build_llm` merges them over
  `llm.options` when `mode: voice`. Mode is fixed for a whole session, so
  this is decided once at construction; `Orchestrator` still holds a
  single `llm` with no mode branch. Only `options` can be overridden: a
  `backend` key there is rejected, since switching LLM backends by mode
  isn't supported. The Streamlit UI's push-to-talk goes through
  `handle_text_turn` and uses the base options. The example config ships
  the override commented out (since 2026-10-20), so voice mode uses the
  main 7B model; its comment suggests the Apache-2.0
  `qwen2.5:1.5b-instruct` for CPU-only machines. It previously named
  `qwen2.5:3b-instruct`, which turned out to be research-licensed (see
  THIRD_PARTY_LICENSES.md).
- **Optional pieces, each off when its config is absent:**
  `build_agent` (the `agent:` section; personas with tools),
  `build_summarizer` (`memory.summarizer.trigger_turn_count`), and
  `take_due_reminders` (polled by the text loop, the voice loop and the
  Streamlit UI whenever an agent is configured). `load_personas` also
  checks every persona's tool names against the tool registry, so a typo
  fails at startup rather than mid-conversation.
- **Errors are actionable, not raw tracebacks**: `__main__.py`'s `main()`
  wraps the whole run in one `except Exception`, printing `str(exc)` and
  exiting 1 (`--debug` re-raises for a full traceback) — this works
  *because* every lower layer already raises exceptions with clear,
  specific messages (`ConfigError`, `PersonaError`,
  `OllamaConnectionError`/`OllamaModelNotFoundError`, this module's own
  `AppSetupError`), not because `__main__.py` special-cases each one.
- **Voice mode is real too now** — `config.mode: voice` dispatches to
  `companion/voice_app.py`'s `run_voice_session` instead of text mode's
  `run_text_loop`; both share the exact same `build_orchestrator`/
  `load_personas`/`select_persona` construction path above. See "Voice
  mode audio I/O" below for the mic-capture/speaker-playback pieces this
  needed and the real playback-latency barge-in fix that came with them.

## Voice mode audio I/O (implemented)

`companion/audio/capture.py` (`MicCapture`) and
`companion/audio/playback.py` (`SpeakerPlayback`), backed by
[sounddevice](https://python-sounddevice.readthedocs.io/) (PortAudio
bindings) — picked over PyAudio: actively maintained, ships prebuilt
wheels (PortAudio itself included) for Windows/macOS/manylinux, so it
doesn't need system PortAudio headers the way PyAudio's sdist build
does; its `Raw*Stream` classes work with raw bytes directly, no numpy
dependency just for audio I/O. Nothing already in this project provided
mic/speaker I/O — faster-whisper/piper/coqui/silero all consume/produce
raw PCM bytes but none of them touch an audio device. Config-driven, same
pattern as every other backend: `audio.backend: sounddevice` (named but
not implemented for anything else, same "named, deferred" pattern as
STT's `whisper_cpp`), `audio.options.{input_device,output_device,
sample_rate,channels}`.

- **`MicCapture` is directly an `AsyncIterator[bytes]`** — pass an
  instance straight in as `handle_voice_turn`'s `mic_audio` argument, no
  adapter needed. Uses sounddevice's callback-based streaming (PortAudio
  hands it buffers as they arrive) bridged onto an `asyncio.Queue` via
  `call_soon_threadsafe`; doesn't need to know `SileroVAD`'s exact
  frame-size requirement — the orchestrator's existing `_FrameBuffer`
  already re-chunks whatever size this emits, exactly as it does for any
  mic source.
- **`SpeakerPlayback` uses blocking `write()` on a background thread**,
  not a raw PortAudio callback — verified directly that a real device's
  blocking `write()` call for 1.0s of audio takes ~1.03s wall-clock, so
  it naturally paces close to real-time without needing PortAudio's
  realtime-callback API (awkward to bridge cleanly from an async
  producer). Tracks a "pending chunks" count to know when playback has
  actually drained, not just when chunks have been handed over — see
  below for why that distinction is the whole point of this module.
- **Config validation for `audio` is structural only, same as every
  other section** (`companion/config/loader.py` doesn't reject unknown
  keys or cross-check option *contents*) — but `audio.options.sample_rate`
  matching `vad.options.sample_rate` is a hard runtime requirement
  (Silero's frame-size math depends on it), so `companion/voice_app.py`
  checks that explicitly at startup, before opening any device, with a
  clear message naming both values — not a downstream `SileroVAD`
  frame-size `ValueError` that doesn't explain *why* the sizes are wrong.
- **Device/permission errors are wrapped, not raw**: any
  `sounddevice.PortAudioError` (invalid device index, unsupported sample
  rate, no device available at all — e.g. headless/CI, device
  busy/permission-denied) is caught at stream-construction time and
  re-raised as `companion.audio.errors.AudioDeviceError` with the
  attempted device/rate and a pointer to `python -m sounddevice` for
  listing what's actually available.

### The real playback-latency barge-in fix (corrects a previously-documented limitation)

The "Voice-turn state machine" section above used to document a known,
unsolved limitation: barge-in was only watched for while
`handle_voice_turn` was still *yielding* response audio chunks, and
synthesis+yielding can finish well before a speaker has actually
finished *sounding* that audio (playback has its own buffering latency).
That was flagged as needing "the caller to signal 'done playing,' which
doesn't exist yet" — theoretical until this pass gave `handle_voice_turn`
a real caller that actually plays audio and can observe that lag
directly.

**Resolved with two new optional parameters on `handle_voice_turn`**:
`on_playback_interrupt: Callable[[], None] | None` and
`wait_for_playback_drained: Callable[[], Awaitable[None]] | None`. Both
default to `None`, in which case `handle_voice_turn` behaves exactly as
it always has — this is *not* a change to an abstract interface (neither
parameter touches `core/interfaces/`, and `handle_voice_turn` is a
concrete `Orchestrator` method, not one of the five ABCs), and every
existing test/caller is unaffected by the addition. `companion/voice_app
.py`'s `run_voice_loop` is the real caller that supplies both, wired to
`SpeakerPlayback.stop`/`SpeakerPlayback.wait_drained`.

Mechanics: when real barge-in is detected, `on_playback_interrupt()` is
called synchronously alongside the existing `self.tts.stop()` — the
caller's chance to immediately discard audio that's queued for playback
but hasn't sounded yet (`self.tts.stop()` alone only ever stopped
*synthesis*; it was never wired to reach into a downstream playback
buffer, and still isn't — that stays this new parameter's job). After a
response finishes *yielding* normally (i.e. no barge-in occurred during
generation), `handle_voice_turn` now awaits
`wait_for_playback_drained()` *before* cancelling the barge-in watcher —
so the watcher stays alive through the real playback tail, not just the
synthesis+yield window, and can still catch a late barge-in during that
tail. The contract: `wait_for_playback_drained()` must resolve once the
caller has nothing left to actually play, whether that's because
playback finished naturally or because `on_playback_interrupt()` just
flushed it — `SpeakerPlayback` satisfies this by tracking a pending-chunk
count that both paths correctly zero out.

Verified as a real regression test (`tests/test_orchestrator_voice_turn
.py::test_late_barge_in_during_playback_drain_is_still_caught`), fakes
only: a barge-in that arrives *after* a response has fully finished
synthesizing/yielding — simulating real playback still sounding out the
tail — still registers, because the watcher is still alive during the
simulated drain-wait. Confirmed this is genuinely load-bearing (not a
vacuously-true test) by running the same scenario with
`wait_for_playback_drained` omitted: the late barge-in is missed, exactly
reproducing the old documented gap.

**Honest limit, not zero**: `SpeakerPlayback.stop()` can discard chunks
still sitting in its queue, but not one the worker thread has already
started `write()`-ing (PortAudio's blocking write isn't cancellable
mid-call) — at most one chunk's worth of residual audio (tens to a few
hundred milliseconds, depending on chunk size), not the multi-second gap
this fix closes. See `companion/audio/playback.py`'s module docstring for
the full precision caveat on what "drained" means.

### A real deadlock found via the first real, continuous mic round trip

The first genuine end-to-end run against a real (non-scripted-frame-list)
mic source hung permanently after the first utterance's response
finished — not slow, actually stuck (confirmed via CPU time sampled
minutes apart on the hung process, which hadn't moved). Root cause: the
`(stream_ended, None, None)` sentinel `_pump_mic` puts onto the shared
`events` queue when the mic source is exhausted is a **one-shot** value,
but it has two possible readers across a session — the main listening
loop, and each response's `watch_for_barge_in` task — that alternate
(never concurrently active, but either can be the one blocked on
`events.get()` at a given moment). Whichever one reads it first "spends"
it. If that's the watcher — exactly what happens when an utterance's
trailing silence leaves several queued events for the watcher to drain
before it reaches `stream_ended` — the listening loop's *next* iteration
blocks on `events.get()` forever, since `_pump_mic` (the sole producer)
has already returned and nothing will ever fill the queue again.

**Fix**: the signal is now self-propagating. `watch_for_barge_in`'s
`stream_ended` branch re-`put()`s it back onto the queue before
returning, so whichever reader encounters it next still sees it. Only
the watcher needs this — the listening loop's own `stream_ended` branch
already ends `handle_voice_turn` for good, so nothing reads after it.

This also forced an explicit semantic decision: mic-ended and real
barge-in are *not* the same event and shouldn't share a response. Real
barge-in means the user is actively talking over the persona right now —
cut the response short immediately. Mic-ended means nothing will ever
arrive again, which is not the user interrupting anything (for a live
`MicCapture` this essentially never happens mid-session; it's reachable
mainly via finite sources like test fixtures or a future replay mode).
Treating it as a barge-in would truncate a response mid-sentence for a
reason unrelated to the user. So `watch_for_barge_in` does **not** set
`interrupted` on `stream_ended` — it just stops watching and returns,
letting the in-flight response finish generating and playing normally;
`wait_for_playback_drained` still runs afterward exactly as it would with
no watcher at all. Only the *next* listening iteration's `stream_ended`
(now correctly re-queued) actually ends the session.

Regression test:
`tests/test_orchestrator_voice_turn.py::test_mic_ending_naturally_after_response_does_not_hang`
— a mic fixture shaped like the real repro (speech, then silence, then
nothing — no trailing barge-in frame, unlike every other fixture in that
file) wrapped in `asyncio.wait_for` so a reintroduced regression fails
fast with a clear timeout instead of hanging the suite. See
`Orchestrator._pump_mic`'s docstring for the same explanation in-code.

### Testing reality check (what's actually verified, and how)

Real mic/speaker I/O is inherently hard to meaningfully unit-test in CI.
Split honestly, not blurred:

- **Automated, offline (no hardware, no `audio-sounddevice` extra
  needed)**: `tests/test_audio_capture.py`/`test_audio_playback.py` test
  `MicCapture`/`SpeakerPlayback`'s own logic (callback-to-queue bridging,
  the pending/drained accounting, `stop()`'s discard-vs-in-flight
  distinction, `PortAudioError` -> `AudioDeviceError` wrapping) against a
  fake `sounddevice` module injected into `sys.modules` — not a real
  device. `tests/test_voice_app.py` tests `run_voice_loop`'s *wiring*
  (every yielded chunk reaches `playback.play()`; `playback.stop`/
  `playback.wait_drained` are the exact callables passed to
  `handle_voice_turn`) against a duck-typed recording stand-in, not a
  real `Orchestrator` or real devices — `handle_voice_turn`'s own
  state-machine correctness is already covered where it belongs, in
  `tests/test_orchestrator_voice_turn.py`.
- **Automated, gated on real hardware** (`audio-sounddevice` extra
  installed, skipped cleanly otherwise — same pattern as every other
  real-backend smoke test): `test_real_mic_capture_smoke`/
  `test_real_speaker_playback_smoke` open the actual default input/output
  device and confirm real bytes come back / a real `write()` completes.
  This confirms the real PortAudio integration *runs* — it does not, and
  cannot, assert that captured audio is meaningful or that played audio
  is audible/correct to a listener.
- **Verified by hand, not by an automated test**: real capture of actual
  room audio from a physical microphone (confirmed non-silent PCM
  samples), real playback of a synthesized tone through a physical output
  device with `write()` timing matching real-time (~1.03s for 1.0s of
  audio — the empirical basis for `SpeakerPlayback`'s "drained" design),
  and a full real round-trip voice conversation exercising real VAD/STT/
  LLM/TTS/MemoryStore together with real audio hardware — a real,
  continuous mic source (not a hand-scripted frame list), which is
  exactly what first surfaced the deadlock documented above. Confirmed
  again after that fix: the same round trip completed cleanly end to end
  (mic exhausted, response generated and played, loop exited, session
  memory written), with no hang. The explicit, unavoidable limit stands
  either way: this was run and inspected programmatically (chunk counts,
  byte lengths, real-time write duration, memory contents), not confirmed
  by a human listening — nothing in this project has ears.

## Agentic personas (implemented)

A persona that lists `tools:` (companion/personas/schema.md) runs as an
agent: on each turn a tool-calling loop decides whether to use its tools.
Design note, including the reasoning this follows: `docs/AGENTIC_TOOLS.md`.

**Pieces, by layer:**
- `companion/tools/`: the tools, as plain async functions with no
  LangChain dependency (`base.py`'s `Tool`/`ToolContext`). `registry.py`
  maps each persona-facing name to a group of small tools, e.g. `notes` ->
  `add_note`/`search_notes`/`delete_note`; 7B models call several
  single-purpose tools more reliably than one tool with modes. Offline:
  `get_datetime`, `calculator` (AST-based, no `eval`, capped exponents),
  `notes`, `reminders`, `recall_memory`. Online, only with
  `agent.options.allow_network: true`: `web_search` (ddgs) and `weather`
  (Open-Meteo); both need no API key.
- `companion/tools/store.py`: notes and reminders in one SQLite file
  (`<agent data_dir>/tools.sqlite3`). Notes are scoped per (user,
  persona), like memory. Reminders are scoped per user: any persona
  delivers a due one, and it says which persona set it.
- `companion/core/agent/`: the `AgentRunner` protocol `Orchestrator`
  depends on, plus the event types (`TextDelta`, `ToolStarted`,
  `ToolFinished`). A `typing.Protocol`, not a sixth abstract interface,
  for the reason AGENTIC_TOOLS.md §1 gives.
- `companion/backends/agent/langgraph_ollama.py`: the one implementation,
  `LangGraphOllamaAgent`, on `langchain.agents.create_agent` (LangGraph
  1.x deprecates `create_react_agent` in its favour) with `ChatOllama`.
  Tool failures (bad arguments, timeouts, crashes) go back to the model
  as an "Error: ..." result instead of failing the turn; `max_steps`
  bounds the loop and ends in an apology, not an exception.
- `companion/app.py`: `build_agent` (from config's optional `agent:`
  section), tool-name validation in `load_personas` (an unknown name fails
  at startup, naming the persona), and `take_due_reminders`.

**Orchestrator.** One optional constructor argument, `agent`. A persona
the agent `handles()` (at least one tool enabled under this config) goes
through it; every other persona, and every persona when there's no
`agent:` section, behaves exactly as before. The agent gets exactly
`PromptBuilder`'s output, so persona, memory and recent turns reach it the
same way they reach the plain LLM, and PromptBuilder stays the only place
prompts are assembled. Memory receives the user message and the final
answer only, never tool calls or tool output.

**Voice mode** (`_sentences_from_agent`):
- The agent stream is read by one background task into a private queue,
  so no async generator is shared between readers. That's the rule the
  `_pump_mic` deadlock section above is about.
- The filler phrase is spoken when the model actually calls its first
  tool (`ToolStarted`), not guessed in advance by a keyword heuristic.
- The wait also wakes on barge-in, so interrupting during the silent gap
  after the filler ends the turn at once; closing the sentence source
  (`contextlib.aclosing`) cancels the agent task, so a late tool result
  can't arrive after the user moved on.
- If the agent produces nothing for `agent_voice_timeout_s` (default
  20 s), a short caveat is spoken and the plain LLM answers instead.
- Fillers, the caveat and announcements (due reminders, spoken before the
  next reply via `handle_voice_turn`'s `pending_announcements` hook) are
  spoken but never written to memory.

**Reminder delivery.** Text CLI: a background task polls every 20 s, so a
reminder appears even while you sit at the prompt. Streamlit: an
`st.fragment(run_every=20)` shows it as a toast. Voice: printed, and
spoken before the next reply. Each reminder is delivered exactly once
(`take_due_reminders` selects and marks delivered in one transaction).

**Model.** Needs Ollama tool-calling support (`ollama show <model>` lists
"tools"). Default `qwen2.5:7b-instruct`. `dolphin-llama3:8b` has no tool
support. `qwen2.5-coder:7b` does in principle, but in the spike below it
wrote its tool calls as plain JSON text and called a tool even for an
advice question, so it isn't recommended.

**Streaming with tools bound (the AGENTIC_TOOLS.md §1 question).**
Measured before building on it (2026-10-19; LangGraph 1.2, langchain-ollama
1.1, Ollama 0.24, `qwen2.5:7b-instruct`): with tools bound, answer tokens
still stream one by one (48 chunks over ~0.8 s for a no-tool answer). A
tool call arrives as a single chunk with no text, at ~0.4–0.9 s. The
streaming bug AGENTIC_TOOLS.md §1 cites (langgraph#5249) didn't reproduce
here, so the keyword pre-filter designed there isn't used: for a persona
with tools, the model decides on every turn whether to call one. If a
future model or library version brings that bug back, the pre-filter in
§1 is the fallback.

**Tests.**
- `tests/test_tools.py`: every tool, with a fixed clock, temporary SQLite
  and mocked HTTP.
- `tests/test_agent_langgraph.py`: the real LangGraph loop with a
  scripted fake chat model, plus a real-qwen smoke test that skips
  without the model.
- `tests/test_orchestrator_text_turn.py` / `test_orchestrator_voice_turn.py`:
  routing, memory contents, filler, barge-in during the tool wait, the
  stall fallback, error propagation, announcements.
- `tests/test_app.py`, `tests/test_streamlit_app.py`,
  `tests/test_voice_app.py`: wiring and reminder delivery.
- `scripts/check_offline.py`: one real tool call under the network guard.

## Open decisions

- **License**: this repo's own code is MIT (`LICENSE`,
  `pyproject.toml`). That does **not** extend to bundled/downloaded model
  weights — see `THIRD_PARTY_LICENSES.md`, kept up to date as each backend
  is integrated. XTTS-v2 in particular ships under Coqui's non-commercial
  CPML; don't enable it as a default voice in anything commercial without
  reading that file first.
- **UI framework**: no longer fully open — `companion/ui/streamlit_app.py`
  is a real, working Streamlit chat UI, built against the same
  `build_*`/`Orchestrator` construction `companion/app.py`'s CLI text
  loop uses. Text chat plus optional **push-to-talk voice** — a browser
  mic recording (`st.audio_input`) transcribed one-shot through the same
  `stt: SpeechToText` (`orchestrator.stt.reset()` + one
  `transcribe_chunk()` call, not the incremental buffered path
  `handle_voice_turn` uses), then the transcript runs through the exact
  same `handle_text_turn` a typed message does; an optional "Speak
  replies" toggle synthesizes the reply through the same
  `tts: TextToSpeech` and plays it via `st.audio`. Not the CLI's
  real-time hands-free voice mode (see its module docstring for why that
  doesn't fit Streamlit's per-interaction rerun model without
  substantially more plumbing) — no new abstract-interface surface,
  UI-layer composition of the existing STT/TTS/Orchestrator contracts,
  same as `companion/app.py`/`companion/voice_app.py` already do.
  Verified for real: `streamlit.testing.v1.AppTest` against a real
  config, real Ollama, and real Chroma for the text path (persona
  selection, a real chat round trip, persona switching, and the
  clear-chat-window control — the last one caught and fixed a real bug
  this way: the button originally popped `recent_turns` from session
  state instead of resetting it, which crashed the next rerun with a
  `KeyError`); `AppTest` has no way to simulate a browser mic recording,
  so the voice path's own functions (`_wav_bytes_to_pcm16_mono`,
  `_transcribe_clip`, `_synthesize_reply`) were instead verified directly
  against real audio — a real Piper-synthesized sentence, deliberately
  resampled to a different rate to exercise the conversion path, wrapped
  in a real WAV container, decoded and fed through real
  `FasterWhisperSTT`, returning the original sentence correctly. The
  actual browser-mic-click-record UI path itself is *not* something this
  verified — same "can't claim tested without a real browser/mic" honesty
  standard `companion/audio/`'s own testing-reality-check already holds
  the CLI's real mic/speaker code to. Not treated as the final or only UI
  this project will ever have — see `companion/ui/README.md`. Since
  2026-10-19 the agentic personas show here too: a "Tools: ..." caption,
  a "Used tools" status box per reply, and reminder toasts via an
  `st.fragment(run_every=20)` poll.
- **Next UI (decided 2026-10-19, not started): a React frontend.** It
  needs a local API server first (HTTP plus a WebSocket for streaming
  text and audio), since Streamlit calls `Orchestrator` in-process and a
  browser app can't. That server is also what makes hands-free real-time
  voice in the browser possible. It stays local, with no auth or cloud,
  and should be designed so the separate deployed version can reuse the
  same protocol. What has to happen first:
  1. Manual testing of the current build (`DECISIONS.md` D28).
  2. The user decides the open choices: frontend framework and build
     tool, state management, API shape, and the server library (none
     chosen yet; don't pick unilaterally).
  3. The server reuses `companion/app.py`'s `build_*` /
     `build_orchestrator` and calls `Orchestrator` exactly as the CLI and
     Streamlit do, rather than a new construction path. One thing the
     Streamlit app does itself has to move into shared code the server
     can call: running the summarizer after a reply. Reminder polling
     (`take_due_reminders`) is already shared.
  4. The voice path needs its own design: streaming browser mic audio
     into `handle_voice_turn` over the WebSocket, with playback hooks
     standing in for `SpeakerPlayback`.
