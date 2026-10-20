# OpenPersona

An open-source, local-first AI companion. (The Python package and CLI
are named `companion` — `python -m companion`.) Talk (voice or text) with a
persona of your choice — a wise old mentor, a casual friend, a romantic
companion, a study buddy, a planner, or one the community writes — running
on local models by default, with no external server required to get
started. Personas can be **agents**: each has its own set of tools (notes,
reminders, a calculator, memory search, and optionally web search and
weather) and decides for itself when to use them.

**Status: both text and voice mode run end-to-end, for real.** STT, TTS,
LLM, VAD, the persona loader, `MemoryStore`, and real mic/speaker audio
I/O are all implemented; `python -m companion` reads config, constructs
real backends, and runs an actual conversation with memory — text or
voice, picked by `config.yaml`'s `mode:` — see "Running the app" below.
Also built:
- five agentic personas with tools (see "Agentic personas");
- rolling summarization of long-term memory (see "Memory summarization");
- a browser UI (Streamlit: text chat plus push-to-talk voice; see
  "Streamlit UI").

Measured latency on a laptop GPU is about 0.85 s from the end of your
speech to the first spoken audio (`docs/HARDWARE_REQUIREMENTS.md`). See
`docs/ARCHITECTURE.md` for the design, and the root `CLAUDE.md` for
exactly what's implemented vs. not.

## Stack (local-first phase)

- STT: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU fallback: whisper.cpp, not yet implemented)
- LLM: [Ollama](https://ollama.com), config-selectable model
- TTS: [Piper](https://github.com/rhasspy/piper) (optional higher-fidelity: [Coqui/XTTS-v2](https://github.com/idiap/coqui-ai-TTS), incl. future voice cloning)
- VAD: [Silero VAD](https://github.com/snakers4/silero-vad)
- Memory: [Chroma](https://www.trychroma.com/) (embedded, no external DB server; LanceDB named as a future alternative, not yet implemented) — an experimental MemArk graph-memory backend is also wired in, see "Memory backend" below; older turns are rolled up into summaries in the background
- Agents: [LangGraph](https://github.com/langchain-ai/langgraph) tool-calling loop over Ollama (`qwen2.5:7b-instruct` by default), see "Agentic personas" below

A deployed, cloud-hosted version is planned as a **separate project**,
not part of this repo — nothing here talks to a cloud service. See
`docs/ARCHITECTURE.md`'s "Build order."

## Running the app

### Text mode

This is genuinely runnable, end to end, with just:

```
pip install -e ".[dev,memory-chroma,agents]"
```

(STT/TTS/VAD get constructed too but text mode never calls into them, so
their extras — `stt-whisper`, `tts-piper`, `vad-silero` — aren't needed
just to talk to a persona in text mode; add them for voice mode, below.
`llm-ollama`'s dependencies are already covered by `dev`. `agents` is
needed because the example config turns agentic personas on; without it
startup stops with a clear "needs the agents extra" error. To run without
agents instead, delete the `agent:` section from your `config.yaml`.)

Then, with [Ollama](https://ollama.com) installed and running and the
default model pulled (it supports tool calling, which agentic personas
need):

```
ollama pull qwen2.5:7b-instruct
```

```
cp companion/config/config.example.yaml companion/config/config.yaml
# edit config.yaml's llm.options.model to a model you've actually pulled
# (ollama list shows what's available; the shipped default,
# qwen2.5:7b-instruct, is a recommendation, not something pre-pulled for you)

python -m companion
```

You'll be talking to whichever persona loads first from
`companion/personas/` (override with `--persona mentor`/`friend`/
`companion`/`study_buddy`/`planner`). Type `exit` to quit. Useful flags: `--persona <id>`,
`--user-id <id>` (memory is scoped per user+persona; defaults to
`local`), `--config <path>` (default: `config.yaml` if you've created
one, else `config.example.yaml` itself — so `python -m companion` with
zero setup still runs, it'll just fail with a clear error at the first
message if the default model isn't pulled), `--debug` (full traceback
instead of a one-line error message).

If something's wrong — Ollama isn't running, the configured model isn't
pulled, a persona file is malformed — you'll get a clear one-line error
telling you what to fix, not a raw traceback (unless you pass `--debug`).

### Agentic personas (tools)

| Persona | id | Tools |
|---|---|---|
| The Old Mentor | `mentor` | recall_memory, notes, reminders, get_datetime |
| Casual Friend | `friend` | recall_memory, get_datetime, weather*, web_search* |
| Romantic Companion | `companion` | recall_memory, reminders, get_datetime |
| Study Buddy | `study_buddy` | calculator, notes, recall_memory, web_search* |
| The Planner | `planner` | reminders, notes, get_datetime, calculator, weather* |

\* network tools: off unless you set `agent.options.allow_network: true`.

Things to try: ask the Planner to "remind me to stretch in 2 minutes",
the Study Buddy "what's 17% of 2,340?", or tell the Mentor to note
something, then restart and ask what notes you have. In the terminal a
"(using ...)" line shows when a tool runs; in Streamlit a "Used tools"
box does. In voice mode the persona says a short filler line ("Let me
check.") while a tool runs, and you can interrupt at any point.

Setup:

```
pip install -e ".[dev,memory-chroma,agents]"   # + tools-online for web_search/weather
ollama pull qwen2.5:7b-instruct                # needs a model with tool support
```

`config.example.yaml` already has the `agent:` section; copy it into your
`config.yaml` if yours predates it. Without that section, every persona
chats through the plain `llm` model and ignores its tools. Plain chat can
stay on any model (even one without tool support); only personas with
tools use the agent model. Notes and reminders live in `data/tools/`. See
`docs/ARCHITECTURE.md`'s "Agentic personas" section for how it works and
`companion/personas/schema.md` to give your own persona tools.

### Streamlit UI (optional, instead of the terminal)

A real browser chat window, with optional push-to-talk voice — same
config, same personas, same memory backend as the CLI above, just a
different front end onto the same `Orchestrator`. See
`companion/ui/README.md` and that file's own module docstring for how
voice fits in (push-to-talk, not the CLI voice mode's real-time
hands-free experience — Streamlit's per-interaction model doesn't fit
that without a much bigger rework).

```
pip install -e ".[dev,memory-chroma,agents,ui-streamlit]"
# (same config.yaml setup as Text mode above, if you haven't already)

streamlit run companion/ui/streamlit_app.py
```

Opens in your browser at `http://localhost:8501`. Pick a persona from
the sidebar dropdown (switching rebuilds just that persona's voice
binding, not the whole app — cheap), set a `User ID` if you want a
different memory scope than `local`, and chat. "Clear chat window"
resets only what's shown on screen, not your companion's actual
long-term memory. For a persona with tools, a "Tools: ..." caption
shows under its name, a collapsed "Used tools" box appears above any
reply that used one, and due reminders pop up as toasts without you
typing anything.

**Voice input**: install `stt-whisper` too
(`pip install -e ".[dev,memory-chroma,agents,ui-streamlit,stt-whisper]"`) and a
🎤 recorder appears above the message box — record a clip, it's
transcribed and sent exactly like a typed message.

**Spoken replies**: also install `tts-piper`, then toggle "🔊 Speak
replies" in the sidebar. This needs a *real* downloaded voice file for
whichever persona you're talking to — the starter personas' `voice_id`s
are placeholders (see "Voice mode" below for how to get one); without a
real file, toggling this on just shows a clear inline warning per reply,
it doesn't break the chat.

### Voice mode

Everything above, plus real STT/TTS/VAD models and a real microphone +
speaker:

```
pip install -e ".[dev,memory-chroma,agents,stt-whisper,tts-piper,vad-silero,audio-sounddevice]"
```

Two extra things voice mode needs that text mode doesn't, plus one
option:

1. **A real voice file for whichever persona you talk to.** The starter
   personas' `voice_id`s (e.g. `piper:en_US-mentor-medium`) are
   placeholders — see `companion/personas/schema.md` — there's no
   matching `.onnx` file shipped. Download one (e.g. via
   `python -c "from piper.download_voices import download_voice; from pathlib import Path; download_voice('en_US-lessac-medium', Path('./data/voices'))"`)
   and either rename it to match a persona's `voice_id`, or edit the
   persona's `voice_id` to match what you downloaded.
2. **`mode: voice` in `config.yaml`**, plus its `audio:` section —
   `sample_rate` must equal `vad.options.sample_rate` (both default to
   `16000`); leave `input_device`/`output_device` as `null` to use your
   system's default mic/speakers, or run `python -m sounddevice` to list
   devices and pick specific ones.
3. **Optionally, a lighter voice model.** By default voice mode uses
   the same `qwen2.5:7b-instruct` as text mode (about 0.85 s to first
   audio on a laptop GPU). On a CPU-only machine, uncomment
   `llm.voice_mode_override` in `config.yaml` to use
   `qwen2.5:1.5b-instruct` instead (Apache-2.0; `ollama pull
   qwen2.5:1.5b-instruct` first). Personas with tools always use the
   `agent:` model.

Then:

```
python -m companion
```

Speak, pause, and the persona replies out loud — press Ctrl+C to quit.
With a persona that has tools, you'll hear a short filler line ("Let me
check.") while a tool runs, and you can talk over it at any point to
interrupt. Due reminders are spoken before the next reply.
Same `--persona`/`--user-id`/`--config`/`--debug` flags as text mode. If
no microphone/speaker is available (e.g. running headless), or the
configured sample rate doesn't match `vad.options.sample_rate`, you get a
clear one-line error naming the problem, not a hang or a raw
`PortAudioError` traceback.

### Memory backend (optional: MemArk instead of Chroma)

Chroma is the default and what both walkthroughs above install. To try
the experimental MemArk graph-memory backend instead — see
`docs/MEMARK_BACKEND.md` for why you might *not* want to. **Not because
of speed** — measured (`scripts/benchmark_memory_backends.py`, real
numbers, this project's own hardware), MemArk's steady-state write/query
latency is roughly comparable to Chroma's (within ~5-15%), with one
disclosed exception: the first query against a brand-new
conversation/persona pairing after a restart is close to 2x slower (a
one-time graph-hydration cost, not a per-turn one). It's still labeled
experimental for a different, unrelated reason: **worse raw recall than
Chroma on MemArk's own published benchmarks**, and **salience decay**
means older memories can stop surfacing by design, not by bug — both
real product tradeoffs, not something a faster benchmark run would fix.

```
pip install -e ".[dev,memory-memark]"   # networkx, faiss-cpu, chromadb
                                          # (chromadb here is only for its
                                          # standalone default embedding
                                          # function, not a Chroma DB)
```

Then in `config.yaml`:

```yaml
memory:
  backend: memark
  options:
    persist_dir: ./data/memark
```

Same `python -m companion` afterwards — everything else (persona
selection, `--user-id`, text/voice mode) is unchanged; only where memory
gets stored and how it's retrieved differs. Memory is scoped the same
way as Chroma's (one file per `user_id`+`persona_id`, under
`persist_dir`), just SQLite instead of a Chroma collection — inspectable
directly with any SQLite browser if you want to see what's actually
stored. If `memory-memark`'s extras aren't installed, selecting this
backend fails with a clear import error at startup, before any
conversation loop runs, not partway through one.

### Memory summarization

Long conversations don't pile up forever as raw turns. Once a persona's
memory for you holds `memory.summarizer.trigger_turn_count` turns (50 by
default), the oldest half, in whole exchanges, are summarized by the LLM
(at a low temperature, for faithfulness) into one summary turn
in the background, after a reply, so it never slows a reply down. Set it
to `0` to turn it off. Works with both memory backends. Design and safety
details (e.g. a summary can't bring back memory you deleted) are in
`docs/ROLLING_SUMMARIZER.md`. It's tested with a fake LLM against real
stores; summary quality with a real model isn't measured yet.

### Memory survives restarts (and how that's tested)

Both backends persist to `persist_dir` and pick up where they left off on
the next `python -m companion` run. `tests/test_memory_persistence_across_processes.py`
proves this with real, separate processes, never a second store object
in the same process: one process writes and exits, a new one reads. It
runs for both backends at three levels: store write/query/delete
(including persona scoping and exact timestamps); the orchestrator, where
a fact from session one reaches session two's LLM prompt with no live
history; and two real `python -m companion` runs against a local Ollama
server, where the second run has to answer a question only the first
run's memory can answer (skipped without Ollama). As a control, the same
question with an empty memory directory got a wrong answer.

### Checking that it runs offline

After models and voices have been downloaded once:

```
python scripts/check_offline.py            # uses config.yaml, like the app
```

This blocks every non-localhost connection and DNS lookup at Python's
socket layer, then runs each configured backend for real: an Ollama
reply, write + query on both memory backends, and a real audio round
trip. TTS speaks a sentence, Silero VAD detects the speech in it, and
faster-whisper transcribes it back. If the config has an `agent:`
section, the agent model also makes one real tool call (the calculator).
It lists any connection attempts it
blocked. faster-whisper tries `huggingface.co` to check for model
updates, then falls back to its local cache. Backends whose extras
aren't installed are reported as SKIP. Limit: native code that opens its
own sockets isn't covered, so for a full guarantee, run it with the
network actually off.

Checked on 2026-10-19 with both TTS backends: Piper, and Coqui with
`tts_models/en/ljspeech/vits` (persona `voice_id: coqui:default`; needs
the espeak-ng system binary). Coqui's multi-GB XTTS-v2 model wasn't run
through it.

See `docs/CONTRIBUTING.md` for the full contributor setup and
`docs/PERSONA_AUTHORING.md` to add a persona (no code changes required).

### Performance

Measured on an i7-12700H laptop with an RTX 3060 (6 GB): about 0.85 s
from the end of your speech to the first audio in voice mode, 7–8B
models at about 35 tokens/s, and tool-using replies under 2 s. Numbers,
CPU-only estimates and how to measure your own machine
(`scripts/benchmark_latency.py`): `docs/HARDWARE_REQUIREMENTS.md`.

## Repo layout

See `docs/ARCHITECTURE.md` for how it's built and `DECISIONS.md` for why
(every settled design decision, with its reasoning). Each package's
`__init__.py` docstring is a short guide to that package. `legacy/` contains an archived, unrelated prior
project (a Node/React text-chat app) kept for reference only — it is not
part of this codebase.

## License

MIT for this repo's own code (see `LICENSE`). Bundled/downloaded model
weights carry their own separate licenses — see `THIRD_PARTY_LICENSES.md`
before assuming MIT extends to a model.
