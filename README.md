# Companion

An open-source, local-first AI companion. Talk (voice or text) with a
persona of your choice — a wise old mentor, a casual friend, a romantic
companion, or one the community writes — running on local models by
default, with no external server required to get started.

**Status: both text and voice mode run end-to-end, for real.** STT, TTS,
LLM, VAD, the persona loader, `MemoryStore`, and real mic/speaker audio
I/O are all implemented; `python -m companion` reads config, constructs
real backends, and runs an actual conversation with memory — text or
voice, picked by `config.yaml`'s `mode:` — see "Running the app" below.
See `docs/ARCHITECTURE.md` for the design and the build order, and the
root `CLAUDE.md` for exactly what's implemented vs. not.

## Stack (local-first phase)

- STT: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU fallback: whisper.cpp, not yet implemented)
- LLM: [Ollama](https://ollama.com), config-selectable model
- TTS: [Piper](https://github.com/rhasspy/piper) (optional higher-fidelity: [Coqui/XTTS-v2](https://github.com/idiap/coqui-ai-TTS), incl. future voice cloning)
- VAD: [Silero VAD](https://github.com/snakers4/silero-vad)
- Memory: [Chroma](https://www.trychroma.com/) (embedded, no external DB server; LanceDB named as a future alternative, not yet implemented) — an experimental MemArk graph-memory backend is also wired in, see "Memory backend" below

A cloud-backed hosted demo mode is planned for after the local-first core
is solid — see `docs/ARCHITECTURE.md`'s "Build order." It is not built yet.

## Running the app

### Text mode

This is genuinely runnable, end to end, with just:

```
pip install -e ".[dev,memory-chroma]"
```

(STT/TTS/VAD get constructed too but text mode never calls into them, so
their extras — `stt-whisper`, `tts-piper`, `vad-silero` — aren't needed
just to talk to a persona in text mode; add them for voice mode, below.
`llm-ollama`'s dependencies are already covered by `dev`.)

Then, with [Ollama](https://ollama.com) installed and running and at
least one model pulled (`ollama pull llama3` or whichever model you'd
rather use):

```
cp companion/config/config.example.yaml companion/config/config.yaml
# edit config.yaml's llm.options.model to a model you've actually pulled
# (ollama list shows what's available; the shipped default,
# qwen2.5:7b-instruct, is a recommendation, not something pre-pulled for you)

python -m companion
```

You'll be talking to whichever persona loads first from
`companion/personas/` (override with `--persona mentor`/`friend`/
`companion`). Type `exit` to quit. Useful flags: `--persona <id>`,
`--user-id <id>` (memory is scoped per user+persona; defaults to
`local`), `--config <path>` (default: `config.yaml` if you've created
one, else `config.example.yaml` itself — so `python -m companion` with
zero setup still runs, it'll just fail with a clear error at the first
message if the default model isn't pulled), `--debug` (full traceback
instead of a one-line error message).

If something's wrong — Ollama isn't running, the configured model isn't
pulled, a persona file is malformed — you'll get a clear one-line error
telling you what to fix, not a raw traceback (unless you pass `--debug`).

### Voice mode

Everything above, plus real STT/TTS/VAD models and a real microphone +
speaker:

```
pip install -e ".[dev,memory-chroma,stt-whisper,tts-piper,vad-silero,audio-sounddevice]"
```

Two extra things voice mode needs that text mode doesn't:

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

Then:

```
python -m companion
```

Speak, pause, and the persona replies out loud — press Ctrl+C to quit.
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

See `docs/CONTRIBUTING.md` for the full contributor setup and
`docs/PERSONA_AUTHORING.md` to add a persona (no code changes required).

## Repo layout

See `docs/ARCHITECTURE.md`. `legacy/` contains an archived, unrelated prior
project (a Node/React text-chat app) kept for reference only — it is not
part of this codebase.

## License

MIT for this repo's own code (see `LICENSE`). Bundled/downloaded model
weights carry their own separate licenses — see `THIRD_PARTY_LICENSES.md`
before assuming MIT extends to a model.
