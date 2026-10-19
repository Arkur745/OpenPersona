# Authoring a persona

Personas are data, not code. Adding one should never require touching
`companion/core/` or `companion/backends/`.

## Steps

1. Read the field-by-field format in
   [`companion/personas/schema.md`](../companion/personas/schema.md).
2. Copy an existing persona (`companion/personas/mentor.yaml`,
   `friend.yaml`, or `companion.yaml`) as a starting point rather than
   writing one from scratch.
3. Pick a unique `id`. This becomes a permanent memory boundary — see
   the "Field notes" in `schema.md` before renaming an `id` later, since
   that effectively orphans existing memory under the old id.
4. Write `tone` and `boundaries` as prose the LLM will actually read
   verbatim (they're folded directly into the prompt by `PromptBuilder`) —
   be concrete and specific, not a marketing description of the persona.
5. Write 2-4 `example_dialogue` snippets that demonstrate voice, not
   facts to memorize. Keep each snippet short.
6. Pick or record a voice:
   - Using a stock Piper voice: set `voice_id: piper:<model-name>` and
     make sure `<model-name>.onnx` exists under your `config.yaml`'s
     `tts.voices_dir` — that's what actually gets resolved per-persona
     (`companion/app.py`'s `build_tts_for_persona`, via
     `PiperTTS.from_voice_id`); `tts.options.voice_model_path` is only a
     fallback for when a persona has no usable `voice_id`, not the normal
     per-persona path.
   - Using a cloned voice (Coqui/XTTS): set `voice_id: coqui:<label>` and
     provide a reference `speaker_wav`. **Only clone a voice with the
     speaker's explicit consent** — this applies to community
     contributions too; do not submit a persona bound to a cloned voice
     unless you have the right to use that voice.
7. Optionally give it tools, which makes it an agent (needs the config's
   `agent:` section and a tool-calling model; see
   `companion/personas/schema.md`):
   - `tools:` lists names from `companion/tools/registry.py`:
     `get_datetime`, `calculator`, `notes`, `reminders`,
     `recall_memory`, and the network tools `web_search` and `weather`,
     which only run when `agent.options.allow_network` is true.
   - Pick tools that fit the persona, not all of them. Each extra tool
     is one more thing a 7B model can call at the wrong moment.
   - `filler_phrases:` are short lines spoken in voice mode while a tool
     runs ("Let me check."). Write them in the persona's voice.
   - A new tool, as opposed to a new persona, *is* a code change: a
     `build_*` function in `companion/tools/` plus one registry entry.
8. If the persona is meant for adult-only content boundaries (see
   `companion.yaml` for the pattern), say so explicitly in `boundaries` —
   age-gating itself is enforced outside the persona file, but the
   persona should never assume it's unnecessary.

## Validating a new persona

`companion.core.persona_loader.load_persona("companion/personas/your-file.yaml")`
parses and validates the file against `schema.md` (required fields,
`id` format, `voice_id` shape, unknown fields, duplicate `id`s across the
directory) and raises `PersonaError` naming the file/field/reason if
something's wrong — run it against a new persona before opening a PR.

## Testing a new persona

Run it for real: `python -m companion --persona <your-persona-id>` (see
README.md's "Running the app") and try a handful of representative user
messages. Read the transcript for: does it consistently sound like the
persona described, and does it correctly deflect anything in
`boundaries`? There is no automated "sounds right" check — this is a
manual read for now. If you've set up a real voice file for it, `mode:
voice` runs the same persona through real STT/TTS/VAD — worth doing at
least once per new voice to confirm the `voice_id` actually resolves and
sounds like what you intended; still a manual listen, not automated.
