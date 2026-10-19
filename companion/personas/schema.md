# Persona config format

A persona is one YAML file in `companion/personas/`. The loader
(`companion/core/persona_loader/`, build-order step 7) parses a file
matching this shape into a `companion.core.interfaces.types.Persona`.

```yaml
id: mentor                 # stable slug, used as persona_id everywhere
                            # (memory scoping, voice binding, CLI/UI selection)
name: "The Old Mentor"     # display name
tone: >
  Calm, plainspoken, a little dry-humored. Speaks in short sentences.
  Draws on decades of (implied) lived experience rather than textbook
  advice. Never preachy.
boundaries: >
  Not a therapist, doctor, or financial advisor and says so plainly if
  asked for that kind of help, redirecting to a real professional.
  Does not roleplay romantic or sexual scenarios.
voice_id: piper:en_US-mentor-medium   # "<tts-backend>:<voice model id>" —
                                       # resolved against whichever TTS
                                       # backend config.yaml selects
example_dialogue:
  - |
    User: I think I made the wrong call leaving that job.
    Persona: Maybe. Or maybe you just haven't seen where the road bends yet.
    Either way, sitting in "what if" won't tell you. What would moving
    forward look like?
  - |
    User: Everyone my age already has it figured out.
    Persona: They don't. They're just better at hiding the scramble. You're
    not behind, you're just watching the highlight reel.
```

## Field notes

- **`id`** must be filesystem- and URL-safe (`[a-z0-9_-]+`) — it's used as
  the persona_id passed to `MemoryStore.query`/`write`, so it also defines
  a memory boundary. Never reuse an `id` for a materially different
  persona; that would blend unrelated memories.
- **`voice_id`** is namespaced by TTS backend (`piper:...`, `coqui:...`) so
  a persona doesn't silently break if the deployment's TTS backend changes
  — the loader should raise a clear error if the configured TTS backend
  doesn't recognize the namespace, not silently fall back. Concretely,
  resolution is a classmethod on the matching backend:
  `PiperTTS.from_voice_id("piper:en_US-lessac-medium", voices_dir)` looks
  for `<voices_dir>/en_US-lessac-medium.onnx`;
  `CoquiTTS.from_voice_id("coqui:some-speaker", model_name=...)` treats the
  part after the colon as a built-in speaker name for a multi-speaker
  Coqui model, and `coqui:default` selects a single-speaker model's only
  voice (see `companion/backends/tts/coqui_local.py`'s module
  docstring for how this extends to voice cloning later without changing
  this field's shape). The persona loader
  (`companion/core/persona_loader/`) only validates that `voice_id` is
  *shaped* `<namespace>:<value>` — actually resolving it via
  `from_voice_id()` into a bound TTS backend instance happens once per
  persona, later, when the app wiring constructs each persona's backend
  instance from `config.yaml` (not per message, and not the loader's own
  job — see `docs/ARCHITECTURE.md`'s "Persona loader" section).
- **`example_dialogue`** grounds style, not facts — keep entries short (2-4
  lines) and illustrative of *tone*, not a script to be replayed. These are
  folded into the prompt by `PromptBuilder`, so keep the total token count
  reasonable (a handful of short exchanges, not a transcript dump).
- Community-contributed personas should need nothing beyond a new YAML file
  here (plus a voice model file if using a non-stock voice) — no code
  changes. If adding a persona ever requires touching Python, that's a gap
  in the loader, not a reason to hardcode the persona.
