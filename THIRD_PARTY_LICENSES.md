# Third-party licenses

This repository's own code is MIT-licensed (see `LICENSE`). That covers
**our code only**. Every backend in `companion/backends/**` optionally
depends on a third-party library and, at runtime, downloads or loads model
weights — those carry their own separate licenses that this project does
not and cannot override. This file tracks what's known as of each backend
being integrated; update it whenever a new backend or a new default model
is added, don't assume anything here automatically covers a model that
isn't listed.

**Nothing on this list is legal advice.** If you're distributing a build
that bundles model weights, or deploying commercially, verify the current
license text yourself at the source linked below — licenses on hosted
model repos can and do change.

| Component | License | Notes |
|---|---|---|
| **faster-whisper** (code) | MIT | https://github.com/SYSTRAN/faster-whisper |
| **Whisper** (original model weights, what faster-whisper's CTranslate2 conversion is built from) | MIT | https://github.com/openai/whisper |
| **Piper** (code) | MIT | https://github.com/rhasspy/piper |
| **Piper voices** (model weights) | Varies per voice | Hosted at https://huggingface.co/rhasspy/piper-voices — most are CC0 or MIT depending on the source dataset, but check the individual voice's `MODEL_CARD` before bundling/redistributing one, don't assume CC0 applies to all of them. |
| **coqui-tts** (code — the actively maintained `idiap/coqui-ai-TTS` fork; see note below) | MPL-2.0 | https://github.com/idiap/coqui-ai-TTS |
| **XTTS-v2** (model weights, Coqui's higher-fidelity/voice-cloning model) | **Coqui Public Model License (CPML) — non-commercial** | https://coqui.ai/cpml — this is the one to actually pay attention to. CPML restricts commercial use without a license grant from Coqui, and Coqui the company shut down in Jan 2024, so there is no clear path to a commercial license today. **Do not enable XTTS-v2 as a default voice in any build described as commercially usable** without resolving this. Stock (non-XTTS) Coqui models under `tts_models/en/...` etc. generally ship under more permissive per-model licenses — check the specific model's card via `TTS.api.TTS.list_models()` before relying on that assumption. |
| **espeak-ng** (system binary, required at runtime by some Coqui phonemizers — not a Python dependency, not bundled by us) | GPL-3.0 | https://github.com/espeak-ng/espeak-ng — install separately (e.g. via your OS package manager); we only ever shell out to / import-bind an already-installed copy, never vendor it. |
| **Silero VAD** (model weights) | MIT | https://github.com/snakers4/silero-vad — not yet integrated (build-order step 6); listed here ahead of time since it's already named in `config.example.yaml`. |
| **Ollama** (server) | MIT | https://github.com/ollama/ollama — the server itself is permissively licensed, but this covers *only* the server. |
| **Whatever LLM Ollama serves** (Llama, Qwen, Mistral, ...) | **Entirely dependent on which model you `ollama pull`** | Not something this project can pin once and forget — e.g. Meta's Llama models ship under the Llama Community License (source-available with acceptable-use restrictions, not OSI-approved), while Qwen2.5 is mostly Apache-2.0 with some larger variants under a separate Qwen license. Whoever configures `llm.options.model` in `config.yaml` is responsible for knowing that model's license and whether it fits their use case — this project doesn't ship a default model weight, only a default *config value* naming one. |
| **MemEngine / MemArk** (vendored source, `companion/memark/_vendor/engine/`) | MIT | https://github.com/Arkur745/MemEngine — not installed via pip; specific files (graph manager, activation engine, FAISS store, and their direct dependencies) are copied verbatim into this repo at a pinned commit, to avoid the whole package's own unconditional dependency list (torch, neo4j, qdrant-client, google-generativeai, groq, streamlit), none of which this project's adapter uses. See `companion/memark/_vendor/PROVENANCE.md` for the exact commit, file list, and reasoning, and `companion/memark/_vendor/LICENSE_MEMENGINE` for the license text copied verbatim from the source repo. |

## On `coqui-tts` vs. the original `TTS` package

Coqui the company shut down in January 2024, and the original
`coqui-ai/TTS` PyPI package (`TTS`) has had no release since 0.22 (Nov
2023) — it's unmaintained. `pyproject.toml`'s `tts-coqui` extra depends on
`coqui-tts` (https://github.com/idiap/coqui-ai-TTS), an actively
maintained fork that stays API-compatible (still imports as `TTS`). Using
the abandoned package would mean no fixes for new Python/PyTorch/
transformers versions going forward.
