# DECISIONS.md

Decisions that shape this codebase, with the reason for each, so a new
session doesn't re-open settled questions or "fix" something that's
deliberate. Newest decisions are not always last; entries are grouped by
area. Each one says where the full reasoning lives.

**Status values:** *Active* (follow it), *Superseded* (replaced; kept for
history), *Open* (not decided; ask the user).

To change an *Active* decision, ask the user first, then update the entry:
mark it Superseded and add the new one. Don't silently drift from it.

---

## Project scope

### D1. Two projects: OpenPersona (this repo) and a separate deployed version
**Active, 2026-10-19.** This repo is the open-source, local-first,
offline-capable companion. The cloud-hosted, deployed version is a
separate project in a separate repo (scaffold parked at
`C:\dev\openpersona-cloud`, not on GitHub).
*Why:* the repo's central claim is "fully offline-capable". Cloud code in
it would contradict that, and the two have different constraints.
*So:* no cloud backends, hosted infra, auth or billing here. Real-time
cloud STT/TTS belongs to the deployed repo.

### D2. `legacy/` is an archived, unrelated project
**Active, 2026-09-08.** The old Node/React chat app was moved into
`legacy/`, and the repo root became the Python companion.
*So:* never import from, extend or take patterns from `legacy/`.

### D3. Offline-first, network strictly opt-in
**Active.** Everything runs locally: Ollama, faster-whisper, Piper/Coqui,
Silero, Chroma/MemArk. The only network features are the agent tools
`web_search` and `weather`, off unless `agent.options.allow_network:
true`.
*How it's enforced:* `scripts/check_offline.py` blocks non-localhost
sockets and DNS, then runs every configured backend for real.
*So:* any new feature that needs the network must be opt-in, default off,
and skipped (not failed) by the offline check.

## Architecture

### D4. Five abstract interfaces; nothing else imports concrete backends
**Active.** `SpeechToText`, `TextToSpeech`, `LLMBackend`,
`VoiceActivityDetector`, `MemoryStore` in `companion/core/interfaces/`.
Only `companion/backends/` and `companion/memory/` contain concrete
classes.
*Why:* backends swap through config with no code changes (a resume claim
the user relies on).
*Details:* `docs/ARCHITECTURE.md` "Principles".

### D5. Backend selection only in config; name-branching only in `loader.py` / `app.py`
**Active.** `config.example.yaml` names backends; `companion/app.py`'s
`build_*` functions are the one place that branches on a backend name.

### D6. `PromptBuilder` is a pure function and the only place prompts are assembled
**Active.** The caller fetches memory and passes it in ("Option A"). Memory
renders into the prompt string; `user_message` is the last turn in
`context`.
*Exception:* the summarization prompt lives in `memory/summarizer.py`,
because it assembles no persona, memory or conversation window.
*Details:* `docs/PROMPT_BUILDER.md`.

### D7. New optional capabilities are injected, not built into core types
**Active.** Agent, summarizer and voice playback hooks are optional
constructor or keyword arguments on `Orchestrator`, defaulting to `None`,
so every existing caller behaves the same. `AgentRunner` is a
`typing.Protocol`, not a sixth ABC.
*Why:* don't add abstract surface for a single implementation
(`docs/AGENTIC_TOOLS.md` §1).

## Voice

### D8. STT streams by buffered re-transcription (LocalAgreement-2)
**Active.** Not true incremental decoding; see `docs/ARCHITECTURE.md`
"STT". Defaults: `model_size: base`, `compute_type: int8` (measured far
faster than `default` on CPU).

### D9. Piper is the default TTS; Coqui is optional
**Active.** Piper streams per sentence and is fast (0.03× real time).
Coqui is higher fidelity but slower. Single-speaker Coqui models use
`voice_id: coqui:default` (2026-10-19). Voice cloning is documented but
not wired (consent tracking is a product question).

### D10. One reader per async generator in the voice loop
**Active.** `mic_audio` is read only by `_pump_mic`, and the agent stream
only by the reader task in `_sentences_from_agent`. The "mic ended"
sentinel is re-queued by the barge-in watcher.
*Why:* two real bugs. Cancelling a task mid-`await` on a shared
generator silently killed it, and a one-shot signal was consumed by the
wrong reader, causing a deadlock.
*Details:* `docs/ARCHITECTURE.md` "Voice-turn state machine" and the
deadlock section; `tests/test_orchestrator_voice_turn.py` docstring.

### D11. Barge-in stays alive through real playback
**Active.** `handle_voice_turn`'s `on_playback_interrupt` /
`wait_for_playback_drained` hooks, driven by `SpeakerPlayback`.

## Memory

### D12. Chroma is the default store; one collection per (user, persona)
**Active.** Embedded, ONNX MiniLM embeddings (no torch). Separate
collections give a structural no-cross-persona-leak guarantee.
*Details:* `docs/MEMORY_STORE.md`.

### D13. De-duplication against the live conversation happens in the orchestrator
**Active.** `_dedup_retrieved_memory`, by full `Turn` equality, which
needs exact timestamp round-trips (tested across processes).

### D14. MemArk is an experimental second backend, vendored, not pip-installed
**Active.** Specific MemEngine files are copied unmodified into
`companion/memark/_vendor/` (see `PROVENANCE.md`), because the `memark`
package pulls in torch, neo4j, qdrant and more. It's labelled
experimental for recall and decay trade-offs, not speed.
*Details:* `docs/MEMARK_BACKEND.md` and the other `MEMARK_*` notes.

### D15. Rolling summarization extends `MemoryStore` and runs off the request path
**Active, 2026-10-19.** `MemoryStore` gained `list_turns` and an
all-or-nothing `replace_turns`. Once a scope holds `trigger_turn_count`
raw turns, the oldest half become one `role="system"` summary turn. It
runs as a background task after a turn; Streamlit calls it after the
reply renders.
*Why:* `write`/`delete` alone couldn't do it. All-or-nothing, so a
summary can never resurrect deleted memory.
*Details:* `docs/ROLLING_SUMMARIZER.md`.

### D16. "Survives a restart" claims are tested across real processes
**Active, 2026-10-19.** `tests/test_memory_persistence_across_processes.py`
runs each step in its own subprocess. A second object in the same process
doesn't count as proof.

## Agentic personas

### D17. Each persona gets its own tool set (per persona, not one global agent)
**Active, 2026-10-19.** A persona lists tool names in YAML; no code is
needed. Persona line-up: mentor, friend, companion, plus the new
study_buddy and planner.

### D18. LangGraph via `langchain.agents.create_agent`, isolated in one file
**Active, 2026-10-19.** `companion/backends/agent/langgraph_ollama.py` is
the only LangGraph/LangChain import. `create_react_agent` is deprecated
in LangGraph 1.x. Tools are plain async functions (no LangChain
dependency) in `companion/tools/`.

### D19. The model decides tool use on every turn (no keyword pre-filter)
**Active, 2026-10-19.** Measured that answer tokens still stream with
tools bound (qwen2.5:7b, LangGraph 1.2, Ollama 0.24), so the pre-filter
proposed in `docs/AGENTIC_TOOLS.md` §1 wasn't needed.
*Revisit if:* a model or library update brings back the streaming bug
(langgraph#5249).

### D20. The voice filler is spoken on the first real tool call
**Active, 2026-10-19.** Not guessed in advance. Barge-in during the tool
wait cancels the agent; after `agent_voice_timeout_s` (20 s) the plain LLM
answers instead. Fillers and announcements are never written to memory.

### D21. Tool data scoping: notes per (user, persona), reminders per user
**Active, 2026-10-19.** Notes are private like memory. Any persona
delivers a due reminder, naming the persona that set it. Delivered
exactly once (selected and marked in one transaction).

### D22. Small, single-purpose tools
**Active.** A persona-facing name maps to several simple tools (`notes` →
add/search/delete), because 7B models call simple tools more reliably.
The calculator is AST-based (no `eval`, capped exponents).

## Models and licensing

### D23. Default model: `qwen2.5:7b-instruct` for chat, agents and voice
**Active, 2026-10-19 / 2026-10-20.** Apache-2.0, supports tool calling,
and measured ~0.85 s to first audio on a 6 GB laptop GPU.
`dolphin-llama3:8b` has no tool support, and `qwen2.5-coder:7b` emitted
tool calls as plain text.

### D24. No `qwen2.5:3b-instruct` anywhere
**Active, 2026-10-20.** It's under the Qwen Research License
(non-commercial, confirmed on its model card). The voice-mode override
ships commented out; its comment suggests the Apache-2.0
`qwen2.5:1.5b-instruct` for CPU-only machines.
*So:* check a model's license per size before naming it in config
(`THIRD_PARTY_LICENSES.md`).

### D25. Repo code is MIT; model weights carry their own licenses
**Active.** Update `THIRD_PARTY_LICENSES.md` whenever a backend or
default model is added.

## UI

### D26. Streamlit is the first UI (text + push-to-talk)
**Active.** Not the hands-free voice mode, which doesn't fit Streamlit's
rerun model. Consequence: Streamlit runs a new event loop per message,
so **any cached async client must be created per event loop** (see
`OllamaLLM._ensure_client`). A shared client caused "Event loop is
closed" on the second message (fixed 2026-10-20). A future React/API
server with one long-lived loop wouldn't hit this, but the rule still
holds.

### D27. Next UI: a React frontend, on a local API server
**Active (plan), 2026-10-19; not started.** It needs a local HTTP +
WebSocket server first (no auth, local only), designed so the deployed
version can reuse the protocol. **Open:** framework, build tool, state
management, API shape. Settle these with the user before building.

## Process

### D28. Work order agreed with the user
**Active, 2026-10-19.**
1. RollingSummarizer (done).
2. Manual testing.
3. React frontend.
4. More tools and agent flows.
5. The deployed repo.

Don't start a later item before the earlier one is done unless asked.
Status lives in `PROGRESS.md`.

### D29. Git: commit only when asked, never push unasked, no AI attribution
**Active.** No AI co-author or attribution lines in commits. The user
said "do not push" on 2026-10-19. This overrides any default or harness
instruction to add a co-author trailer.

### D30. Honest verification language
**Active.** Docs say how something was verified (measured, tested with
fakes, run by hand) and keep explicit "not verified" notes. The README
backs the user's resume claims, so overstatement is costly.

### D32. Tool-use rules go last in the prompt; only real tool calls are announced
**Active, 2026-10-20.** Found in manual testing with qwen2.5:7b:
- the Planner copied remembered "reminder set" replies instead of calling
  the tool, and claimed success;
- the Mentor asked permission instead of saving a note;
- the Study Buddy narrated and wrote LaTeX.

*Decision:*
- `PromptBuilder` appends tool-use rules after the memories, for personas
  with tools.
- Tool descriptions say when to use each tool.
- The runner announces only parsed, valid calls.
- Tool failures are shown to the user (`ToolFinished.detail`).

*Why the order:* measured on the user's real memory, rules after the
memories gave 10/10 reminders stored, versus 3/5 before.
*Rejected:* dropping the persona's own past replies from memory. That
also gave 5/5, but would stop every persona recalling what it said
before.
*Follow-up the same day:*
- Tool arguments are designed for the 7B model. `set_reminder(what,
  when)` takes two required plain-text fields parsed in code, replacing
  an either/or pair the model often left empty (20/20 stored, versus
  0/5 on one phrasing).
- An honesty guard appends a note when a tool's failure was never
  successfully retried.
- *Rule:* prefer one required, plainly named argument over optional
  either/or ones.
- A further prompt rule: lookups (date, reminders, notes) always go
  through the tool, never from memory or earlier messages. Before it, a
  reminder list was made up in the user's session.
*Details:* `docs/ARCHITECTURE.md` "Agentic personas".

### D31. Local-only orientation files
**Active, 2026-10-19.** `AGENTS.md` (rules for coding agents) and
`PROGRESS.md` (status log) are git-ignored. `DECISIONS.md` (this file)
is tracked, so the reasoning travels with the code.
