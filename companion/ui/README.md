# UI

**Streamlit** — `streamlit_app.py`, a real chat window built against the
same `Orchestrator`/`build_*` functions `companion/app.py`'s CLI text
loop uses, not a separate app-construction path. Text chat plus optional
**push-to-talk voice** (record a clip in the browser, transcribed with
the same `stt: SpeechToText`, spoken replies via the same
`tts: TextToSpeech`) — not the CLI's real-time hands-free voice mode,
which needs a persistent event loop + real-time barge-in state machine
that doesn't fit Streamlit's per-interaction rerun model without
substantially more work. See that file's module docstring for the full
scope/rationale and README.md's "Running the app" section for how to
launch it (including the extra installs voice needs).

This isn't a claim that Streamlit is the final or only UI this project
will ever have — it was picked as a fast, local-first-friendly way to
get a real chat window running. A richer client (web app with its own
frontend framework, a TUI, a desktop app) is still open if/when that's
worth the larger lift; nothing here should be read as having settled
that question for good.
