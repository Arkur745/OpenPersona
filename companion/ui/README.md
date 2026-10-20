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

Agentic personas work here too:
- a persona with tools shows a "Tools: ..." caption under its name;
- each reply that used a tool gets a collapsed "Used tools" status box
  listing them;
- due reminders appear as toasts, from an `st.fragment` that reruns every
  20 seconds, so you don't have to type anything.

**Speak replies** autoplays each spoken reply (no click needed). For
push-to-talk messages, the audio is kept in session state across the
`st.rerun()` that follows and played once afterwards; before that fix,
the rerun wiped the player before it could play.

Rolling summarization also runs here, after a reply has rendered rather
than as a background task, because each Streamlit turn runs under its own
event loop (see `docs/ROLLING_SUMMARIZER.md` section 5).

Tests: `tests/test_streamlit_app.py` drives the real script with
Streamlit's `AppTest` and fake backends: a chat round trip, the
clear-chat button, and the tool display.

This isn't a claim that Streamlit is the final or only UI this project
will ever have — it was picked as a fast, local-first-friendly way to
get a real chat window running. **The next UI is planned as a React
frontend** (decided 2026-10-19, not started). It needs a local API server
first (HTTP plus a WebSocket for streaming text and audio), since
Streamlit calls `Orchestrator` in-process and a browser app can't. That
server is also what makes hands-free real-time voice in the browser
possible. Framework and API details are still open.
