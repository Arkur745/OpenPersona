"""Text-to-speech backends: `piper_local.PiperTTS` (default, native
per-sentence streaming) and `coqui_local.CoquiTTS` (optional, higher
fidelity). Each instance is bound to one voice, resolved from a persona's
`voice_id` by the backend's `from_voice_id()`. `_sync_stream.py` is the
shared helper that bridges blocking synthesis onto a background thread
with stop support; reuse it for any new TTS backend.
"""
