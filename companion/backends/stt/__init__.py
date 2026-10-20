"""Speech-to-text backends. `whisper_local.FasterWhisperSTT` is the one
implementation: buffered re-transcription with LocalAgreement-2
stabilization, 16 kHz mono PCM16 in (see its module docstring and
docs/ARCHITECTURE.md "STT"). The config also names `whisper_cpp`, which
isn't implemented.
"""
