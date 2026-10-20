"""Real microphone and speaker I/O for voice mode, via sounddevice
(imported lazily):
- `MicCapture` (`capture.py`): an async iterator of PCM16 chunks, usable
  directly as `handle_voice_turn`'s `mic_audio`.
- `SpeakerPlayback` (`playback.py`): queued playback with `stop()` and
  `wait_drained()`, which drive the orchestrator's barge-in hooks.
- `AudioDeviceError` (`errors.py`): every PortAudio failure is wrapped in
  it with an actionable message.

Not one of the abstract interfaces; see docs/ARCHITECTURE.md "Voice mode
audio I/O".
"""

from companion.audio.capture import MicCapture
from companion.audio.errors import AudioDeviceError
from companion.audio.playback import SpeakerPlayback

__all__ = ["AudioDeviceError", "MicCapture", "SpeakerPlayback"]
