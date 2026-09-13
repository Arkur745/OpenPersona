"""Shared error type for companion/audio/ — see capture.py and
playback.py's module docstrings for what raises it and why."""


class AudioDeviceError(RuntimeError):
    """A real audio input/output device couldn't be opened as configured:
    no device available (e.g. running headless/CI), an invalid device
    index/name, a sample rate the hardware doesn't support, or the device
    is busy/permission-denied. Always wraps the underlying
    `sounddevice.PortAudioError` with a clearer, more specific message —
    callers should never see a raw `PortAudioError` traceback."""
