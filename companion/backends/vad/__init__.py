"""Voice activity detection. `silero_vad.SileroVAD` classifies fixed-size
frames (512 samples at 16 kHz, exposed as `frame_bytes`, which isn't part
of the abstract interface) with internal debouncing. torch is imported
lazily.
"""
