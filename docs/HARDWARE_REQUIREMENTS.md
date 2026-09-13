# Hardware requirements (draft)

**Status: draft, based on published specs for the underlying open-source
projects, not yet benchmarked against this codebase.** Numbers should be
replaced with measured figures once the pipeline is wired end-to-end
(build-order step 9) — treat everything below as a starting estimate for
setting expectations, not a guarantee.

## Summary

| Tier | CPU-only viable? | Approx. RAM | Approx. VRAM | Notes |
|---|---|---|---|---|
| Minimum (text mode only) | Yes | 8 GB | — | Small/quantized LLM via Ollama on CPU; expect multi-second replies. |
| Minimum (voice mode) | Marginal | 8 GB | — | STT/TTS add real-time pressure; CPU-only voice mode will feel laggy. |
| Recommended | Yes (LLM slow) / better with GPU | 16 GB | 6-8 GB | Comfortable text mode on CPU; smooth voice mode wants a GPU. |
| Full fidelity (XTTS voice cloning, 8B+ LLM, larger Whisper) | No | 32 GB | 8-12 GB | For contributors doing TTS-quality or larger-model work. |

## By component

- **LLM (Ollama)** — the dominant cost. A quantized 7-8B instruct model
  (Llama 3.1 8B, Qwen2.5 7B) needs roughly 6-8 GB of RAM or VRAM at
  4-bit quantization; runs on CPU but slower (seconds per reply vs.
  sub-second on a mid-range GPU). The config's `voice_mode_override` to a
  ~3B model exists specifically to keep voice-mode latency acceptable on
  more modest hardware — see `config/config.example.yaml`.
- **STT (faster-whisper)** — `base`/`small` model sizes are usable on CPU
  in roughly real time; `medium`/`large-v3` want a GPU to keep up with
  live speech. Default config uses `base`.
- **TTS (Piper)** — the lightest component; runs comfortably on CPU with
  low latency, which is why it's the default. **Coqui/XTTS-v2** (optional,
  for voice cloning / higher fidelity) is materially heavier — expect to
  want a GPU with several GB of VRAM for real-time-ish synthesis.
- **VAD (Silero)** — negligible; runs on CPU in real time on essentially
  any hardware from the last decade.
- **Memory (Chroma/LanceDB)** — negligible for typical single-user
  conversation history volumes; embedded, no separate server.

## Open items

- Real benchmarks (tokens/sec, STT real-time factor, TTS latency) per
  tier, once the orchestrator is wired end-to-end.
- Guidance for Apple Silicon (Metal) specifically, distinct from generic
  "CPU-only."
