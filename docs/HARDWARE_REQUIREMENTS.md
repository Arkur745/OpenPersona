# Hardware requirements

**Status: measured on one machine (below), 2026-10-19.** The tier table
further down extrapolates from those numbers plus the published model
sizes, so treat the tiers as guidance. To get numbers for your own
machine:

```
python scripts/benchmark_latency.py --models <model1>,<model2>
python scripts/benchmark_memory_backends.py
```

## Measured: laptop with a mid-range GPU

Hardware: Intel Core i7-12700H (14 cores / 20 threads), 16 GB RAM,
NVIDIA RTX 3060 Laptop GPU (6 GB), Windows 11, Python 3.13. Ollama 0.24.
Medians of 3 warm runs (`scripts/benchmark_latency.py --runs 3`), test
audio a 5.6 s sentence spoken by Piper.

| Stage | Setting | Time | Notes |
|---|---|---|---|
| TTS (Piper) | `en_US-*-medium` | first chunk **0.11 s**, whole sentence 0.19 s | 0.03× real time; CPU only |
| VAD (Silero) | 32 ms frames | **0.35 ms** per frame | ~1% of real time; CPU only |
| STT (faster-whisper) | `base`, CPU, int8 | **0.52 s** per clip | 0.09× real time |
| | `small`, CPU, int8 | 1.40 s | 0.25× |
| | `base`, CUDA, float16 | **0.14 s** | 0.03×; the config's `device: auto` picks this when CUDA is available |
| | `small`, CUDA, float16 | 0.21 s | 0.04× |
| Memory query | Chroma, warm | **0.15–0.19 s** | MemArk is within ~5%; see `docs/MEMARK_BACKEND.md` |
| LLM | `dolphin-llama3:8b` | first token 0.18 s, first sentence 0.33 s, **35 tok/s** | 92% on GPU; cold load 7.0 s |
| | `qwen2.5:7b-instruct` | first token 0.16 s, first sentence 0.36 s, **34 tok/s** | 92% on GPU; cold load 7.3 s |
| Agent tool turn | `qwen2.5:7b-instruct`, simple tool | tool call at **0.55–0.94 s**, first answer sentence **0.87–1.57 s** | e.g. "what day is it?" / "18% of 640"; the filler phrase plays at the tool call |

**Voice mode, time from you stopping talking to the first audio** (VAD
silence window + final STT pass + memory query + LLM first sentence +
TTS first chunk):

- plain reply: **~0.85 s** (dolphin 0.84 s, qwen 0.87 s; STT base on CUDA);
- reply that uses a tool: filler phrase at ~0.9–1.3 s, answer at
  ~1.2–1.9 s.

With STT on CPU instead (no CUDA), add ~0.4 s (`base`) or ~1.3 s (`small`).

Things these numbers don't include:
- the first reply after startup, which pays the LLM's cold load (~7 s);
- tools that call the network (`web_search`, `weather`), which add
  whatever the remote service takes;
- Coqui TTS. The only Coqui number so far: `tts_models/en/ljspeech/vits`
  on CPU took 7.9 s for its first ~4 s sentence in
  `scripts/check_offline.py`, model load included; warm per-sentence time
  wasn't measured, and neither was XTTS-v2.

## Tiers (extrapolated)

| Tier | Example hardware | What to expect |
|---|---|---|
| Minimum, text mode | 4+ CPU cores, 8 GB RAM, no GPU | 7B model on CPU: replies stream at a few tokens/s; usable for text, slow. A smaller model such as `qwen2.5:1.5b-instruct` (Apache-2.0) should be faster (not measured here). Note that `qwen2.5:3b-instruct` is research-licensed (see THIRD_PARTY_LICENSES.md). |
| Minimum, voice mode | 8+ cores, 16 GB RAM, no GPU | STT `base` on CPU is fine (0.09× real time measured above); the LLM on CPU is the bottleneck, so uncomment `llm.voice_mode_override` (`qwen2.5:1.5b-instruct`). Expect a few seconds before each reply. |
| Recommended | 6 GB+ NVIDIA GPU, 16 GB RAM (the machine above) | ~0.85 s to first audio in voice mode, 7–8B models at ~35 tok/s, tool turns under 2 s. |
| Full fidelity (XTTS-v2 voice cloning, larger Whisper, bigger LLMs) | 12 GB+ VRAM, 32 GB RAM | Not measured here. A 7B LLM (~4.7 GB) plus XTTS-v2 (a further ~2 GB) likely won't both fit on a 6 GB card. |

## Memory use

- A 7B model at Ollama's default 4-bit quantization is ~4.7 GB on disk and
  needs about that much VRAM or RAM; on the 6 GB card above, 92% of it fit
  on the GPU.
- faster-whisper `base` is ~150 MB, Piper voices ~60 MB each, Silero VAD
  ~2 MB.
- The agent model and the plain chat model are loaded separately if they
  differ (e.g. dolphin for chat, qwen for tool personas), so switching
  between them can cost a reload. Using the same tool-capable model for
  both avoids that.

## Open items

- Numbers from a CPU-only machine and from Apple Silicon (Metal).
- Coqui/XTTS-v2 timings.
