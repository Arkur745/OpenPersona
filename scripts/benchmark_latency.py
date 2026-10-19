"""Per-stage latency benchmark for the voice pipeline, on this machine.

Usage:

    python scripts/benchmark_latency.py [--config PATH] [--persona ID]
        [--models m1,m2,...] [--runs N] [--json OUT]

Measures each stage on real backends built through the same
`companion.app.build_*` functions the app uses, using a real Piper
voice as test audio:

- **TTS (Piper):** time to the first audio chunk and the total for one
  ~5 s sentence, plus the real-time factor.
- **VAD (Silero):** mean time per 32 ms frame.
- **STT (faster-whisper):** model load (cold), then a one-shot
  transcription of that clip on CPU (`int8`) and on CUDA if available,
  for `base` and `small`.
- **LLM (Ollama):** for each model, load time, time to first token, time
  to the first complete sentence (what voice mode waits on before TTS can
  start), and generation tokens/s from Ollama's own counters. Each model
  gets one warm-up call, then `--runs` measured calls; medians are
  reported. Models that aren't pulled are skipped.

- **Memory:** one warm `query()` on the configured backend (a throwaway
  store with a few turns in it), since every turn queries memory before
  calling the LLM. `scripts/benchmark_memory_backends.py` goes deeper.

It ends with an estimate of voice mode's response latency: from when
you stop talking to the first audio. That is the sum of the VAD silence
window, the final STT pass, the memory query, the LLM's time to its first
sentence and TTS's first chunk.

Run on an otherwise idle machine; the numbers are wall-clock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_offline import _resample_pcm16  # sibling script in scripts/

SENTENCE = (
    "I was thinking about what you said yesterday, and I think you're right. "
    "Let's try the new plan tomorrow morning."
)
PROMPT_SYSTEM = (
    "You are a calm, plainspoken mentor. Speak in short sentences and reply "
    "in two or three sentences."
)
PROMPT_USER = "I keep putting off starting my side project. Any advice?"
SENTENCE_END = (".", "!", "?")


def _cpu_and_ram() -> tuple[str, float | None]:
    """platform.processor() is only a family/model string on Windows, so
    ask the OS for the marketing name and total RAM where we can."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 ("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name; "
                  "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")],
                capture_output=True, text=True, timeout=20, check=False,
            ).stdout.split("\n")
            return out[0].strip(), round(int(out[1].strip()) / 2**30, 1)
        if Path("/proc/cpuinfo").exists():
            name = next(line.split(":", 1)[1].strip()
                        for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name"))
            kb = int(Path("/proc/meminfo").read_text().split()[1])
            return name, round(kb / 2**20, 1)
    except (OSError, ValueError, IndexError, StopIteration, subprocess.TimeoutExpired):
        pass
    return platform.processor() or platform.machine(), None


def _hardware() -> dict:
    cpu, ram_gb = _cpu_and_ram()
    info = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu": cpu, "ram_gb": ram_gb}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        info["gpu"] = out or None
    except (OSError, subprocess.TimeoutExpired):
        info["gpu"] = None
    return info


async def bench_tts(config, persona) -> tuple[dict, bytes]:
    from companion.app import build_tts_for_persona

    tts = build_tts_for_persona(config, persona)
    async for _ in tts.synthesize_stream("Warm up."):
        pass
    start = time.perf_counter()
    first = None
    chunks = []
    async for chunk in tts.synthesize_stream(SENTENCE):
        if first is None:
            first = time.perf_counter() - start
        chunks.append(chunk)
    total = time.perf_counter() - start
    pcm = b"".join(chunks)
    audio_s = len(pcm) / 2 / tts.sample_rate
    result = {"voice": persona.voice_id, "first_chunk_s": first, "total_s": total,
              "audio_s": audio_s, "rtf": total / audio_s}
    return result, _resample_pcm16(pcm, tts.sample_rate, 16000)


def bench_vad(config, speech_16k: bytes) -> dict:
    from companion.app import build_vad

    vad = build_vad(config)
    frame = vad.frame_bytes
    frames = [speech_16k[i:i + frame] for i in range(0, len(speech_16k) - frame + 1, frame)]
    vad.is_speech(frames[0])  # model load
    vad.reset()
    start = time.perf_counter()
    for f in frames:
        vad.is_speech(f)
    per_frame = (time.perf_counter() - start) / len(frames)
    return {"frames": len(frames), "per_frame_ms": per_frame * 1000,
            "frame_ms": vad.frame_samples / 16000 * 1000}


async def bench_stt(speech_16k: bytes, runs: int) -> list[dict]:
    import ctranslate2

    from companion.backends.stt.whisper_local import FasterWhisperSTT

    devices = ["cpu"] + (["cuda"] if ctranslate2.get_cuda_device_count() > 0 else [])
    audio_s = len(speech_16k) / 2 / 16000
    results = []
    for device in devices:
        for size in ("base", "small"):
            compute = "int8" if device == "cpu" else "float16"
            stt = FasterWhisperSTT(model_size=size, device=device, compute_type=compute,
                                   language="en", min_refresh_seconds=0.0)
            start = time.perf_counter()
            try:
                await stt._ensure_model()
            except Exception as exc:  # noqa: BLE001 — report, don't abort the benchmark
                results.append({"model": size, "device": device, "error": str(exc)})
                continue
            load_s = time.perf_counter() - start
            times, text = [], ""
            for _ in range(runs + 1):  # first pass is a warm-up
                stt.reset()
                start = time.perf_counter()
                text = await stt.transcribe_stream(speech_16k) or ""
                times.append(time.perf_counter() - start)
            warm = statistics.median(times[1:])
            results.append({"model": size, "device": device, "compute_type": compute,
                            "load_s": load_s, "transcribe_s": warm, "rtf": warm / audio_s,
                            "text": text})
    return results


async def bench_memory(config, runs: int) -> dict:
    import tempfile
    from types import SimpleNamespace

    from companion.app import build_memory
    from companion.config.loader import BackendConfig
    from companion.core.interfaces import Turn

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = build_memory(SimpleNamespace(memory=BackendConfig(
            config.memory.backend, {**config.memory.options, "persist_dir": tmp})))
        for text in ("I'm learning Rust.", "My sister Maya lives in Lisbon.",
                     "I run every Sunday.", "I'm allergic to peanuts."):
            await store.write("bench", "bench", Turn(role="user", text=text))
        times = []
        for _ in range(runs + 1):  # first query is a warm-up
            start = time.perf_counter()
            await store.query("bench", "bench", PROMPT_USER, k=5)
            times.append(time.perf_counter() - start)
    return {"backend": config.memory.backend, "query_s": statistics.median(times[1:])}


async def bench_llm(host: str, models: list[str], runs: int) -> list[dict]:
    from ollama import AsyncClient, ResponseError

    client = AsyncClient(host=host)
    pulled = {m.model for m in (await client.list()).models}
    messages = [{"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": PROMPT_USER}]
    results = []
    for model in models:
        if model not in pulled:
            results.append({"model": model, "skipped": "not pulled"})
            continue
        await client.generate(model=model, keep_alive=0)  # unload, so the warm-up measures a cold load
        samples = []
        for i in range(runs + 1):
            start = time.perf_counter()
            ttft = ttfs = None
            text = ""
            final = None
            async for chunk in await client.chat(model=model, messages=messages, stream=True,
                                                 options={"temperature": 0.7, "num_predict": 160}):
                piece = chunk.message.content if chunk.message else ""
                if piece and ttft is None:
                    ttft = time.perf_counter() - start
                text += piece or ""
                if ttfs is None and any(p in text for p in SENTENCE_END):
                    ttfs = time.perf_counter() - start
                if chunk.done:
                    final = chunk
            sample = {"ttft_s": ttft, "first_sentence_s": ttfs,
                      "total_s": time.perf_counter() - start,
                      "tokens_per_s": final.eval_count / (final.eval_duration / 1e9),
                      "load_s": final.load_duration / 1e9}
            if i == 0:
                cold = sample
            else:
                samples.append(sample)
        try:
            ps = (await client.ps()).models
            vram = next((m.size_vram for m in ps if m.model == model), None)
            size = next((m.size for m in ps if m.model == model), None)
            gpu_share = None if not size else round(100 * (vram or 0) / size)
        except ResponseError:
            gpu_share = None
        results.append({
            "model": model,
            "cold_load_s": cold["load_s"],
            "cold_ttft_s": cold["ttft_s"],
            "ttft_s": statistics.median(s["ttft_s"] for s in samples),
            "first_sentence_s": statistics.median(s["first_sentence_s"] or s["total_s"] for s in samples),
            "tokens_per_s": statistics.median(s["tokens_per_s"] for s in samples),
            "gpu_percent": gpu_share,
        })
    return results


def _fmt(value, unit="s", digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}{unit}"


async def run(args) -> dict:
    from companion.app import load_personas, select_persona
    from companion.config.loader import load_config, resolve_config_path

    config = load_config(resolve_config_path(args.config))
    persona = select_persona(load_personas(config), args.persona)
    models = args.models.split(",") if args.models else [config.llm.options.get("model")]
    report = {"hardware": _hardware()}

    print("TTS...", flush=True)
    report["tts"], speech = await bench_tts(config, persona)
    print("VAD...", flush=True)
    report["vad"] = bench_vad(config, speech)
    print("STT...", flush=True)
    report["stt"] = await bench_stt(speech, args.runs)
    print("Memory...", flush=True)
    report["memory"] = await bench_memory(config, args.runs)
    print("LLM...", flush=True)
    report["llm"] = await bench_llm(config.llm.options.get("host", "http://localhost:11434"),
                                    models, args.runs)

    from companion.backends.stt.whisper_local import _resolve_device

    vad_silence_s = config.vad.options.get("min_silence_duration_ms", 100.0) / 1000
    stt_device = _resolve_device(config.stt.options.get("device", "auto"))
    stt_default = next((s for s in report["stt"] if s.get("model") == config.stt.options.get(
        "model_size", "base") and "transcribe_s" in s and s["device"] == stt_device), None)
    estimates = []
    for llm in report["llm"]:
        if "ttft_s" not in llm or stt_default is None:
            continue
        total = (vad_silence_s + stt_default["transcribe_s"] + report["memory"]["query_s"]
                 + llm["first_sentence_s"] + report["tts"]["first_chunk_s"])
        estimates.append({"model": llm["model"],
                          "stt": f"{stt_default['model']}/{stt_default['device']}",
                          "time_to_first_audio_s": total})
    report["voice_estimate"] = estimates
    return report


def print_report(r: dict) -> None:
    hw = r["hardware"]
    print(f"\nHardware: {hw['cpu']} | RAM {hw['ram_gb']} GB | GPU: {hw['gpu']} | {hw['os']} | "
          f"Python {hw['python']}\n")
    t = r["tts"]
    print(f"TTS {t['voice']}: first chunk {_fmt(t['first_chunk_s'])}, total {_fmt(t['total_s'])} "
          f"for {t['audio_s']:.1f}s of audio (RTF {t['rtf']:.2f})")
    v = r["vad"]
    print(f"VAD: {v['per_frame_ms']:.2f} ms per {v['frame_ms']:.0f} ms frame")
    m = r["memory"]
    print(f"Memory ({m['backend']}): {m['query_s'] * 1000:.0f} ms per query\n")
    print("| STT model | device | load | transcribe 1 clip | RTF | transcript |")
    print("|---|---|---|---|---|---|")
    for s in r["stt"]:
        if "error" in s:
            print(f"| {s['model']} | {s['device']} | error: {s['error'][:60]} | | | |")
            continue
        print(f"| {s['model']} | {s['device']} ({s['compute_type']}) | {_fmt(s['load_s'])} | "
              f"{_fmt(s['transcribe_s'])} | {s['rtf']:.2f} | {s['text'][:50]} |")
    print("\n| LLM | cold load | TTFT | first sentence | tokens/s | on GPU |")
    print("|---|---|---|---|---|---|")
    for m in r["llm"]:
        if "skipped" in m:
            print(f"| {m['model']} | skipped ({m['skipped']}) | | | | |")
            continue
        gpu = "—" if m["gpu_percent"] is None else f"{m['gpu_percent']}%"
        print(f"| {m['model']} | {_fmt(m['cold_load_s'])} | {_fmt(m['ttft_s'])} | "
              f"{_fmt(m['first_sentence_s'])} | {m['tokens_per_s']:.1f} | {gpu} |")
    print("\nVoice mode, estimated time from you stopping talking to first audio:")
    for e in r["voice_estimate"]:
        print(f"  {e['model']} (STT {e['stt']}): {_fmt(e['time_to_first_audio_s'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    parser.add_argument("--persona", default=None)
    parser.add_argument("--models", default=None, help="comma-separated Ollama models")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json", default=None, help="also write the raw report here")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
