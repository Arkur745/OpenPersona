"""Tests for companion/backends/tts/_sync_stream.py — the sync-generator-
to-async-generator bridge shared by PiperTTS and CoquiTTS. Pure asyncio/
threading logic, no TTS library involved."""

import asyncio
import threading
import time

import pytest

from companion.backends.tts._sync_stream import iter_blocking_generator


@pytest.mark.asyncio
async def test_yields_items_in_order():
    def make_gen():
        yield b"a"
        yield b"b"
        yield b"c"

    received = [item async for item in iter_blocking_generator(make_gen, threading.Event())]

    assert received == [b"a", b"b", b"c"]


@pytest.mark.asyncio
async def test_empty_generator_yields_nothing():
    def make_gen():
        return iter(())

    received = [item async for item in iter_blocking_generator(make_gen, threading.Event())]

    assert received == []


@pytest.mark.asyncio
async def test_exception_in_generator_propagates_to_caller():
    def make_gen():
        yield b"a"
        raise RuntimeError("boom")

    received = []
    with pytest.raises(RuntimeError, match="boom"):
        async for item in iter_blocking_generator(make_gen, threading.Event()):
            received.append(item)

    assert received == [b"a"]


@pytest.mark.asyncio
async def test_stop_event_set_before_start_yields_nothing():
    stop_event = threading.Event()
    stop_event.set()

    def make_gen():
        yield b"a"
        yield b"b"

    received = [item async for item in iter_blocking_generator(make_gen, stop_event)]

    assert received == []


@pytest.mark.asyncio
async def test_stop_event_halts_further_yields_mid_stream():
    stop_event = threading.Event()

    def make_gen():
        yield b"a"
        yield b"b"
        yield b"c"

    received = []
    async for item in iter_blocking_generator(make_gen, stop_event):
        received.append(item)
        if len(received) == 1:
            stop_event.set()

    assert received == [b"a"]


def test_stop_event_actually_interrupts_the_producer_thread():
    """Not just that the consumer stops yielding (covered above) — the
    producer thread must stop pulling further items from the underlying
    generator too, so synthesis is actually cut short."""
    stop_event = threading.Event()
    produced = []

    def make_gen():
        for i in range(5):
            produced.append(i)
            yield str(i).encode()
            time.sleep(0.05)  # give the consumer a chance to call stop()

    async def run():
        received = []
        async for item in iter_blocking_generator(make_gen, stop_event):
            received.append(item)
            stop_event.set()
        return received

    received = asyncio.run(run())
    time.sleep(0.2)  # let the producer thread notice stop_event and exit

    assert received == [b"0"]
    assert len(produced) <= 2
