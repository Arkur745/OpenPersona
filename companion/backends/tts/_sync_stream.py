"""Bridge a synchronous, blocking audio generator into an async generator
without blocking the event loop between chunks.

Both Piper's and Coqui's Python APIs expose synthesis as a plain
(blocking, CPU/GPU-bound) generator — there is no asyncio-native streaming
API in either. This is the one place that generator-to-thread bridging is
implemented, shared by both backends rather than duplicated.
"""

import asyncio
import threading
from typing import AsyncIterator, Callable, Iterator

_SENTINEL = object()


async def iter_blocking_generator(
    make_generator: Callable[[], Iterator[bytes]],
    stop_event: threading.Event,
) -> AsyncIterator[bytes]:
    """Run `make_generator()` to completion on a background thread, yielding
    each item back on the event loop as it's produced.

    `stop_event` is checked on both sides: the producer thread stops
    pulling further items from the generator once it's set (so synthesis
    is actually cut short, not just ignored), and the consumer stops
    yielding once it's set (so a chunk produced right as `stop()` was
    called doesn't sneak out after the caller asked to abort).
    """
    loop = asyncio.get_running_loop()
    queue = _ThreadSafeQueue(loop)

    def _run() -> None:
        try:
            for item in make_generator():
                if stop_event.is_set():
                    break
                queue.put(item)
        except Exception as exc:  # noqa: BLE001 — propagated to the async side, not swallowed
            queue.put(exc)
            return
        queue.put(_SENTINEL)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            return
        if isinstance(item, BaseException):
            raise item
        if stop_event.is_set():
            return
        yield item


class _ThreadSafeQueue:
    """Minimal thread-to-asyncio handoff: `put` is called from the worker
    thread, `get` is awaited from the event loop. Just enough for this
    module's one-producer-one-consumer use — not a general-purpose queue.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()

    def put(self, item) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self):
        return await self._queue.get()
