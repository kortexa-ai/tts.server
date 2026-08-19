import asyncio
import threading

import pytest

from kortexa.tts.server import _disconnect_safe_stream


@pytest.mark.asyncio
async def test_disconnected_stream_releases_inference_for_the_next_request():
    semaphore = asyncio.Semaphore(1)
    background_tasks: set[asyncio.Task[None]] = set()
    first_chunk_started = threading.Event()
    allow_first_chunk = threading.Event()

    def interrupted_content():
        first_chunk_started.set()
        assert allow_first_chunk.wait(timeout=2)
        yield b"first"
        yield b"discarded"

    stream = _disconnect_safe_stream(
        interrupted_content(),
        semaphore,
        background_tasks,
    )
    pending_chunk = asyncio.create_task(anext(stream))
    assert await asyncio.to_thread(first_chunk_started.wait, 2)
    pending_chunk.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending_chunk
    allow_first_chunk.set()

    async def next_content():
        chunks = []
        async for chunk in _disconnect_safe_stream(
            [b"next"],
            semaphore,
            background_tasks,
        ):
            chunks.append(chunk)
        return chunks

    assert await asyncio.wait_for(next_content(), timeout=2) == [b"next"]
    if background_tasks:
        await asyncio.wait_for(
            asyncio.gather(*background_tasks),
            timeout=2,
        )
