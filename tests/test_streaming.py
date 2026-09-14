import asyncio
import threading

import pytest

from kortexa.tts.server import _disconnect_safe_stream, create_app


def test_media_facade_routes_are_registered():
    paths = {route.path for route in create_app().routes}

    assert "/generate" in paths
    assert "/ws" in paths


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


@pytest.mark.asyncio
async def test_iterator_closes_before_next_inference_is_admitted():
    semaphore = asyncio.Semaphore(1)
    tasks = set()
    closing = threading.Event()
    allow_close = threading.Event()
    next_started = threading.Event()

    def first_content():
        try:
            yield b"first"
        finally:
            closing.set()
            assert allow_close.wait(2)

    def next_content():
        next_started.set()
        yield b"next"

    stream = _disconnect_safe_stream(first_content(), semaphore, tasks)
    try:
        assert await anext(stream) == b"first"
        await stream.aclose()
        assert await asyncio.to_thread(closing.wait, 2)
        following = _disconnect_safe_stream(next_content(), semaphore, tasks)
        pending = asyncio.create_task(anext(following))
        await asyncio.sleep(0)
        assert semaphore.locked()
        assert not next_started.is_set()
        allow_close.set()
        assert await asyncio.wait_for(pending, 2) == b"next"
        await following.aclose()
    finally:
        allow_close.set()
        await stream.aclose()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["iter", "next", "close"])
async def test_iterator_errors_reach_consumer_without_hanging(failure):
    class BrokenIterator:
        def __iter__(self):
            if failure == "iter":
                raise RuntimeError("iter failed")
            return self

        def __next__(self):
            if failure == "next":
                raise RuntimeError("next failed")
            raise StopIteration

        def close(self):
            if failure == "close":
                raise RuntimeError("close failed")

    semaphore = asyncio.Semaphore(1)
    tasks = set()
    stream = _disconnect_safe_stream(BrokenIterator(), semaphore, tasks)
    try:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            await asyncio.wait_for(anext(stream), 2)
    finally:
        await stream.aclose()
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert not semaphore.locked()


@pytest.mark.asyncio
async def test_stream_keeps_thread_local_context_through_iteration_and_close(
    monkeypatch,
):
    # Force each executor dispatch onto a fresh thread, exposing accidental
    # migration that the default pool might otherwise hide by reusing a worker.
    loop = asyncio.get_running_loop()
    threads = []

    async def new_thread(func, *args, **kwargs):
        future = loop.create_future()

        def run():
            try:
                result = func(*args, **kwargs)
            except BaseException as error:
                loop.call_soon_threadsafe(future.set_exception, error)
            else:
                loop.call_soon_threadsafe(future.set_result, result)

        thread = threading.Thread(target=run)
        threads.append(thread)
        thread.start()
        return await future

    monkeypatch.setattr(asyncio, "to_thread", new_thread)
    state = threading.local()

    def content():
        state.inference = True
        try:
            yield b"one"
            assert state.inference
            yield b"two"
        finally:
            assert state.inference
            del state.inference

    tasks = set()
    stream = _disconnect_safe_stream(content(), asyncio.Semaphore(1), tasks)
    assert [chunk async for chunk in stream] == [b"one", b"two"]
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), 2)
    for thread in threads:
        thread.join(timeout=2)
