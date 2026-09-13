# tests/test_stream_reader.py

"""P0-1: stream reader must never leak the slot lock.

Scenarios:
- normal completion: all chunks delivered, slot released, save called;
- client disconnect (generator closed early): reader must not block on a full
  queue, slot must be released, backend response closed;
- consumer vanishes without aclose: reader's put must not block forever;
- reader task must be kept alive with a strong reference until it completes.
"""

import asyncio
import gc
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import chat_flow
import slot_manager as sm_module
from slot_manager import SlotManager


class FakeResp:
    """Mimics the httpx.Response parts used by the reader: aiter_raw + aclose."""

    def __init__(self, chunks, delay=0.0, fail_after=None):
        self._chunks = list(chunks)
        self._delay = delay
        self._fail_after = fail_after
        self.closed = False

    async def aiter_raw(self):
        for i, c in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError("backend stream error")
            if self._delay:
                await asyncio.sleep(self._delay)
            yield c

    async def aclose(self):
        self.closed = True


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    manager.set_clients([client])
    return manager


@pytest.fixture()
def no_meta(monkeypatch):
    import hashing

    monkeypatch.setattr(hashing, "write_meta", lambda *a, **k: None)


async def _acquire(sm, g=(0, "model", 0)):
    lock = sm._lock_for(g)
    await lock.acquire()
    return lock


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_normal_stream_delivers_all_and_releases(sm, no_meta):
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(40)]
    resp = FakeResp(chunks, delay=0.005)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )

    received = [c async for c in gen]

    assert received == chunks
    assert not any(c.startswith(b"data: ") and b'"error"' in c for c in received), (
        "clean completion must not emit an error event"
    )
    assert resp.closed
    assert not lock.locked(), "slot must be released after normal stream completion"
    sm.backends[0]["client"].save_slot.assert_awaited_once()


@pytest.mark.asyncio
async def test_slot_released_when_client_disconnects(sm, no_meta, monkeypatch):
    """Client disconnect: gen is closed early; the reader must not block on a
    full queue and must release the slot."""
    # Capture the queue instance so its contents can be inspected after the
    # reader task is cancelled (the queue is owned by start_stream_task's
    # frame and is not reachable from the generator).
    queue_ref: dict = {}
    real_queue = asyncio.Queue

    def capturing_queue(*args, **kwargs):
        q = real_queue(*args, **kwargs)
        queue_ref["q"] = q
        return q

    monkeypatch.setattr(chat_flow.asyncio, "Queue", capturing_queue)

    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(64)]
    resp = FakeResp(chunks, delay=0.005)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )

    it = gen.__aiter__()
    for _ in range(3):
        await it.__anext__()
    # client disconnect:
    await it.aclose()

    gc.collect()
    await _pump(0.5)

    assert not lock.locked(), "slot lock leaked after client disconnect"
    assert resp.closed
    # Cancellation is not a backend error: no error event may be queued.
    # The queue is owned by start_stream_task's frame (kept alive by the
    # reader task until it finishes), so wrap asyncio.Queue to capture the
    # reference and inspect its contents after the reader is cancelled.
    assert queue_ref["q"] is not None, "queue reference must be captured"
    queued = []
    while not queue_ref["q"].empty():
        queued.append(queue_ref["q"].get_nowait())
    assert not any(
        c is not None and c.startswith(b"data: ") and b'"error"' in c
        for c in queued
    ), "client-disconnect cancellation must not emit an error event"


@pytest.mark.asyncio
async def test_slot_released_when_consumer_vanishes(sm, no_meta, monkeypatch):
    """Defense in depth: consumer vanishes without aclose; the reader's put must
    not block forever."""
    monkeypatch.setattr(chat_flow, "STREAM_PUT_TIMEOUT", 0.2)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(64)]
    resp = FakeResp(chunks, delay=0.005)

    _ = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )
    # never consume, never close

    await _pump(2.0)

    assert not lock.locked(), "slot lock leaked when consumer vanished"
    assert resp.closed


@pytest.mark.asyncio
async def test_backend_error_mid_stream_releases_slot(sm, no_meta):
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b", b"c"], delay=0.005, fail_after=2)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )

    received = [c async for c in gen]

    assert received[:2] == [b"a", b"b"]
    assert len(received) == 3, "mid-stream error must yield an SSE error event"
    err = received[2]
    assert err.startswith(b"data: ") and err.endswith(b"\n\n")
    payload = json.loads(err[len(b"data: "):-2])
    assert "error" in payload
    assert "stream interrupted" in payload["error"]
    assert resp.closed
    assert not lock.locked(), "slot must be released when backend stream errors"


class CancelResp:
    """Backend that cancels its own stream mid-way: aiter_raw raises
    asyncio.CancelledError after the first chunk (simulates a transport or
    backend cancel surfacing as CancelledError inside the reader)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    async def aiter_raw(self):
        for i, c in enumerate(self._chunks):
            if i == 1:
                raise asyncio.CancelledError()
            yield c

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_backend_cancel_mid_stream_emits_error_event(sm, no_meta):
    """A CancelledError raised from inside resp.aiter_raw() is a backend
    failure, not a client disconnect: the reader must still push the SSE
    error event before the sentinel (regression: the cancel path used to
    re-raise silently and truncate the stream)."""
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = CancelResp([b"a", b"b", b"c"])

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )

    received = [c async for c in gen]

    assert received[:1] == [b"a"]
    assert len(received) == 2, (
        "backend cancel mid-stream must yield an SSE error event"
    )
    err = received[1]
    assert err.startswith(b"data: ") and err.endswith(b"\n\n")
    payload = json.loads(err[len(b"data: "):-2])
    assert "error" in payload
    assert "stream interrupted" in payload["error"]
    assert resp.closed
    assert not lock.locked(), "slot must be released when backend cancels the stream"


@pytest.mark.asyncio
async def test_small_stream_does_not_save_cache(sm, monkeypatch):
    """P1-6: small stream requests must not pollute the disk cache, but the
    slot is still released and chunks are delivered."""
    import hashing

    write_meta_async = AsyncMock()
    monkeypatch.setattr(hashing, "write_meta_async", write_meta_async)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.005)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=False
    )
    received = [c async for c in gen]

    assert received == [b"a", b"b"]
    assert not lock.locked()
    sm.backends[0]["client"].save_slot.assert_not_awaited()
    write_meta_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_big_stream_saves_cache(sm, monkeypatch):
    """P1-6: big stream requests still save the cache on completion."""
    import hashing

    write_meta_async = AsyncMock()
    monkeypatch.setattr(hashing, "write_meta_async", write_meta_async)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.005)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )
    _ = [c async for c in gen]

    assert not lock.locked()
    sm.backends[0]["client"].save_slot.assert_awaited_once()
    write_meta_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_partial_stream_does_not_save_cache(sm, monkeypatch):
    """P1-4: backend stream error mid-stream: the partial KV cache must not be
    saved (it is useless for restore and wastes disk), but the slot is
    released."""
    import hashing

    write_meta_async = AsyncMock()
    monkeypatch.setattr(hashing, "write_meta_async", write_meta_async)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b", b"c"], delay=0.005, fail_after=2)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )
    _ = [c async for c in gen]

    assert not lock.locked(), "slot must be released when backend stream errors"
    sm.backends[0]["client"].save_slot.assert_not_awaited()
    write_meta_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_meta_written_when_save_succeeds(sm, monkeypatch):
    """P1-1: meta file is written only when the slot save succeeded."""
    import hashing

    write_meta_async = AsyncMock()
    monkeypatch.setattr(hashing, "write_meta_async", write_meta_async)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.005)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )
    _ = [c async for c in gen]
    await _pump(0.2)

    write_meta_async.assert_awaited_once()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_meta_not_written_when_save_fails(sm, monkeypatch):
    """P1-1: a failed slot save must not leave a meta file pointing at a
    cache that was never saved."""
    import hashing

    write_meta_async = AsyncMock()
    monkeypatch.setattr(hashing, "write_meta_async", write_meta_async)
    sm.backends[0]["client"].save_slot = AsyncMock(return_value=False)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.005)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )
    _ = [c async for c in gen]
    await _pump(0.2)

    write_meta_async.assert_not_awaited()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_reader_task_kept_alive_until_done(sm, no_meta):
    """The reader task must be tracked with a strong reference until it completes
    (otherwise the event loop's weak refs allow GC mid-execution)."""
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.01)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )

    tasks = getattr(chat_flow, "_READER_TASKS", None)
    assert tasks is not None, "reader tasks must be tracked with strong references"
    assert len(tasks) == 1, "in-flight reader task must be tracked"

    _ = [c async for c in gen]
    await _pump(0.2)

    assert len(tasks) == 0, "completed reader tasks must be untracked"
    assert not lock.locked(), "slot must be released after tracked task completes"


@pytest.mark.asyncio
async def test_put_timeout_emits_error_event_before_sentinel(sm, no_meta, monkeypatch):
    """A stalled consumer (queue never drained) must cause the reader to emit
    an SSE error event before the sentinel, so the client can tell the stream
    was truncated."""
    # Capture the queue instance (owned by start_stream_task's frame) so its
    # contents can be inspected after the reader finishes.
    queue_ref: dict = {}
    real_queue = asyncio.Queue

    def capturing_queue(*args, **kwargs):
        q = real_queue(*args, **kwargs)
        queue_ref["q"] = q
        return q

    monkeypatch.setattr(chat_flow.asyncio, "Queue", capturing_queue)
    monkeypatch.setattr(chat_flow, "STREAM_PUT_TIMEOUT", 1.0)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(40)]
    resp = FakeResp(chunks, delay=0.05)

    _ = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )

    # Stalled consumer: drain a few items, then stop draining so the queue
    # fills and the reader's put times out (~t=2.0). Only after the timeout,
    # free two slots (one for the SSE error event, one for the sentinel) and
    # stall again; freeing them earlier would let the reader push more chunks.
    q = queue_ref["q"]
    for _ in range(4):
        await q.get()
    await asyncio.sleep(2.1)
    for _ in range(2):
        await q.get()
    await asyncio.sleep(2.0)

    assert not lock.locked(), "slot must be released when the consumer stalled"
    assert resp.closed
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    assert items and items[-1] is None, "the last queued item must be the sentinel"
    err = items[-2]
    assert err.startswith(b"data: ") and err.endswith(b"\n\n"), (
        "the item before the sentinel must be the SSE error event"
    )
    payload = json.loads(err[len(b"data: "):-2])
    assert "error" in payload
    assert "stream interrupted" in payload["error"]


@pytest.mark.asyncio
async def test_put_timeout_does_not_save_partial_stream(sm, monkeypatch):
    """A queue put timeout means the stream was not read to the end: the
    partial KV cache must not be saved."""
    import hashing

    write_meta_async = AsyncMock()
    monkeypatch.setattr(hashing, "write_meta_async", write_meta_async)
    monkeypatch.setattr(chat_flow, "STREAM_PUT_TIMEOUT", 0.1)
    g = (0, "model", 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(64)]
    resp = FakeResp(chunks, delay=0.005)

    _ = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )
    await _pump(1.0)

    assert not lock.locked(), "slot must be released when the consumer vanished"
    assert resp.closed
    write_meta_async.assert_not_awaited()
