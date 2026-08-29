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
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
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


async def _acquire(sm, g=(0, 0)):
    lock = sm._locks[g]
    await lock.acquire()
    return lock


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_normal_stream_delivers_all_and_releases(sm, no_meta):
    g = (0, 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(40)]
    resp = FakeResp(chunks, delay=0.005)

    gen = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
    )

    received = [c async for c in gen]

    assert received == chunks
    assert resp.closed
    assert not lock.locked(), "slot must be released after normal stream completion"
    sm.backends[0]["client"].save_slot.assert_awaited_once()


@pytest.mark.asyncio
async def test_slot_released_when_client_disconnects(sm, no_meta):
    """Client disconnect: gen is closed early; the reader must not block on a
    full queue and must release the slot."""
    g = (0, 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(64)]
    resp = FakeResp(chunks, delay=0.005)

    gen = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
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


@pytest.mark.asyncio
async def test_slot_released_when_consumer_vanishes(sm, no_meta, monkeypatch):
    """Defense in depth: consumer vanishes without aclose; the reader's put must
    not block forever."""
    monkeypatch.setattr(app_module, "STREAM_PUT_TIMEOUT", 0.2)
    g = (0, 0)
    lock = await _acquire(sm, g)
    chunks = [b"c%d" % i for i in range(64)]
    resp = FakeResp(chunks, delay=0.005)

    _ = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
    )
    # never consume, never close

    await _pump(2.0)

    assert not lock.locked(), "slot lock leaked when consumer vanished"
    assert resp.closed


@pytest.mark.asyncio
async def test_backend_error_mid_stream_releases_slot(sm, no_meta):
    g = (0, 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b", b"c"], delay=0.005, fail_after=2)

    gen = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
    )

    received = [c async for c in gen]

    assert received == [b"a", b"b"]
    assert resp.closed
    assert not lock.locked(), "slot must be released when backend stream errors"


@pytest.mark.asyncio
async def test_partial_stream_does_not_save_cache(sm, monkeypatch):
    """P1-4: backend stream error mid-stream: the partial KV cache must not be
    saved (it is useless for restore and wastes disk), but the slot is
    released."""
    import hashing

    write_meta_async = AsyncMock()
    monkeypatch.setattr(hashing, "write_meta_async", write_meta_async)
    g = (0, 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b", b"c"], delay=0.005, fail_after=2)

    gen = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
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
    g = (0, 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.005)

    gen = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
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
    g = (0, 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.005)

    gen = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
    )
    _ = [c async for c in gen]
    await _pump(0.2)

    write_meta_async.assert_not_awaited()
    assert not lock.locked()


@pytest.mark.asyncio
async def test_reader_task_kept_alive_until_done(sm, no_meta):
    """The reader task must be tracked with a strong reference until it completes
    (otherwise the event loop's weak refs allow GC mid-execution)."""
    g = (0, 0)
    lock = await _acquire(sm, g)
    resp = FakeResp([b"a", b"b"], delay=0.01)

    gen = await app_module.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm
    )

    tasks = getattr(app_module, "_READER_TASKS", None)
    assert tasks is not None, "reader tasks must be tracked with strong references"
    assert len(tasks) == 1, "in-flight reader task must be tracked"

    _ = [c async for c in gen]
    await _pump(0.2)

    assert len(tasks) == 0, "completed reader tasks must be untracked"
    assert not lock.locked(), "slot must be released after tracked task completes"
