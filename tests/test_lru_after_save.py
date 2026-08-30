# tests/test_lru_after_save.py

"""LRU check after each slot write (save):

- a successful big save (stream and non-stream) triggers the .bin LRU
  cleanup (clean_bin_cache) in the background, without delaying the response;
- a failed save and small requests do not trigger it;
- a disabled .bin cache (empty dir or max_mb <= 0) never triggers it;
- a second save while a check is in flight does not start a second one.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import bin_cache
import chat_flow


class FakeResp:
    """Mimics the httpx.Response parts used by the reader: aiter_raw + aclose."""

    def __init__(self, chunks, delay=0.0):
        self._chunks = list(chunks)
        self._delay = delay

    async def aiter_raw(self):
        for c in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield c

    async def aclose(self):
        pass


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


@pytest.fixture(autouse=True)
def lru_env(monkeypatch):
    """Enabled .bin cache, a clean_bin_cache spy, and a fresh in-flight flag."""
    monkeypatch.setattr(chat_flow, "BIN_CACHE_DIR", "/tmp/bin")
    monkeypatch.setattr(chat_flow, "BIN_CACHE_MAX_MB", 100)
    clean = MagicMock()
    monkeypatch.setattr(bin_cache, "clean_bin_cache", clean)
    monkeypatch.setattr(chat_flow, "_lru_check_in_flight", False)
    return clean


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


async def _chat(sm, content, stream=False):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    return await app_module.chat(FakeRequest(data))


@pytest.mark.asyncio
async def test_big_json_save_triggers_lru_check(sm, meta_dir, lru_env, monkeypatch):
    """A successful big non-stream save triggers exactly one LRU check with
    the configured dir and cap."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    await _chat(sm, "hello world")
    await _pump()
    lru_env.assert_called_once_with("/tmp/bin", 100)


@pytest.mark.asyncio
async def test_big_stream_save_triggers_lru_check(sm, meta_dir, lru_env):
    """A successful big stream save triggers exactly one LRU check."""
    g = (0, "model", 0)
    lock = sm._lock_for(g)
    await lock.acquire()
    resp = FakeResp([b"a", b"b"], delay=0.005)

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True
    )
    _ = [c async for c in gen]
    await _pump()

    lru_env.assert_called_once_with("/tmp/bin", 100)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_failed_save_does_not_trigger_lru_check(
    sm, meta_dir, lru_env, monkeypatch
):
    """A failed slot save must not trigger the LRU check."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    sm.backends[0]["client"].save_slot = AsyncMock(return_value=False)
    await _chat(sm, "hello world")
    await _pump()
    lru_env.assert_not_called()


@pytest.mark.asyncio
async def test_small_request_does_not_trigger_lru_check(sm, meta_dir, lru_env):
    """Small requests never save, so they never trigger the LRU check."""
    await _chat(sm, "small")
    await _pump()
    lru_env.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_bin_cache_never_triggers(sm, meta_dir, lru_env, monkeypatch):
    """An empty BIN_CACHE_DIR disables the .bin cache entirely."""
    monkeypatch.setattr(chat_flow, "BIN_CACHE_DIR", "")
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    await _chat(sm, "hello world")
    await _pump()
    lru_env.assert_not_called()


@pytest.mark.asyncio
async def test_zero_max_mb_never_triggers(sm, meta_dir, lru_env, monkeypatch):
    """max_mb <= 0 disables the size cap, hence the LRU check."""
    monkeypatch.setattr(chat_flow, "BIN_CACHE_MAX_MB", 0)
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    await _chat(sm, "hello world")
    await _pump()
    lru_env.assert_not_called()


@pytest.mark.asyncio
async def test_in_flight_check_is_not_duplicated(lru_env, monkeypatch):
    """A second save while a check is running must not start a second one;
    the in-flight flag must reset when the check finishes."""
    started = threading.Event()
    release = threading.Event()
    slow_calls = []

    def slow_clean(dir, max_mb):
        slow_calls.append(1)
        started.set()
        release.wait(5)

    monkeypatch.setattr(bin_cache, "clean_bin_cache", slow_clean)

    chat_flow._schedule_lru_check()
    # Yield control so the task starts the worker thread (which sets
    # `started`); blocking the loop here would deadlock the test.
    deadline = time.time() + 5
    while not started.is_set() and time.time() < deadline:
        await asyncio.sleep(0.01)
    assert started.is_set(), "the first check must start"
    chat_flow._schedule_lru_check()  # skipped: a check is already in flight
    release.set()
    await _pump()

    assert len(slow_calls) == 1, "a second concurrent check must be skipped"
    assert not chat_flow._lru_check_in_flight, "the in-flight flag must reset"
