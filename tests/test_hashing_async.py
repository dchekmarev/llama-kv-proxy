# tests/test_hashing_async.py

"""P0-3: meta file I/O must not block the event loop."""

import asyncio
import threading
import time

import pytest

import hashing as hs


@pytest.mark.asyncio
async def test_find_best_restore_candidate_runs_in_thread_and_keeps_loop_responsive(
    monkeypatch,
):
    info = {}

    def blocking_scan():
        info["thread"] = threading.current_thread().ident
        time.sleep(0.3)
        return []

    monkeypatch.setattr(hs, "scan_all_meta", blocking_scan)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(10):
            ticks += 1
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    result = await hs.find_best_restore_candidate_async(["b1"], [], 100, 0.6, "model")
    await hb

    assert result is None
    assert info["thread"] != threading.main_thread().ident, (
        "scan must run off the event loop thread"
    )
    assert ticks >= 3, f"event loop was blocked during scan: {ticks} heartbeats"


@pytest.mark.asyncio
async def test_write_meta_runs_in_thread(monkeypatch):
    info = {}

    def blocking_write(*a, **k):
        info["thread"] = threading.current_thread().ident
        time.sleep(0.2)

    monkeypatch.setattr(hs, "write_meta", blocking_write)

    await hs.write_meta_async("key", "prefix", ["b"], 100, "model")

    assert info["thread"] != threading.main_thread().ident


@pytest.mark.asyncio
async def test_request_prefix_values_runs_in_thread_and_keeps_loop_responsive(
    monkeypatch,
):
    """M-1: the per-request hashing/tokenization must run off the event loop
    thread, and the loop must stay responsive while it runs."""
    info = {}

    def blocking_values(*a, **k):
        info["thread"] = threading.current_thread().ident
        time.sleep(0.3)
        return ("p", "k", ["b"], ["h"], 10)

    monkeypatch.setattr(hs, "request_prefix_values", blocking_values)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(10):
            ticks += 1
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    result = await hs.request_prefix_values_async([{"role": "user", "content": "x"}], "model", 100)
    await hb

    assert result == ("p", "k", ["b"], ["h"], 10)
    assert info["thread"] != threading.main_thread().ident, (
        "hashing must run off the event loop thread"
    )
    assert ticks >= 3, f"event loop was blocked during hashing: {ticks} heartbeats"
