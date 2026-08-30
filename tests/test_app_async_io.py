# tests/test_app_async_io.py

"""M6: meta file I/O on the request/eviction path must not block the event
loop: _key_model_pairs and the /cache/stats endpoint run their scans in the
thread pool."""

import asyncio
import threading
import time

import pytest

import app as app_module
import hashing as hs


@pytest.mark.asyncio
async def test_key_model_pairs_runs_scan_in_thread(monkeypatch):
    """_key_model_pairs must run the meta scan off the event loop thread."""
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
    pairs = await app_module._key_model_pairs()
    await hb

    assert pairs == {}
    assert info["thread"] != threading.main_thread().ident, (
        "scan must run off the event loop thread"
    )
    assert ticks >= 3, f"event loop was blocked during scan: {ticks} heartbeats"


@pytest.mark.asyncio
async def test_cache_stats_endpoint_runs_in_thread(sm, monkeypatch):
    """The /cache/stats endpoint must run the stats scan off the loop thread."""
    info = {}

    def blocking_stats():
        info["thread"] = threading.current_thread().ident
        time.sleep(0.3)
        return {"files": 0, "total_bytes": 0, "hits": 0, "misses": 0}

    monkeypatch.setattr(hs, "cache_stats", blocking_stats)
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(10):
            ticks += 1
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    stats = await app_module.cache_stats()
    await hb

    assert stats["files"] == 0
    assert info["thread"] != threading.main_thread().ident, (
        "stats must run off the event loop thread"
    )
    assert ticks >= 3, f"event loop was blocked during stats: {ticks} heartbeats"
