# tests/test_cache_eviction.py

"""P3-1: cache eviction (TTL + size caps), /cache endpoints, hit/miss
counters, and best-effort backend .bin purge."""

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import hashing as hs
import slot_manager as sm_module
from llama_client import LlamaClient
from slot_manager import SlotManager


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    client.restore_slot = AsyncMock(return_value=True)
    client.get_model_id_cached = AsyncMock(return_value="m1")
    client.get_loaded_model = AsyncMock(return_value="m1")
    client.chat_completions = AsyncMock(return_value={"choices": []})
    client.delete_cache_file = AsyncMock(return_value=True)
    manager.set_clients([client])
    return manager


@pytest.fixture()
def counters(monkeypatch):
    monkeypatch.setattr(hs, "_hits", 0)
    monkeypatch.setattr(hs, "_misses", 0)


def _write_meta(dirpath, key: str, mtime: float | None = None) -> str:
    path = os.path.join(dirpath, f"{key}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"key": key, "model_id": "m1", "blocks": [], "timestamp": time.time()}, f
        )
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


async def test_ttl_eviction_deletes_old_meta(meta_dir, counters):
    """Meta files older than the TTL are deleted, fresh ones are kept."""
    now = time.time()
    old = _write_meta(meta_dir, "oldkey", mtime=now - 2 * 3600)
    fresh = _write_meta(meta_dir, "freshkey", mtime=now)

    res = hs.evict_meta(ttl_hours=1, max_files=100, max_mb=100)

    assert "oldkey" in res["deleted"]
    assert not os.path.exists(old)
    assert os.path.exists(fresh)
    assert res["remaining"] == 1


async def test_max_files_cap_keeps_newest(meta_dir, counters):
    """When the file count cap is exceeded the oldest files go first."""
    now = time.time()
    paths = [
        _write_meta(meta_dir, f"k{i}", mtime=now - (5 - i) * 100) for i in range(5)
    ]

    res = hs.evict_meta(ttl_hours=0, max_files=2, max_mb=100)

    assert len(res["deleted"]) == 3
    assert res["remaining"] == 2
    # The two newest survive.
    assert os.path.exists(paths[4])
    assert os.path.exists(paths[3])
    assert not os.path.exists(paths[0])


async def test_max_mb_cap_deletes_all_when_zero(meta_dir, counters):
    """A zero size cap deletes everything."""
    _write_meta(meta_dir, "a")
    _write_meta(meta_dir, "b")

    res = hs.evict_meta(ttl_hours=0, max_files=100, max_mb=0)

    assert res["remaining"] == 0
    assert sorted(res["deleted"]) == ["a", "b"]


async def test_clear_all_meta(meta_dir, counters):
    """clear_all_meta removes every meta file and returns the keys."""
    _write_meta(meta_dir, "a")
    _write_meta(meta_dir, "b")

    deleted = await hs.clear_all_meta_async()

    assert sorted(deleted) == ["a", "b"]
    assert hs._meta_files() == []


async def test_cache_stats(meta_dir, counters):
    """cache_stats reports file count, size, and hit/miss counters."""
    _write_meta(meta_dir, "a")
    hs.record_hit()
    hs.record_hit()
    hs.record_miss()

    stats = hs.cache_stats()

    assert stats["files"] == 1
    assert stats["total_bytes"] > 0
    assert stats["hits"] == 2
    assert stats["misses"] == 1


async def test_evict_meta_async(meta_dir, counters):
    """The async wrapper runs eviction and returns the same shape."""
    now = time.time()
    _write_meta(meta_dir, "oldkey", mtime=now - 2 * 3600)

    res = await hs.evict_meta_async(ttl_hours=1, max_files=100, max_mb=100)

    assert "oldkey" in res["deleted"]
    assert res["remaining"] == 0


async def test_delete_cache_file_success():
    """A 200/204 delete response counts as success."""
    client = LlamaClient("http://be")
    resp = MagicMock()
    resp.status_code = 200
    client.client.delete = AsyncMock(return_value=resp)

    assert await client.delete_cache_file("abc") is True
    await client.close()


async def test_delete_cache_file_not_found():
    """A 404 (no such file / no endpoint) is a soft failure."""
    client = LlamaClient("http://be")
    resp = MagicMock()
    resp.status_code = 404
    client.client.delete = AsyncMock(return_value=resp)

    assert await client.delete_cache_file("abc") is False
    await client.close()


async def test_delete_cache_file_backend_down():
    """A backend error is a soft failure, never an exception."""
    client = LlamaClient("http://be")
    client.client.delete = AsyncMock(side_effect=Exception("down"))

    assert await client.delete_cache_file("abc") is False
    await client.close()


async def test_cache_clear_endpoint(sm, meta_dir):
    """/cache/clear deletes meta files and purges the backend best-effort."""
    _write_meta(meta_dir, "a")
    _write_meta(meta_dir, "b")
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]

    resp = await app_module.cache_clear()

    assert resp["deleted"] == 2
    assert client.delete_cache_file.await_count == 2
    assert hs._meta_files() == []


async def test_cache_stats_endpoint(sm, meta_dir, counters):
    """/cache/stats returns the current cache state."""
    _write_meta(meta_dir, "a")
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]

    stats = await app_module.cache_stats()

    assert stats["files"] == 1
    assert stats["total_bytes"] > 0


async def test_health_probes_backends_concurrently(sm):
    """L3: health() must probe all backends concurrently (asyncio.gather). A
    sequential probe would block on the barrier (only one party present) and
    time out; a concurrent probe releases it."""
    import asyncio

    barrier = asyncio.Barrier(3)
    clients = []
    for i in range(3):
        c = MagicMock()

        async def fake_health(i=i, barrier=barrier):
            await barrier.wait()
            return {"ok": True, "i": i}

        c.health = fake_health
        clients.append(c)
    app_module.app.state.sm = sm
    app_module.app.state.clients = clients

    out = await asyncio.wait_for(app_module.health(), timeout=2.0)

    assert out["ok"] is True
    assert len(out["backends"]) == 3


async def test_purge_deletes_bin_file_directly(sm, tmp_path, monkeypatch):
    """When BIN_CACHE_DIR is set, the .bin file is removed from disk too."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bin_file = bin_dir / "abc"
    bin_file.write_bytes(b"x" * 100)
    monkeypatch.setattr(app_module, "BIN_CACHE_DIR", str(bin_dir))

    client = sm.backends[0]["client"]
    await app_module._purge_backend_files([client], [("abc", "m1")])

    assert not bin_file.exists()
    assert client.delete_cache_file.await_count == 1


async def test_cache_clear_removes_orphan_bin_files(sm, meta_dir, tmp_path, monkeypatch):
    """/cache/clear also removes orphaned .bin files from the mounted dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "orphan").write_bytes(b"x" * 100)
    monkeypatch.setattr(app_module, "BIN_CACHE_DIR", str(bin_dir))
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]

    await app_module.cache_clear()

    assert not (bin_dir / "orphan").exists()
