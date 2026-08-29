# tests/test_stale_meta.py

"""P1-2: a meta file pointing at a cache that cannot be restored is stale and
must be dropped, otherwise every big request retries a doomed restore."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import hashing as hs
import slot_manager as sm_module
from slot_manager import SlotManager


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


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
    manager.set_clients([client])
    return manager


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "META_DIR", str(tmp_path))
    return tmp_path


def _big_content():
    return " ".join(["w"] * 600)


def _write_meta_for(content, meta_dir):
    prefix = hs.raw_prefix([{"role": "user", "content": content}])
    key = hs.prefix_key_sha256("m1\n" + prefix)
    blocks = hs.block_hashes_from_text(prefix, hs.WORDS_PER_BLOCK)
    hs.write_meta(key, prefix, blocks, hs.WORDS_PER_BLOCK, "m1")
    return key


async def _chat(sm, content):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    return await app_module.chat(FakeRequest(data))


@pytest.mark.asyncio
async def test_delete_meta_removes_file(meta_dir):
    key = "a" * 64
    hs.write_meta(key, "p", ["b"], hs.WORDS_PER_BLOCK, "m1")
    assert (meta_dir / f"{key}.meta.json").exists()

    assert hs.delete_meta(key) is True
    assert not (meta_dir / f"{key}.meta.json").exists()


@pytest.mark.asyncio
async def test_delete_meta_missing_returns_false(meta_dir):
    assert hs.delete_meta("b" * 64) is False


@pytest.mark.asyncio
async def test_delete_meta_async(meta_dir):
    key = "c" * 64
    hs.write_meta(key, "p", ["b"], hs.WORDS_PER_BLOCK, "m1")

    assert await hs.delete_meta_async(key) is True
    assert not (meta_dir / f"{key}.meta.json").exists()


@pytest.mark.asyncio
async def test_stale_meta_dropped_when_restore_fails(sm, meta_dir, monkeypatch):
    """restore -> False: the meta must be deleted (spied via delete_meta_async)."""
    content = _big_content()
    key = _write_meta_for(content, meta_dir)
    sm.backends[0]["client"].restore_slot = AsyncMock(return_value=False)

    delete_meta_async = AsyncMock()
    monkeypatch.setattr(hs, "delete_meta_async", delete_meta_async)

    await _chat(sm, content)

    delete_meta_async.assert_awaited_once_with(key)


@pytest.mark.asyncio
async def test_meta_kept_when_restore_succeeds(sm, meta_dir, monkeypatch):
    """restore -> True: no meta deletion."""
    content = _big_content()
    _write_meta_for(content, meta_dir)
    sm.backends[0]["client"].restore_slot = AsyncMock(return_value=True)

    delete_meta_async = AsyncMock()
    monkeypatch.setattr(hs, "delete_meta_async", delete_meta_async)

    await _chat(sm, content)

    delete_meta_async.assert_not_awaited()
