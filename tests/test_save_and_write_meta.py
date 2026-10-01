# tests/test_save_and_write_meta.py

"""_save_and_write_meta: subsumed metas are dropped ONLY after the new meta is
written. A failed meta write must not delete the still-valid shorter caches
(data-loss guard)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import chat_flow
import hashing as hs
from cache import bin_cache


def _sm():
    sm = MagicMock()
    sm.save_after = AsyncMock(return_value=True)
    return sm


def _write_prefix(key, prefix_hashes, model_id="m1"):
    # A meta's key is its full-conversation hash == the last prefix hash.
    assert key == prefix_hashes[-1]
    hs.write_meta(key, "p", [], 100, model_id, prefix_hashes=prefix_hashes)


def _patch(monkeypatch):
    monkeypatch.setattr(hs, "write_meta_async", AsyncMock())
    monkeypatch.setattr(chat_flow, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(chat_flow, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: None)


@pytest.mark.asyncio
async def test_drops_subsumed_metas_after_successful_meta(meta_dir, monkeypatch):
    """Successful save + meta: the strict-prefix meta is deleted, the new kept."""
    _write_prefix("h_ab", ["h_a", "h_ab"])
    _write_prefix("h_abc", ["h_a", "h_ab", "h_abc"])
    _patch(monkeypatch)

    ok = await chat_flow._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert ok is True
    assert not (meta_dir / "h_ab.meta.json").exists(), "subsumed meta must be deleted"
    assert (meta_dir / "h_abc.meta.json").exists(), "new meta must be kept"


@pytest.mark.asyncio
async def test_keeps_subsumed_metas_when_meta_write_fails(meta_dir, monkeypatch):
    """A failed meta write must NOT delete the still-valid shorter caches."""
    _write_prefix("h_ab", ["h_a", "h_ab"])

    async def _raise(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(hs, "write_meta_async", _raise)
    monkeypatch.setattr(chat_flow, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(chat_flow, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: None)

    ok = await chat_flow._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert ok is True, "the slot save itself succeeded"
    assert (meta_dir / "h_ab.meta.json").exists(), "subsumed meta must be kept"
    assert not (meta_dir / "h_abc.meta.json").exists(), "failed meta must not exist"


@pytest.mark.asyncio
async def test_passes_saved_hashes_and_deletes_request_hashes(meta_dir, monkeypatch):
    """The meta is written with response-extended hashes, while subsumed
    deletion still uses the incoming request prompt hashes."""
    write_mock = AsyncMock()
    delete_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(hs, "write_meta_async", write_mock)
    monkeypatch.setattr(hs, "delete_subsumed_metas_async", delete_mock)
    monkeypatch.setattr(chat_flow, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(chat_flow, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: None)

    ok = await chat_flow._save_and_write_meta(
        [],
        _sm(),
        ("g",),
        "key",
        "saved_prefix",
        ["sb"],
        ["req_h"],
        "m1",
        saved_prefix_hashes=["saved_h"],
    )

    assert ok is True
    write_mock.assert_awaited_once()
    args = write_mock.await_args.args
    assert args == (
        "key",
        "saved_prefix",
        ["sb"],
        hs.WORDS_PER_BLOCK,
        "m1",
        ["req_h"],
        None,
        ["saved_h"],
    )
    delete_mock.assert_awaited_once_with("key", ["req_h"], "m1")


@pytest.mark.asyncio
async def test_falls_back_to_prompt_hashes_when_saved_missing(meta_dir, monkeypatch):
    write_mock = AsyncMock()
    delete_mock = AsyncMock(return_value=[])
    monkeypatch.setattr(hs, "write_meta_async", write_mock)
    monkeypatch.setattr(hs, "delete_subsumed_metas_async", delete_mock)
    monkeypatch.setattr(chat_flow, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(chat_flow, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: None)

    ok = await chat_flow._save_and_write_meta(
        [], _sm(), ("g",), "key", "p", ["b"], ["req_h"], "m1"
    )

    assert ok is True
    args = write_mock.await_args.args
    assert args[7] == ["req_h"]
    delete_mock.assert_awaited_once_with("key", ["req_h"], "m1")


@pytest.mark.asyncio
async def test_discards_empty_capture_and_keeps_subsumed(meta_dir, monkeypatch):
    """A header-only .bin (empty slot capture, e.g. slot erased mid-generation)
    must not be recorded as a meta and must not delete the valid shorter
    caches."""
    _write_prefix("h_ab", ["h_a", "h_ab"])

    write_mock = AsyncMock()
    delete_mock = AsyncMock()
    del_bin_mock = MagicMock()
    monkeypatch.setattr(hs, "write_meta_async", write_mock)
    monkeypatch.setattr(hs, "delete_subsumed_metas_async", delete_mock)
    monkeypatch.setattr(chat_flow, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(chat_flow, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(chat_flow, "BIN_CACHE_DIR", "/tmp/bin")
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: 1200)
    monkeypatch.setattr(bin_cache, "delete_bin_file", del_bin_mock)

    ok = await chat_flow._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert ok is False, "an empty capture must not count as a save"
    write_mock.assert_not_awaited()
    delete_mock.assert_not_awaited()
    del_bin_mock.assert_called_once()
    assert (meta_dir / "h_ab.meta.json").exists(), "subsumed meta must be kept"


async def test_empty_capture_clears_the_saved_key(meta_dir, monkeypatch):
    """The .bin was just deleted, so the slot holds no cacheable conversation.

    save_after had already recorded the key and refreshed the LRU mark before
    the size check threw the capture away, leaving the slot marked as holding a
    cache that does not exist: its LRU position was refreshed for nothing, and
    a later no-op-restore decision would be taken on a key with no meta."""
    _write_prefix("h_ab", ["h_a", "h_ab"])

    # A real SlotManager: the state under test is its _last_saved record.
    from backend.slot_manager import SlotManager

    sm = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    sm.set_clients([client])
    g = (0, "m1", 0)
    monkeypatch.setattr(hs, "write_meta_async", AsyncMock())
    monkeypatch.setattr(hs, "delete_subsumed_metas_async", AsyncMock())
    monkeypatch.setattr(chat_flow, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(chat_flow, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(chat_flow, "BIN_CACHE_DIR", "/tmp/bin")
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: 1200)
    monkeypatch.setattr(bin_cache, "delete_bin_file", MagicMock())

    # The slot really did hold this conversation before the capture.
    assert await sm.save_after(g, "h_abc") is True
    assert sm._last_saved[g] == "h_abc"

    ok = await chat_flow._save_and_write_meta(
        [], sm, g, "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert ok is False
    assert g not in sm._last_saved, (
        "a discarded capture must not leave the slot marked as holding it"
    )
