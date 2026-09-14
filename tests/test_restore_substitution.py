# tests/test_restore_substitution.py

"""Restore substitution: when a save subsumes a key that a big request is
waiting to restore, the waiter must restore the longer replacement cache
instead of the deleted file (the file is gone by the time the slot is
acquired). Alias is recorded before the .bin is purged and resolved at restore
time via ``resolve_restore_key`` passed into ``slot_manager.acquire_for_request``.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import bin_cache
import chat_flow
import hashing as hs
import slot_manager as sm_module
from slot_manager import SlotManager

# ---- helpers ----------------------------------------------------------------

def _sm():
    sm = MagicMock()
    sm.save_after = AsyncMock(return_value=True)
    return sm


def _write_prefix(key, prefix_hashes, model_id="m1"):
    assert key == prefix_hashes[-1]
    hs.write_meta(key, "p", [], 100, model_id, prefix_hashes=prefix_hashes)


def _patch(monkeypatch):
    monkeypatch.setattr(hs, "write_meta_async", AsyncMock())
    monkeypatch.setattr(chat_flow, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(chat_flow, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: None)
    chat_flow._PENDING_RESTORES.clear()
    chat_flow._RESTORE_ALIAS.clear()


def _manager():
    sm_module.BACKENDS = [{"url": "http://be", "n_slots": 1}]
    manager = SlotManager()
    client = MagicMock()
    client.restore_slot = AsyncMock(return_value=True)
    client.save_slot = AsyncMock(return_value=True)
    manager.set_clients([client])
    manager.set_backend_slots(0, "model1", [{"id": 0}])
    return manager, client


# ---- register / unregister ---------------------------------------------------

def test_register_unregister_counts():
    chat_flow._PENDING_RESTORES.clear()
    chat_flow._RESTORE_ALIAS.clear()
    chat_flow._register_pending_restore("k")
    chat_flow._register_pending_restore("k")
    assert chat_flow._PENDING_RESTORES["k"] == 2
    chat_flow._unregister_pending_restore("k")
    assert chat_flow._PENDING_RESTORES["k"] == 1
    chat_flow._unregister_pending_restore("k")
    assert "k" not in chat_flow._PENDING_RESTORES


def test_unregister_drops_alias_when_count_zero():
    chat_flow._PENDING_RESTORES.clear()
    chat_flow._RESTORE_ALIAS.clear()
    chat_flow._RESTORE_ALIAS["k"] = "new"
    chat_flow._PENDING_RESTORES["k"] = 1
    chat_flow._unregister_pending_restore("k")
    assert chat_flow._RESTORE_ALIAS.get("k") is None


def test_unregister_keeps_alias_while_pending():
    chat_flow._PENDING_RESTORES.clear()
    chat_flow._RESTORE_ALIAS.clear()
    chat_flow._RESTORE_ALIAS["k"] = "new"
    chat_flow._PENDING_RESTORES["k"] = 2
    chat_flow._unregister_pending_restore("k")
    assert chat_flow._RESTORE_ALIAS["k"] == "new"


# ---- resolve ----------------------------------------------------------------

def test_resolve_chains_aliases():
    chat_flow._RESTORE_ALIAS.clear()
    chat_flow._RESTORE_ALIAS["a"] = "b"
    chat_flow._RESTORE_ALIAS["b"] = "c"
    assert chat_flow._resolve_restore_key("a") == "c"


def test_resolve_unknown_key_unchanged():
    chat_flow._RESTORE_ALIAS.clear()
    assert chat_flow._resolve_restore_key("other") == "other"


# ---- _save_and_write_meta alias on subsumed delete --------------------------

@pytest.mark.asyncio
async def test_aliases_pending_key_on_subsumed_delete(meta_dir, monkeypatch):
    _write_prefix("h_ab", ["h_a", "h_ab"])
    _patch(monkeypatch)
    chat_flow._register_pending_restore("h_ab")

    ok = await chat_flow._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert ok is True
    assert chat_flow._RESTORE_ALIAS["h_ab"] == "h_abc"


@pytest.mark.asyncio
async def test_no_alias_without_pending(meta_dir, monkeypatch):
    _write_prefix("h_ab", ["h_a", "h_ab"])
    _patch(monkeypatch)

    await chat_flow._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert chat_flow._RESTORE_ALIAS.get("h_ab") is None


@pytest.mark.asyncio
async def test_bin_still_purged_even_when_alias_recorded(meta_dir, monkeypatch):
    _write_prefix("h_ab", ["h_a", "h_ab"])
    _patch(monkeypatch)
    purge_mock = AsyncMock()
    monkeypatch.setattr(chat_flow, "_purge_backend_files", purge_mock)
    chat_flow._register_pending_restore("h_ab")

    await chat_flow._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    purge_mock.assert_awaited_once()
    deleted_keys = [k for k, _model in purge_mock.await_args.args[1]]
    assert deleted_keys == ["h_ab"]


# ---- slot_manager resolve parameter ----------------------------------------

@pytest.mark.asyncio
async def test_acquire_resolves_restore_key():
    manager, client = _manager()

    _g, _lock, restored = await manager.acquire_for_request(
        "model1",
        "old_key",
        resolve_restore_key=lambda k: "replaced" if k == "old_key" else k,
    )

    assert restored is True
    client.restore_slot.assert_awaited_once_with(0, "replaced", model="model1")


@pytest.mark.asyncio
async def test_acquire_no_resolve_when_none():
    manager, client = _manager()

    _g, _lock, restored = await manager.acquire_for_request("model1", "key123")

    assert restored is True
    client.restore_slot.assert_awaited_once_with(0, "key123", model="model1")