# tests/test_slot_manager.py

"""P0-2: acquire_for_request must never leak the slot lock.

Scenarios:
- normal acquire (with/without restore key);
- restore_slot raising (backend down): lock must not leak, request proceeds
  without cache (restored=False);
- cancellation during restore (wait_for timeout in app.py): lock must be
  released.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import slot_manager as sm_module
from slot_manager import SlotManager


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.restore_slot = AsyncMock(return_value=True)
    client.save_slot = AsyncMock(return_value=True)
    manager.set_clients([client])
    return manager


@pytest.mark.asyncio
async def test_acquire_without_restore(sm):
    g, lock, restored = await sm.acquire_for_request(None)
    assert g in sm._all_slots
    assert lock.locked()
    assert restored is None
    sm.release(g)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_acquire_with_restore(sm):
    g, lock, restored = await sm.acquire_for_request("key123")
    assert restored is True
    sm.backends[0]["client"].restore_slot.assert_awaited_once_with(g[1], "key123")
    sm.release(g)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_restore_exception_does_not_leak_lock(sm):
    """restore_slot raising (backend down) must not leak the lock; the request
    proceeds without cache (restored=False)."""
    sm.backends[0]["client"].restore_slot = AsyncMock(
        side_effect=RuntimeError("backend down")
    )
    g, lock, restored = await sm.acquire_for_request("key123")
    assert restored is False
    assert lock.locked(), "slot stays with the request"
    sm.release(g)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_cancellation_during_restore_releases_lock(sm):
    """Task cancellation while restore is in flight must release the lock."""

    async def slow_restore(slot_id, basename):
        await asyncio.sleep(10)

    sm.backends[0]["client"].restore_slot = slow_restore

    task = asyncio.create_task(sm.acquire_for_request("key123"))
    await asyncio.sleep(0.05)  # let it acquire the lock and enter restore
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked after cancellation"


@pytest.mark.asyncio
async def test_wait_for_timeout_releases_lock(sm):
    """The production path: app.py wraps acquire in wait_for; a timeout during
    restore must release the lock."""

    async def slow_restore(slot_id, basename):
        await asyncio.sleep(10)

    sm.backends[0]["client"].restore_slot = slow_restore

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(sm.acquire_for_request("key123"), timeout=0.1)

    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked after wait_for timeout"


@pytest.mark.asyncio
async def test_acquire_marks_slot_used(sm):
    """P1-5: occupying a slot is the 'last used' moment for LRU — even for
    small requests that never save."""
    g, _, _ = await sm.acquire_for_request(None)
    try:
        assert sm._last_used[g] > 0, "acquire must mark the slot as used"
    finally:
        sm.release(g)


@pytest.mark.asyncio
async def test_failed_save_does_not_refresh_usage(sm, monkeypatch):
    """P1-5: a failed save must not refresh the LRU mark."""
    fake = {"now": 1000.0}
    monkeypatch.setattr(sm_module.time, "time", lambda: fake["now"])
    g, _, _ = await sm.acquire_for_request(None)
    ts_acquire = sm._last_used[g]
    try:
        sm.backends[0]["client"].save_slot = AsyncMock(return_value=False)
        ok = await sm.save_after(g, "k" * 16)
        assert ok is False
        assert sm._last_used[g] == ts_acquire, (
            "failed save must not refresh the LRU mark"
        )
    finally:
        sm.release(g)


@pytest.mark.asyncio
async def test_successful_save_refreshes_usage(sm, monkeypatch):
    """P1-5: a successful save refreshes the LRU mark."""
    fake = {"now": 1000.0}
    monkeypatch.setattr(sm_module.time, "time", lambda: fake["now"])
    g, _, _ = await sm.acquire_for_request(None)
    ts_acquire = sm._last_used[g]
    try:
        fake["now"] = 2000.0
        sm.backends[0]["client"].save_slot = AsyncMock(return_value=True)
        ok = await sm.save_after(g, "k" * 16)
        assert ok is True
        assert sm._last_used[g] == 2000.0, "successful save must refresh the mark"
        assert sm._last_used[g] > ts_acquire
    finally:
        sm.release(g)


def test_oldest_slot_selected_by_usage(sm):
    """P1-5: with no free slots, the least recently used one is selected."""
    sm._last_used[(0, 0)] = 1000.0
    sm._last_used[(0, 1)] = 1500.0

    g, _lock = sm._get_free_or_oldest()

    assert g == (0, 0), "the least recently used slot must be selected"
