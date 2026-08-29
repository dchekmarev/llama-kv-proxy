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
