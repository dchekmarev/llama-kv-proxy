# tests/test_slot_manager.py

"""SlotManager: acquire must never leak the slot lock, and slots are scoped by
model (a router serves each model on its own child with its own slots).

Scenarios:
- normal acquire (with/without restore key);
- restore_slot raising (backend down): lock must not leak, request proceeds
  without cache (restored=False);
- cancellation during restore (wait_for timeout in app.py): lock must be
  released;
- per-model isolation: two models never share a slot;
- bootstrap fallback: an undiscovered model pins slot 0 on the first backend.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import slot_manager as sm_module
from llama_client import RESTORE_ERROR
from slot_manager import SlotManager


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.restore_slot = AsyncMock(return_value=True)
    client.save_slot = AsyncMock(return_value=True)
    manager.set_clients([client])
    # Populate the pool for the test model (discovery-driven).
    manager.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}])
    return manager


@pytest.mark.asyncio
async def test_acquire_without_restore(sm):
    g, lock, restored, used_key = await sm.acquire_for_request("model1")
    assert g[0] == 0 and g[1] == "model1"
    assert lock.locked()
    assert restored is None
    assert used_key is None, "no restore attempted -> no used key"
    sm.release(g)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_acquire_with_restore(sm):
    g, lock, restored, used_key = await sm.acquire_for_request("model1", "key123")
    assert restored is True
    assert used_key == "key123"
    sm.backends[0]["client"].restore_slot.assert_awaited_once_with(
        g[2], "key123", model=g[1]
    )
    sm.release(g)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_restore_exception_does_not_leak_lock(sm):
    """restore_slot raising (backend down) must not leak the lock; the request
    proceeds without cache. The restored sentinel is RESTORE_ERROR (not a
    plain False) so the caller keeps the meta for a future retry."""
    sm.backends[0]["client"].restore_slot = AsyncMock(
        side_effect=RuntimeError("backend down")
    )
    g, lock, restored, used_key = await sm.acquire_for_request("model1", "key123")
    assert restored is RESTORE_ERROR
    assert used_key == "key123", "the key was attempted even though restore failed"
    assert lock.locked(), "slot stays with the request"
    sm.release(g)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_cancellation_during_restore_releases_lock(sm):
    """Task cancellation while restore is in flight must release the lock."""

    async def slow_restore(slot_id, basename, model=None):
        await asyncio.sleep(10)

    sm.backends[0]["client"].restore_slot = slow_restore

    task = asyncio.create_task(sm.acquire_for_request("model1", "key123"))
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

    async def slow_restore(slot_id, basename, model=None):
        await asyncio.sleep(10)

    sm.backends[0]["client"].restore_slot = slow_restore

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(sm.acquire_for_request("model1", "key123"), timeout=0.1)

    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked after wait_for timeout"


@pytest.mark.asyncio
async def test_acquire_marks_slot_used(sm):
    """Occupying a slot is the 'last used' moment for LRU — even for small
    requests that never save."""
    g, _, _, _ = await sm.acquire_for_request("model1")
    try:
        assert sm._last_used[g] > 0, "acquire must mark the slot as used"
    finally:
        sm.release(g)


@pytest.mark.asyncio
async def test_failed_save_does_not_refresh_usage(sm, monkeypatch):
    """A failed save must not refresh the LRU mark."""
    fake = {"now": 1000.0}
    monkeypatch.setattr(sm_module.time, "time", lambda: fake["now"])
    g, _, _, _ = await sm.acquire_for_request("model1")
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
    """A successful save refreshes the LRU mark and passes the model."""
    fake = {"now": 1000.0}
    monkeypatch.setattr(sm_module.time, "time", lambda: fake["now"])
    g, _, _, _ = await sm.acquire_for_request("model1")
    ts_acquire = sm._last_used[g]
    try:
        fake["now"] = 2000.0
        sm.backends[0]["client"].save_slot = AsyncMock(return_value=True)
        ok = await sm.save_after(g, "k" * 16)
        assert ok is True
        assert sm._last_used[g] == 2000.0, "successful save must refresh the mark"
        assert sm._last_used[g] > ts_acquire
        sm.backends[0]["client"].save_slot.assert_awaited_once_with(
            g[2], "k" * 16, model=g[1]
        )
    finally:
        sm.release(g)


@pytest.mark.asyncio
async def test_waiters_repick_on_release_no_pileup(sm):
    """2 slots, 2 holders, 4 waiters: releasing both slots must let two
    waiters re-pick and proceed. The old code queued every waiter on the
    oldest slot's lock, so only one waiter woke and the other slot idled."""
    g1, _, _, _ = await sm.acquire_for_request("model1")
    g2, _, _, _ = await sm.acquire_for_request("model1")
    assert g1 != g2

    waiters = [asyncio.create_task(sm.acquire_for_request("model1")) for _ in range(4)]
    for _ in range(10):
        await asyncio.sleep(0.01)  # let all waiters park
    assert len(sm._waiters.get("model1", ())) == 4, "all 4 must be waiting"

    sm.release(g1)
    sm.release(g2)
    done, pending = await asyncio.wait(waiters, timeout=2.0)
    assert len(done) == 2, f"both freed slots must be taken, done={len(done)}"

    for t in done:
        sm.release(t.result()[0])
    done2, pending2 = await asyncio.wait(list(pending), timeout=2.0)
    assert not pending2, "no waiter may stay asleep while a slot is free"
    for t in done2:
        sm.release(t.result()[0])

    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"
    assert not sm._waiters.get("model1"), "no waiter may remain registered"


def test_oldest_slot_selected_by_usage(sm):
    """With no free slots, the least recently used one is selected."""
    sm._last_used[(0, "model1", 0)] = 1000.0
    sm._last_used[(0, "model1", 1)] = 1500.0

    g, _lock = sm._get_free_or_oldest("model1")

    assert g == (0, "model1", 0), "the least recently used slot must be selected"


def test_models_do_not_share_slots(sm):
    """Two models are routed to their own slots, never the same one."""
    sm.set_backend_slots(0, "modelA", [{"id": 0}, {"id": 1}])
    sm.set_backend_slots(0, "modelB", [{"id": 0}, {"id": 1}])

    ga, _ = sm._get_free_or_oldest("modelA")
    gb, _ = sm._get_free_or_oldest("modelB")

    assert ga == (0, "modelA", 0)
    assert gb == (0, "modelB", 0)
    assert ga != gb, "different models must not share a slot identity"


@pytest.mark.asyncio
async def test_empty_pool_falls_back_to_slot_zero(sm):
    """An undiscovered model (not loaded yet) pins slot 0 on the first backend."""
    g, lock, restored, used_key = await sm.acquire_for_request("not-loaded-yet")
    assert g == (0, "not-loaded-yet", 0)
    assert restored is None
    assert used_key is None
    sm.release(g)
    assert not lock.locked()


def test_aggregated_state_includes_model(sm):
    """aggregated_state carries the model and the LRU mark."""
    sm._last_used[(0, "model1", 0)] = 123.0
    state = sm.aggregated_state()
    first = next(s for s in state if s["slot"] == 0)
    assert first["backend"] == 0
    assert first["model"] == "model1"
    assert first["last_used"] == 123.0
