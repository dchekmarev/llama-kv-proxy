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
from promstats import counter_sum, restore_skipped_same_slot_total
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


@pytest.mark.asyncio
async def test_waiter_wakeup_is_fifo(sm):
    """One slot, several waiters: each release must serve the
    longest-waiting request first (FIFO), not an arbitrary waiter. A set
    here gave arbitrary order and starved the oldest request behind newer
    ones until ACQUIRE_TIMEOUT."""
    sm.set_backend_slots(0, "model1", [{"id": 0}])  # single slot
    holder, _, _, _ = await sm.acquire_for_request("model1")

    w1 = asyncio.create_task(sm.acquire_for_request("model1"))
    w2 = asyncio.create_task(sm.acquire_for_request("model1"))
    w3 = asyncio.create_task(sm.acquire_for_request("model1"))
    for _ in range(10):
        await asyncio.sleep(0.01)  # let all waiters park, in creation order
    assert len(sm._waiters["model1"]) == 3, "all 3 must be waiting"

    sm.release(holder)
    done, _ = await asyncio.wait(
        [w1, w2, w3], timeout=2.0, return_when=asyncio.FIRST_COMPLETED
    )
    assert done == {w1}, "the oldest waiter must be served first"

    sm.release(w1.result()[0])
    done, _ = await asyncio.wait(
        [w2, w3], timeout=2.0, return_when=asyncio.FIRST_COMPLETED
    )
    assert done == {w2}, "then the next-oldest, not the newest"

    sm.release(w2.result()[0])
    done, _ = await asyncio.wait([w3], timeout=2.0)
    assert done == {w3}
    sm.release(w3.result()[0])
    assert not sm._waiters.get("model1"), "no waiter may remain registered"


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_block_fifo(sm):
    """A waiter cancelled while sleeping (ACQUIRE_TIMEOUT) must unregister
    itself; the next release wakes the next live waiter, not the corpse."""
    sm.set_backend_slots(0, "model1", [{"id": 0}])  # single slot
    holder, _, _, _ = await sm.acquire_for_request("model1")

    w1 = asyncio.create_task(sm.acquire_for_request("model1"))
    w2 = asyncio.create_task(sm.acquire_for_request("model1"))
    for _ in range(10):
        await asyncio.sleep(0.01)
    assert len(sm._waiters["model1"]) == 2

    w1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await w1
    assert len(sm._waiters["model1"]) == 1, "cancelled waiter must unregister"

    sm.release(holder)
    done, _ = await asyncio.wait([w2], timeout=2.0)
    assert done == {w2}, "the live waiter must be woken past the cancelled one"
    sm.release(w2.result()[0])


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


# --- On-demand freshen (slot cut observed within ~1s, not at the next poll) ---


@pytest.mark.asyncio
async def test_freshen_replaces_pool_on_cut(sm):
    """A freshen reporting fewer slots must REPLACE the pool and drop the stale
    slot's bookkeeping — the stale id is what wraps onto a live physical slot."""
    sm.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}, {"id": 2}])
    sm._last_used[(0, "model1", 2)] = 123.0
    client = sm.backends[0]["client"]
    client.is_router = AsyncMock(return_value=False)
    client.get_slots = AsyncMock(return_value=[{"id": 0}, {"id": 1}])

    await sm.freshen_model("model1")

    assert sm._pools[(0, "model1")] == [0, 1], "pool must be replaced, not merged"
    assert (0, "model1", 2) not in sm._last_used, "stale slot LRU mark must be dropped"
    assert (0, "model1", 2) not in sm._locks, "stale free slot lock must be dropped"
    client.get_slots.assert_awaited_once()


@pytest.mark.asyncio
async def test_freshen_throttled_per_pool(sm, monkeypatch):
    """A burst of freshens within the interval triggers at most one re-poll;
    after the interval elapses the next freshen polls again."""
    fake = {"now": 1000.0}
    monkeypatch.setattr(sm_module.time, "time", lambda: fake["now"])
    client = sm.backends[0]["client"]
    client.is_router = AsyncMock(return_value=False)
    client.get_slots = AsyncMock(return_value=[{"id": 0}, {"id": 1}])

    await sm.freshen_model("model1")
    await sm.freshen_model("model1")
    await sm.freshen_model("model1")
    assert client.get_slots.await_count == 1, "throttle must hold within the interval"

    fake["now"] = 1000.0 + sm_module.SLOT_FRESHEN_INTERVAL_S + 0.1
    await sm.freshen_model("model1")
    assert client.get_slots.await_count == 2, "must re-poll once the interval passes"


@pytest.mark.asyncio
async def test_freshen_non_fatal_on_error(sm):
    """A failing re-poll must keep the existing pool and not raise."""
    sm.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}])
    client = sm.backends[0]["client"]
    client.is_router = AsyncMock(return_value=False)
    client.get_slots = AsyncMock(side_effect=RuntimeError("backend down"))

    await sm.freshen_model("model1")  # must not raise

    assert sm._pools[(0, "model1")] == [0, 1], "existing pool must be kept on error"


@pytest.mark.asyncio
async def test_freshen_non_fatal_on_malformed_body(sm):
    """A malformed /slots body (non-dict entries) must not raise and must keep
    the existing pool — set_backend_slots' s.get("id") is covered by the guard."""
    sm.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}])
    client = sm.backends[0]["client"]
    client.is_router = AsyncMock(return_value=False)
    client.get_slots = AsyncMock(return_value=[0, 1, 2])  # ints, not dicts

    await sm.freshen_model("model1")  # must not raise

    assert sm._pools[(0, "model1")] == [0, 1], "existing pool must be kept on malformed body"


@pytest.mark.asyncio
async def test_freshen_only_polls_serving_model(sm):
    """Only the pool for the requested model is re-polled, not sibling models."""
    sm.set_backend_slots(0, "model1", [{"id": 0}])
    sm.set_backend_slots(0, "other", [{"id": 0}])
    client = sm.backends[0]["client"]
    client.is_router = AsyncMock(return_value=False)
    client.get_slots = AsyncMock(return_value=[{"id": 0}])

    await sm.freshen_model("model1")

    assert client.get_slots.await_count == 1, "sibling model pool must not be polled"
    client.get_slots.assert_awaited_with(), "plain backend: no model arg"


@pytest.mark.asyncio
async def test_freshen_router_passes_model(sm):
    """A router backend is re-polled with the target model."""
    client = sm.backends[0]["client"]
    client.is_router = AsyncMock(return_value=True)
    client.get_slots = AsyncMock(return_value=[{"id": 0}, {"id": 1}])

    await sm.freshen_model("model1")

    client.get_slots.assert_awaited_once_with(model="model1")


# --- Hygiene: bookkeeping for slots that leave the pool ---


def test_set_backend_slots_drops_removed_free_slot(sm):
    """Removing a free slot from the pool drops its lock and LRU mark."""
    sm.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}, {"id": 2}])
    sm._last_used[(0, "model1", 2)] = 55.0
    sm._lock_for((0, "model1", 2))  # ensure a (free) lock exists

    sm.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}])

    assert (0, "model1", 2) not in sm._locks
    assert (0, "model1", 2) not in sm._last_used


@pytest.mark.asyncio
async def test_release_drops_bookkeeping_for_cut_slot(sm):
    """A slot cut from the pool while held is cleaned up on release (the held
    lock could not be dropped by set_backend_slots)."""
    sm.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}, {"id": 2}])
    g, lock, _, _ = await sm.acquire_for_request("model1")
    assert lock.locked()

    remaining = [{"id": s} for s in (0, 1, 2) if s != g[2]]
    sm.set_backend_slots(0, "model1", remaining)  # cut the held slot

    sm.release(g)

    assert g not in sm._locks, "cut slot lock must be dropped on release"
    assert g not in sm._last_used, "cut slot LRU mark must be dropped on release"


# --- Skip a restore into the slot that already holds the target key ---
#
# A save does not clear the slot: after saving key K from slot S, S still
# holds K's KV. When the next request re-picks S with restore candidate K,
# the restore would only re-read the .bin and rebuild identical KV — a
# no-op that must be skipped. The per-slot _last_saved record tracks which
# key's KV a slot currently holds.


@pytest.mark.asyncio
async def test_save_after_records_held_key(sm):
    g, _, _, _ = await sm.acquire_for_request("model1")
    try:
        await sm.save_after(g, "k1" * 8)
        assert sm._last_saved[g] == "k1" * 8
    finally:
        sm.release(g)


@pytest.mark.asyncio
async def test_failed_save_does_not_record_held_key(sm):
    g, _, _, _ = await sm.acquire_for_request("model1")
    try:
        sm.backends[0]["client"].save_slot = AsyncMock(return_value=False)
        ok = await sm.save_after(g, "k1" * 8)
        assert ok is False
        assert g not in sm._last_saved
    finally:
        sm.release(g)


@pytest.mark.asyncio
async def test_restore_skipped_when_slot_already_holds_key(sm):
    """The continuation after a save re-picks the same slot: the restore of
    the just-saved key is skipped (no backend call) and counts as a
    successful restore, so the pre-chat erase is not triggered."""
    sm.set_backend_slots(0, "model1", [{"id": 0}])  # single slot: LRU re-picks it
    g, _, _, _ = await sm.acquire_for_request("model1")
    await sm.save_after(g, "k1" * 8)
    sm.release(g)

    g2, lock, restored, used_key = await sm.acquire_for_request("model1", "k1" * 8)
    try:
        assert g2 == g
        assert restored is True, "a skipped restore counts as a successful one"
        assert used_key == "k1" * 8
        sm.backends[0]["client"].restore_slot.assert_not_awaited()
        assert g2 not in sm._last_saved, "the record is consumed by the skip"
        assert counter_sum(restore_skipped_same_slot_total, model="model1") == 1.0
    finally:
        sm.release(g2)
        assert not lock.locked()


@pytest.mark.asyncio
async def test_restore_not_skipped_for_different_key(sm):
    sm.set_backend_slots(0, "model1", [{"id": 0}])
    g, _, _, _ = await sm.acquire_for_request("model1")
    await sm.save_after(g, "k1" * 8)
    sm.release(g)

    g2, _, restored, _ = await sm.acquire_for_request("model1", "k2" * 8)
    try:
        assert restored is True
        sm.backends[0]["client"].restore_slot.assert_awaited_once_with(
            g2[2], "k2" * 8, model=g2[1]
        )
        assert sm._last_saved[g2] == "k2" * 8, (
            "a successful restore records the key the slot now holds"
        )
    finally:
        sm.release(g2)


@pytest.mark.asyncio
async def test_restore_not_skipped_on_other_slot(sm):
    """With 2 slots the picker takes the free one, not the just-saved slot:
    the restore there is real work."""
    g, _, _, _ = await sm.acquire_for_request("model1")
    await sm.save_after(g, "k1" * 8)
    sm.release(g)

    g2, _, restored, _ = await sm.acquire_for_request("model1", "k1" * 8)
    try:
        assert g2 != g, "the free slot must be picked over the just-saved one"
        assert restored is True
        sm.backends[0]["client"].restore_slot.assert_awaited_once()
    finally:
        sm.release(g2)


@pytest.mark.asyncio
async def test_second_restore_of_same_key_is_skipped(sm):
    """restore K -> chat -> a duplicate request with the same candidate K:
    the slot still holds K's KV (appended tokens are truncated by the
    backend's prefix match), so the second restore is skipped."""
    sm.set_backend_slots(0, "model1", [{"id": 0}])
    g, _, restored1, _ = await sm.acquire_for_request("model1", "k1" * 8)
    assert restored1 is True
    sm.release(g)

    g2, _, restored2, _ = await sm.acquire_for_request("model1", "k1" * 8)
    try:
        assert restored2 is True
        assert sm.backends[0]["client"].restore_slot.await_count == 1
    finally:
        sm.release(g2)


@pytest.mark.asyncio
async def test_failed_restore_keeps_record(sm):
    """A failed restore of another key leaves the slot's KV untouched, so the
    old record stays valid for a later matching request."""
    sm.set_backend_slots(0, "model1", [{"id": 0}])
    g, _, _, _ = await sm.acquire_for_request("model1")
    await sm.save_after(g, "k1" * 8)
    sm.release(g)

    sm.backends[0]["client"].restore_slot = AsyncMock(return_value=False)
    g2, _, restored, _ = await sm.acquire_for_request("model1", "k2" * 8)
    try:
        assert restored is False
        assert sm._last_saved.get(g2) == "k1" * 8
    finally:
        sm.release(g2)


@pytest.mark.asyncio
async def test_acquire_without_restore_clears_record(sm):
    sm.set_backend_slots(0, "model1", [{"id": 0}])
    g, _, _, _ = await sm.acquire_for_request("model1")
    await sm.save_after(g, "k1" * 8)
    sm.release(g)
    assert g in sm._last_saved

    g2, _, _, _ = await sm.acquire_for_request("model1")
    try:
        assert g2 == g
        assert g not in sm._last_saved, (
            "a chat without restore changes the slot's KV and invalidates the record"
        )
    finally:
        sm.release(g2)


@pytest.mark.asyncio
async def test_skip_disabled_by_flag(sm, monkeypatch):
    monkeypatch.setattr(sm_module, "SKIP_RESTORE_SAME_SLOT", False)
    sm.set_backend_slots(0, "model1", [{"id": 0}])
    g, _, _, _ = await sm.acquire_for_request("model1")
    await sm.save_after(g, "k1" * 8)
    sm.release(g)

    g2, _, restored, _ = await sm.acquire_for_request("model1", "k1" * 8)
    try:
        assert restored is True
        sm.backends[0]["client"].restore_slot.assert_awaited_once()
    finally:
        sm.release(g2)


def test_dropped_slot_clears_record(sm):
    sm.set_backend_slots(0, "model1", [{"id": 0}, {"id": 1}])
    sm._last_saved[(0, "model1", 1)] = "k" * 16

    sm.set_backend_slots(0, "model1", [{"id": 0}])

    assert (0, "model1", 1) not in sm._last_saved
