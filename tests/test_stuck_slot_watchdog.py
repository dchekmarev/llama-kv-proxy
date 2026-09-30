# tests/test_stuck_slot_watchdog.py

"""Watchdog: a backend slot that reports is_processing for longer than
STUCK_SLOT_THRESHOLD_S is presumed wedged (stuck in PROCESSING_PROMPT) and is
erased to recover the backend without a restart. The /slots payload has no
per-slot timestamp, so the proxy tracks how long it has seen each slot busy.
A threshold of 0 disables the watchdog."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module


class _Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.fixture(autouse=True)
def _clean_tracker():
    app_module._stuck_slot_first_busy.clear()
    yield
    app_module._stuck_slot_first_busy.clear()


@pytest.fixture()
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(app_module.time, "time", c)
    return c


@pytest.fixture()
def client():
    c = MagicMock()
    c.erase_slot = AsyncMock(return_value=True)
    return c


def _slot(sid: int, busy: bool) -> dict:
    return {"id": sid, "is_processing": busy}


@pytest.mark.asyncio
async def test_first_observation_only_records(clock, client):
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    client.erase_slot.assert_not_awaited()
    assert (0, "m1", 0) in app_module._stuck_slot_first_busy


@pytest.mark.asyncio
async def test_within_threshold_no_erase(clock, client, monkeypatch):
    monkeypatch.setattr(app_module, "STUCK_SLOT_THRESHOLD_S", 300)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    clock.advance(100)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    client.erase_slot.assert_not_awaited()


@pytest.mark.asyncio
async def test_beyond_threshold_erases(clock, client, monkeypatch):
    monkeypatch.setattr(app_module, "STUCK_SLOT_THRESHOLD_S", 300)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    clock.advance(301)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    client.erase_slot.assert_awaited_once_with(0, model="m1")


@pytest.mark.asyncio
async def test_erase_rearms_tracker(clock, client, monkeypatch):
    # After an erase the timer resets, so the very next poll does not re-erase.
    monkeypatch.setattr(app_module, "STUCK_SLOT_THRESHOLD_S", 300)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    clock.advance(301)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    assert client.erase_slot.await_count == 1
    clock.advance(1)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    assert client.erase_slot.await_count == 1


@pytest.mark.asyncio
async def test_idle_slot_clears_tracker(clock, client, monkeypatch):
    monkeypatch.setattr(app_module, "STUCK_SLOT_THRESHOLD_S", 300)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    assert (0, "m1", 0) in app_module._stuck_slot_first_busy
    clock.advance(1000)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, False)])
    client.erase_slot.assert_not_awaited()
    assert (0, "m1", 0) not in app_module._stuck_slot_first_busy


@pytest.mark.asyncio
async def test_zero_threshold_disables(clock, client, monkeypatch):
    monkeypatch.setattr(app_module, "STUCK_SLOT_THRESHOLD_S", 0)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    clock.advance(10_000)
    await app_module._check_stuck_slots(0, "m1", client, [_slot(0, True)])
    client.erase_slot.assert_not_awaited()
    assert not app_module._stuck_slot_first_busy
