# tests/test_backend_health.py

"""Health-aware backend selection with several backends.

The slot poll already records each backend's liveness (llama_kv_proxy_backend_up).
Model-id/alias resolution, the pass-through client pick and slot candidates
consult that verdict: a backend currently known down is skipped while a healthy
one can answer, and only when every backend is down does the proxy fall back to
the first backend -- exactly the pre-health behavior, with the HOLD governing
the request.
"""

import asyncio
import contextlib
import time
from itertools import pairwise
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import chat_flow
from backend import slot_manager as sm_module
from backend.health import BackendHealth
from backend.slot_manager import SlotManager


def _client(model_id: str = "m1", alias: str | None = None) -> MagicMock:
    client = MagicMock()
    client.probe = AsyncMock(return_value=True)
    client.get_model_id_cached = AsyncMock(return_value=model_id)
    client.resolve_model_id_cached = AsyncMock(return_value=alias)
    client.get_loaded_model = AsyncMock(return_value=model_id)
    client.chat_completions = AsyncMock(return_value={"choices": []})
    client.save_slot = AsyncMock(return_value=True)
    client.restore_slot = AsyncMock(return_value=True)
    client.is_router = AsyncMock(return_value=False)
    client.get_slots = AsyncMock(return_value=None)
    client.get_model_ctx = AsyncMock(return_value=None)
    return client


@pytest.fixture()
def backends(monkeypatch):
    """A SlotManager with two backends and their mocked clients."""
    monkeypatch.setattr(
        sm_module,
        "BACKENDS",
        [{"url": "http://be0", "n_slots": 2}, {"url": "http://be1", "n_slots": 2}],
    )
    manager = SlotManager()
    clients = [_client(), _client()]
    manager.set_clients(clients)
    return manager, clients


@pytest.mark.asyncio
async def test_model_resolution_uses_live_backend(backends):
    """A model-less request must not read the down backend's stale id."""
    sm, clients = backends
    clients[0].get_model_id_cached = AsyncMock(return_value="unknown")
    clients[1].get_model_id_cached = AsyncMock(return_value="m2")
    sm.set_backend_up(0, False)

    model, no_cache = await chat_flow._resolve_effective_model(sm, clients, None, False)

    assert model == "m2"
    assert no_cache is False
    clients[0].get_model_id_cached.assert_not_awaited()


@pytest.mark.asyncio
async def test_alias_resolution_uses_live_backend(backends):
    """The alias is resolved by the live backend while the other is down."""
    sm, clients = backends
    clients[0].resolve_model_id_cached = AsyncMock(return_value=None)
    clients[1].resolve_model_id_cached = AsyncMock(return_value="m2")
    sm.set_backend_up(0, False)

    model, no_cache = await chat_flow._resolve_effective_model(
        sm, clients, "default", False
    )

    assert model == "m2"
    assert no_cache is False
    clients[0].resolve_model_id_cached.assert_not_awaited()


def test_health_order_prefers_live_then_falls_back(backends):
    """The pass-through/client pick prefers a live backend, first when all down."""
    sm, clients = backends

    sm.set_backend_up(0, False)
    assert chat_flow._health_order(sm, clients)[0] is clients[1]

    sm.set_backend_up(1, False)
    assert chat_flow._health_order(sm, clients) == [clients[0]]


@pytest.mark.asyncio
async def test_slot_selection_skips_down_backend(backends):
    """With both backends serving the model, the live one's slot is picked."""
    sm, _clients = backends
    sm.set_backend_slots(0, "m1", [{"id": 0}])
    sm.set_backend_slots(1, "m1", [{"id": 0}])
    sm.set_backend_up(0, False)

    g, _lock, _restored, _key = await sm.acquire_for_request("m1")
    sm.release(g)

    assert g[0] == 1, "a down backend's slot must not be picked while one is up"


def test_all_down_keeps_every_candidate(backends):
    """Every backend serving the model down: candidates unchanged, so the HOLD
    (not a rejection) governs the request exactly as before."""
    sm, _clients = backends
    sm.set_backend_slots(0, "m1", [{"id": 0}])
    sm.set_backend_slots(1, "m1", [{"id": 0}])
    sm.set_backend_up(0, False)
    sm.set_backend_up(1, False)

    assert sm._slots_for_model("m1") == [(0, "m1", 0), (1, "m1", 0)]


@pytest.mark.asyncio
async def test_poll_records_backend_liveness(backends):
    """The slot poll feeds the same verdict into the SlotManager."""
    sm, clients = backends
    clients[0].is_router = AsyncMock(side_effect=RuntimeError("down"))
    clients[1].get_model_id = AsyncMock(return_value="m1")
    clients[1].get_slots = AsyncMock(return_value=[{"id": 0, "state": "free"}])
    app_module.app.state.sm = sm
    app_module.app.state.clients = clients

    await app_module._poll_slots()

    assert sm.backend_up(0) is False
    assert sm.backend_up(1) is True


# --- shared health: single-flight prober, recovery wakeup, backoff ----------


async def _drain(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_concurrent_waiters_share_one_prober():
    """N concurrent waiters on a down backend run ONE prober, not N loops."""
    health = BackendHealth()
    health.set_up(0, False)
    probe = AsyncMock(return_value=False)
    tasks = [
        asyncio.create_task(
            health.wait_up(0, timeout=0.4, probe=probe, interval=0.05)
        )
        for _ in range(5)
    ]
    await asyncio.sleep(0.16)

    assert health.waiters(0) == 5
    assert health.prober_running(0)
    assert probe.await_count < len(tasks)

    for task in tasks:
        await _drain(task)
    assert health.waiters(0) == 0
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_all_waiters_wake_on_recovery():
    """Every parked waiter returns True once the shared prober sees recovery."""
    health = BackendHealth()
    health.set_up(0, False)
    calls = 0

    async def probe() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    tasks = [
        asyncio.create_task(health.wait_up(0, timeout=2.0, probe=probe, interval=0.02))
        for _ in range(4)
    ]

    assert await asyncio.gather(*tasks) == [True, True, True, True]
    assert health.waiters(0) == 0
    await asyncio.sleep(0)
    assert not health.prober_running(0)


@pytest.mark.asyncio
async def test_timed_out_waiter_leaves_nothing_behind():
    """A waiter that exhausts its budget cancels the idle prober and detaches."""
    health = BackendHealth()
    health.set_up(0, False)
    probe = AsyncMock(return_value=False)

    assert await health.wait_up(0, timeout=0.05, probe=probe, interval=0.01) is False
    assert health.waiters(0) == 0
    assert not health.prober_running(0)


@pytest.mark.asyncio
async def test_cancelled_waiter_leaves_nothing_behind():
    """A killed waiter detaches cleanly (no waiter count, no prober task)."""
    health = BackendHealth()
    health.set_up(0, False)
    probe = AsyncMock(return_value=False)
    task = asyncio.create_task(
        health.wait_up(0, timeout=5.0, probe=probe, interval=0.01)
    )
    await asyncio.sleep(0.03)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert health.waiters(0) == 0
    assert not health.prober_running(0)


@pytest.mark.asyncio
async def test_backoff_grows_and_caps():
    """Probe cadence doubles from the base up to the cap while down."""
    health = BackendHealth(backoff_s=0.05, max_s=0.2)
    health.set_up(0, False)
    times: list[float] = []

    async def probe() -> bool:
        times.append(time.monotonic())
        return False

    task = asyncio.create_task(
        health.wait_up(0, timeout=5.0, probe=probe, interval=0.001)
    )
    await asyncio.sleep(0.7)
    await _drain(task)

    gaps = [b - a for a, b in pairwise(times)]
    assert len(gaps) >= 3
    assert gaps[0] == pytest.approx(0.05, abs=0.04)
    assert gaps[-1] > gaps[0]
    assert gaps[-1] <= 0.2 + 0.06


@pytest.mark.asyncio
async def test_zero_backoff_keeps_fixed_interval():
    """Backoff 0 (default) keeps the fast fixed cadence for prompt recovery."""
    health = BackendHealth(backoff_s=0.0, max_s=30.0)
    health.set_up(0, False)
    times: list[float] = []

    async def probe() -> bool:
        times.append(time.monotonic())
        return False

    task = asyncio.create_task(
        health.wait_up(0, timeout=5.0, probe=probe, interval=0.02)
    )
    await asyncio.sleep(0.25)
    await _drain(task)

    gaps = [b - a for a, b in pairwise(times)]
    assert len(gaps) >= 5
    assert max(gaps) < 0.12
