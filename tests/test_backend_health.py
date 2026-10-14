# tests/test_backend_health.py

"""Health-aware backend selection with several backends.

The slot poll already records each backend's liveness (llama_kv_proxy_backend_up).
Model-id/alias resolution, the pass-through client pick and slot candidates
consult that verdict: a backend currently known down is skipped while a healthy
one can answer, and only when every backend is down does the proxy fall back to
the first backend -- exactly the pre-health behavior, with the HOLD governing
the request.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import chat_flow
from backend import slot_manager as sm_module
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
