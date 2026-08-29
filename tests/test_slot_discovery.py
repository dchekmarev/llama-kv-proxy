# tests/test_slot_discovery.py

"""P3-2: slot discovery via GET /slots, per-model pools, and the aggregated
/proxy/slots endpoint.

Pools are discovery-driven and keyed by (backend, model): a plain backend has
one pool, a router has one pool per loaded model.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import slot_manager as sm_module
from llama_client import LlamaClient
from slot_manager import SlotManager


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 4}])
    return SlotManager()


def _slot(i: int, state: str = "busy") -> dict:
    return {"id": i, "state": state, "n_ctx": 4096, "total_tokens": 100 + i}


async def test_get_slots_parses_list():
    """A JSON list response is returned as-is."""
    client = LlamaClient("http://be")
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=[_slot(0), _slot(1)])
    client.client.get = AsyncMock(return_value=resp)

    slots = await client.get_slots()

    assert slots == [_slot(0), _slot(1)]
    await client.close()


async def test_get_slots_non_list_response():
    """A non-list JSON response is treated as unsupported."""
    client = LlamaClient("http://be")
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"error": "nope"})
    client.client.get = AsyncMock(return_value=resp)

    assert await client.get_slots() is None
    await client.close()


async def test_get_slots_backend_error():
    """A backend error is a soft failure (None), never an exception."""
    client = LlamaClient("http://be")
    client.client.get = AsyncMock(side_effect=Exception("down"))

    assert await client.get_slots() is None
    await client.close()


def test_set_backend_slots_populates_pool(sm):
    """Discovery populates the (backend, model) pool with the reported slots."""
    sm.set_backend_slots(0, "m1", [_slot(0), _slot(1)])

    assert sm._pools[(0, "m1")] == [0, 1]
    assert sm._backend_slots[(0, "m1")] == [_slot(0), _slot(1)]


def test_set_backend_slots_replaces_pool(sm):
    """A later discovery replaces the pool (e.g. a model was reloaded)."""
    sm.set_backend_slots(0, "m1", [_slot(0), _slot(1), _slot(2)])
    sm.set_backend_slots(0, "m1", [_slot(0)])

    assert sm._pools[(0, "m1")] == [0]


def test_router_per_model_pools(sm):
    """A router keeps separate pools per model on the same backend."""
    sm.set_backend_slots(0, "modelA", [_slot(0)])
    sm.set_backend_slots(0, "modelB", [_slot(0)])

    assert sm._pools[(0, "modelA")] == [0]
    assert sm._pools[(0, "modelB")] == [0]
    assert sm._slots_for_model("modelA") == [(0, "modelA", 0)]
    assert sm._slots_for_model("modelB") == [(0, "modelB", 0)]


def test_routing_uses_discovered_slots(sm):
    """Routing only picks slots that were actually discovered."""
    sm.set_backend_slots(0, "m1", [_slot(0), _slot(1)])

    g, _lock = sm._get_free_or_oldest("m1")

    assert g in ((0, "m1", 0), (0, "m1", 1))


def test_aggregated_state(sm):
    """aggregated_state merges backend fields, the model, and the LRU mark."""
    sm.set_backend_slots(0, "m1", [_slot(0), _slot(1)])
    sm._last_used[(0, "m1", 0)] = 123.0

    state = sm.aggregated_state()

    assert len(state) == 2
    first = next(s for s in state if s["slot"] == 0)
    assert first["backend"] == 0
    assert first["model"] == "m1"
    assert first["state"] == "busy"
    assert first["n_ctx"] == 4096
    assert first["total_tokens"] == 100
    assert first["last_used"] == 123.0
    # Never-used slots carry no LRU mark (None).
    second = next(s for s in state if s["slot"] == 1)
    assert second["last_used"] is None


async def test_slots_endpoint(sm):
    """GET /proxy/slots returns the aggregated state."""
    sm.set_backend_slots(0, "m1", [_slot(0)])
    app_module.app.state.sm = sm

    resp = await app_module.slots_state()

    assert len(resp["slots"]) == 1
    assert resp["slots"][0]["slot"] == 0
    assert resp["slots"][0]["model"] == "m1"
