# tests/test_slot_discovery.py

"""P3-2: slot discovery via GET /slots, pool narrowing on mismatch,
and the aggregated /slots endpoint."""

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


def test_set_backend_slots_narrows_pool(sm):
    """Fewer actual slots than configured: the pool is narrowed."""
    sm.set_backend_slots(0, [_slot(0), _slot(1)])

    assert (0, 0) in sm._active_slots
    assert (0, 1) in sm._active_slots
    assert (0, 2) not in sm._active_slots
    assert (0, 3) not in sm._active_slots
    assert sm._backend_slots[0] == [_slot(0), _slot(1)]


def test_set_backend_slots_full_count_keeps_pool(sm):
    """All configured slots reported: the pool is unchanged."""
    sm.set_backend_slots(0, [_slot(i) for i in range(4)])

    assert len(sm._active_slots) == 4


def test_routing_uses_narrowed_pool(sm):
    """After narrowing, routing only picks slots that actually exist."""
    sm.set_backend_slots(0, [_slot(0), _slot(1)])

    g, _lock = sm._get_free_or_oldest()

    assert g in ((0, 0), (0, 1))


def test_aggregated_state(sm):
    """aggregated_state merges backend fields with the proxy LRU mark."""
    sm.set_backend_slots(0, [_slot(0), _slot(1)])
    sm._last_used[(0, 0)] = 123.0

    state = sm.aggregated_state()

    assert len(state) == 2
    first = next(s for s in state if s["slot"] == 0)
    assert first["backend"] == 0
    assert first["state"] == "busy"
    assert first["n_ctx"] == 4096
    assert first["total_tokens"] == 100
    assert first["last_used"] == 123.0
    # Never-used slots carry the initial LRU mark 0.0 (not None).
    second = next(s for s in state if s["slot"] == 1)
    assert second["last_used"] == 0.0


async def test_slots_endpoint(sm):
    """GET /slots returns the aggregated state."""
    sm.set_backend_slots(0, [_slot(0)])
    app_module.app.state.sm = sm

    resp = await app_module.slots_state()

    assert len(resp["slots"]) == 1
    assert resp["slots"][0]["slot"] == 0
