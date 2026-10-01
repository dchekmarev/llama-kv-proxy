# tests/test_health.py

"""P3-4: /proxy/health endpoint — backend availability probe plus slot state."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
from backend import slot_manager as sm_module
from backend.llama_client import LlamaClient
from backend.slot_manager import SlotManager


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    return SlotManager()


async def test_health_ok():
    """A 200 /v1/models response reports ok with the model id."""
    client = LlamaClient("http://be")
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"data": [{"id": "m1"}]})
    resp.raise_for_status = MagicMock()
    client.client.get = AsyncMock(return_value=resp)

    h = await client.health()

    assert h["ok"] is True
    assert h["model_id"] == "m1"
    assert h["url"] == "http://be"
    await client.close()


async def test_health_backend_down():
    """A backend error reports ok=False, never an exception."""
    client = LlamaClient("http://be")
    client.client.get = AsyncMock(side_effect=Exception("down"))

    h = await client.health()

    assert h["ok"] is False
    assert h["model_id"] is None
    await client.close()


async def test_health_endpoint_mixed_backends(sm):
    """/proxy/health aggregates per-backend probes and the overall ok flag."""
    good = MagicMock()
    good.health = AsyncMock(
        return_value={"url": "http://be1", "ok": True, "model_id": "m1"}
    )
    bad = MagicMock()
    bad.health = AsyncMock(
        return_value={"url": "http://be2", "ok": False, "model_id": None}
    )
    sm.set_backend_slots(0, "m1", [{"id": 0, "state": "busy"}])
    app_module.app.state.sm = sm
    app_module.app.state.clients = [good, bad]

    resp = await app_module.health()

    assert resp["ok"] is False
    assert len(resp["backends"]) == 2
    assert resp["backends"][0]["ok"] is True
    assert resp["backends"][1]["ok"] is False
    assert len(resp["slots"]) == 1
    assert resp["slots"][0]["slot"] == 0
