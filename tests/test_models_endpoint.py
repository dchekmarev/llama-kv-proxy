# tests/test_models_endpoint.py

"""/v1/models must proxy the union of the backends' model lists, with a static MODEL_ID fallback."""

from unittest.mock import AsyncMock, MagicMock

import app as app_module
from llama_client import LlamaClient


def make_client():
    c = LlamaClient("http://be")
    c.client = MagicMock()
    return c


def fake_models_resp(ids):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": [{"id": i} for i in ids]})
    return resp


async def test_get_models_returns_backend_list():
    """The full backend model list is returned as-is."""
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp(["m1", "m2"]))
    assert await c.get_models() == [{"id": "m1"}, {"id": "m2"}]


async def test_get_models_uses_short_timeout():
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp(["m1"]))
    await c.get_models()
    kwargs = c.client.get.call_args.kwargs
    assert "timeout" in kwargs, "/v1/models must use an explicit short timeout"
    assert kwargs["timeout"] <= 10, f"timeout too large: {kwargs['timeout']}"


async def test_get_models_none_on_error():
    c = make_client()
    c.client.get = AsyncMock(side_effect=Exception("down"))
    assert await c.get_models() is None


async def test_get_models_none_on_unexpected_shape():
    c = make_client()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": "not-a-list"})
    c.client.get = AsyncMock(return_value=resp)
    assert await c.get_models() is None


async def test_models_endpoint_proxies_backend():
    """The endpoint returns the backend list verbatim."""
    mock = MagicMock()
    mock.get_models = AsyncMock(return_value=[{"id": "m1"}, {"id": "m2"}])
    app_module.app.state.clients = [mock]

    resp = await app_module.models()

    assert resp == {"data": [{"id": "m1"}, {"id": "m2"}]}


async def test_models_endpoint_fallback_when_backend_down(monkeypatch):
    """Backend unavailable -> the configured MODEL_ID is advertised instead."""
    monkeypatch.setattr(app_module, "MODEL_ID", "llama.cpp")
    mock = MagicMock()
    mock.get_models = AsyncMock(return_value=None)
    app_module.app.state.clients = [mock]

    resp = await app_module.models()

    assert resp == {"data": [{"id": "llama.cpp"}]}


async def test_models_endpoint_unions_all_backends():
    """All backends are queried; the union is deduped by id, first-seen order."""
    c1 = MagicMock()
    c1.get_models = AsyncMock(return_value=[{"id": "m1"}, {"id": "m2"}])
    c2 = MagicMock()
    c2.get_models = AsyncMock(return_value=[{"id": "m2"}, {"id": "m3"}])
    app_module.app.state.clients = [c1, c2]

    resp = await app_module.models()

    assert resp == {"data": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]}
    assert c1.get_models.await_count == 1
    assert c2.get_models.await_count == 1


async def test_models_endpoint_partial_outage_returns_live_backends():
    """One backend down -> the live backends' models are still advertised."""
    c1 = MagicMock()
    c1.get_models = AsyncMock(return_value=None)
    c2 = MagicMock()
    c2.get_models = AsyncMock(return_value=[{"id": "m9"}])
    app_module.app.state.clients = [c1, c2]

    resp = await app_module.models()

    assert resp == {"data": [{"id": "m9"}]}


async def test_models_endpoint_fallback_when_all_backends_down(monkeypatch):
    """Every backend down -> the configured MODEL_ID is advertised."""
    monkeypatch.setattr(app_module, "MODEL_ID", "llama.cpp")
    clients = []
    for i in range(2):
        c = MagicMock()
        c.get_models = AsyncMock(return_value=None)
        clients.append(c)
    app_module.app.state.clients = clients

    resp = await app_module.models()

    assert resp == {"data": [{"id": "llama.cpp"}]}
