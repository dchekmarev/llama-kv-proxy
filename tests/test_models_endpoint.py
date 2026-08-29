# tests/test_models_endpoint.py

"""/v1/models must proxy the backend model list, with a static MODEL_ID fallback."""

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
