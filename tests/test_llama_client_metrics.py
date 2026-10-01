# tests/test_llama_client_metrics.py

"""LlamaClient metrics + active-model discovery for the /metrics aggregator.

- get_active_models(): router -> only the loaded models; plain -> the single
  model; undeterminable -> empty list.
- get_metrics(model): raw /metrics text (?model=X), short timeout, None on
  any failure (a down backend must not break the scrape).
"""

from unittest.mock import AsyncMock, MagicMock

from backend.llama_client import LlamaClient


def make_client():
    c = LlamaClient("http://be")
    c.client = MagicMock()
    return c


def text_resp(status_code, text):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.raise_for_status = MagicMock()
    return r


def router_models_resp():
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(
        return_value={
            "data": [
                {"id": "qwen.fast", "status": {"value": "unloaded"}},
                {"id": "qwen.nvfp4", "status": {"value": "loaded"}},
            ]
        }
    )
    r.raise_for_status = MagicMock()
    return r


def plain_models_resp(model_id):
    r = MagicMock()
    r.status_code = 200
    r.json = MagicMock(return_value={"data": [{"id": model_id}]})
    r.raise_for_status = MagicMock()
    return r


# --- get_active_models -----------------------------------------------------


async def test_get_active_models_router_loaded_only():
    c = make_client()
    c.client.get = AsyncMock(return_value=router_models_resp())
    assert await c.get_active_models() == ["qwen.nvfp4"]


async def test_get_active_models_plain_single():
    c = make_client()
    c.client.get = AsyncMock(return_value=plain_models_resp("m1"))
    assert await c.get_active_models() == ["m1"]


async def test_get_active_models_unknown_empty():
    c = make_client()
    c.client.get = AsyncMock(side_effect=Exception("down"))
    assert await c.get_active_models() == []


# --- get_metrics -----------------------------------------------------------


async def test_get_metrics_returns_raw_text():
    c = make_client()
    c.client.get = AsyncMock(return_value=text_resp(200, "foo 1\n"))
    assert await c.get_metrics("m1") == "foo 1\n"


async def test_get_metrics_adds_model_param():
    c = make_client()
    c.client.get = AsyncMock(return_value=text_resp(200, "foo 1\n"))
    await c.get_metrics("m1")
    assert c.client.get.call_args.kwargs.get("params") == {"model": "m1"}


async def test_get_metrics_no_param_when_none():
    c = make_client()
    c.client.get = AsyncMock(return_value=text_resp(200, "foo 1\n"))
    await c.get_metrics()
    assert c.client.get.call_args.kwargs.get("params") is None


async def test_get_metrics_uses_short_timeout():
    c = make_client()
    c.client.get = AsyncMock(return_value=text_resp(200, "foo 1\n"))
    await c.get_metrics("m1")
    kwargs = c.client.get.call_args.kwargs
    assert "timeout" in kwargs, "/metrics must use an explicit short timeout"
    assert kwargs["timeout"] <= 10, f"timeout too large: {kwargs['timeout']}"


async def test_get_metrics_none_on_error():
    c = make_client()
    c.client.get = AsyncMock(side_effect=Exception("down"))
    assert await c.get_metrics("m1") is None
