# tests/test_llama_client.py

"""P0-4: model_id must be cached with a TTL and fetched with a short timeout."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import config
from llama_client import LlamaClient


def make_client():
    c = LlamaClient("http://be")
    c.client = MagicMock()
    return c


def fake_models_resp(model_id):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"data": [{"id": model_id}]})
    return resp


@pytest.mark.asyncio
async def test_model_id_cached_within_ttl():
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    assert await c.get_model_id_cached() == "m1"
    assert await c.get_model_id_cached() == "m1"
    assert c.client.get.await_count == 1


@pytest.mark.asyncio
async def test_model_id_refetched_after_ttl():
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    await c.get_model_id_cached()
    c._model_id_at -= config.MODEL_ID_TTL + 1  # expire the cache
    assert await c.get_model_id_cached() == "m1"
    assert c.client.get.await_count == 2


@pytest.mark.asyncio
async def test_failure_falls_back_to_last_known_id():
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    assert await c.get_model_id_cached() == "m1"
    c._model_id_at -= config.MODEL_ID_TTL + 1  # expire
    c.client.get = AsyncMock(side_effect=RuntimeError("backend down"))
    assert await c.get_model_id_cached() == "m1", "must fall back to the last known id"


@pytest.mark.asyncio
async def test_unknown_retries_quickly():
    c = make_client()
    c.client.get = AsyncMock(side_effect=RuntimeError("backend down"))
    assert await c.get_model_id_cached() == "unknown"
    # Immediately after: no HTTP call (short retry interval)
    assert await c.get_model_id_cached() == "unknown"
    assert c.client.get.await_count == 1
    # After the short retry interval: fetch again
    c._model_id_at -= config.UNKNOWN_MODEL_ID_RETRY + 1
    assert await c.get_model_id_cached() == "unknown"
    assert c.client.get.await_count == 2


@pytest.mark.asyncio
async def test_get_model_id_uses_short_timeout():
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    await c.get_model_id()
    kwargs = c.client.get.call_args.kwargs
    assert "timeout" in kwargs, "/v1/models must use an explicit short timeout"
    assert kwargs["timeout"] <= 10, f"timeout too large: {kwargs['timeout']}"
