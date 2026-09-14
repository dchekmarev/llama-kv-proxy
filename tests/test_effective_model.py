# tests/test_effective_model.py

"""H1: a model-less request must resolve the effective model via the TTL-cached
get_model_id_cached (no per-request HTTP round-trip), falling back to MODEL_ID
when the cached id is "unknown"."""

from unittest.mock import AsyncMock

import pytest

import app as app_module


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


def _chat(data):
    return app_module.chat(FakeRequest(data))


def _small_data():
    return {
        "messages": [{"role": "user", "content": "hello world"}],
        "stream": False,
    }


@pytest.mark.asyncio
async def test_modelless_request_uses_cached_model_id(sm):
    """A request without `model` resolves via get_model_id_cached and never
    calls the uncached get_loaded_model."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]

    await _chat(_small_data())

    client.get_model_id_cached.assert_awaited_once()
    client.get_loaded_model.assert_not_awaited()
    # The resolved id is forwarded to the backend.
    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "m1"


@pytest.mark.asyncio
async def test_modelless_unknown_falls_back_to_model_id(sm):
    """get_model_id_cached -> "unknown" must fall back to MODEL_ID, not send
    the literal string "unknown" to the backend."""
    client = sm.backends[0]["client"]
    client.get_model_id_cached = AsyncMock(return_value="unknown")
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]

    await _chat(_small_data())

    body = client.chat_completions.await_args.args[0]
    assert body["model"] == app_module.MODEL_ID
    assert body["model"] != "unknown"


@pytest.mark.asyncio
async def test_client_model_wins_over_cache(sm):
    """An explicit client `model` that has a discovered pool is used as-is;
    the cache is not consulted."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "client-model", [{"id": 0}, {"id": 1}])

    data = _small_data()
    data["model"] = "client-model"
    await _chat(data)

    client.get_model_id_cached.assert_not_awaited()
    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "client-model"


@pytest.mark.asyncio
async def test_alias_with_single_pool_resolves_to_loaded_model(sm):
    """An alias model with no discovered pool resolves to the single detected
    model, so pooling and cache keys are shared with real-name requests."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}, {"id": 1}])

    data = _small_data()
    data["model"] = "default"
    await _chat(data)

    client.get_model_id_cached.assert_awaited_once()
    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "m1"


@pytest.mark.asyncio
async def test_alias_with_no_pool_stays_as_is(sm):
    """Startup transient: before slot discovery no pool exists for any model,
    so the alias is left untouched (bootstrap/autoload path, as before)."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]

    data = _small_data()
    data["model"] = "default"
    await _chat(data)

    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "default"


@pytest.mark.asyncio
async def test_alias_with_multiple_pools_stays_as_is(sm):
    """Ambiguous: several models detected, so an unknown alias is not mapped
    (the proxy cannot know which model the backend resolves it to)."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}])
    sm.set_backend_slots(0, "m2", [{"id": 0}])

    data = _small_data()
    data["model"] = "default"
    await _chat(data)

    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "default"
