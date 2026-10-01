# tests/test_effective_model.py

"""H1: a model-less request must resolve the effective model via the TTL-cached
get_model_id_cached (no per-request HTTP round-trip), falling back to MODEL_ID
when the cached id is "unknown".

A client `model` alias with no discovered pool is mapped to a model id -- first
through the backend's preset table (ids + aliases), then through the single
detected model. When neither applies the name goes upstream unchanged but the
request is proxied without cache treatment, so the alias never becomes a cache
namespace of its own."""

from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow


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
async def test_alias_resolves_via_preset_table(sm):
    """The primary resolution: the backend's own preset table (ids + aliases).
    It does not depend on which model is loaded, so the alias maps to the real
    id and shares the pool and cache keys with real-name requests."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}, {"id": 1}])
    client.resolve_model_id_cached = AsyncMock(return_value="m1")

    data = _small_data()
    data["model"] = "default"
    await _chat(data)

    client.resolve_model_id_cached.assert_awaited_once_with("default")
    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "m1"


@pytest.mark.asyncio
async def test_alias_table_beats_single_model_heuristic(sm):
    """The preset table is authoritative: when it maps the name, the single
    detected model is not consulted (they can disagree mid-switch)."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}])
    client.resolve_model_id_cached = AsyncMock(return_value="m-from-table")

    data = _small_data()
    data["model"] = "default"
    await _chat(data)

    client.get_model_id_cached.assert_not_awaited()
    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "m-from-table"


@pytest.mark.asyncio
async def test_alias_falls_back_to_single_model_when_not_in_preset(sm):
    """A name outside the preset: a plain single-model backend may still ignore
    the requested name and serve its only model, so the single detected model
    is used when it is unambiguous."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}])

    data = _small_data()
    data["model"] = "default"
    await _chat(data)

    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "m1"


@pytest.mark.asyncio
async def test_unresolvable_alias_is_proxied_without_cache(sm, caplog, monkeypatch):
    """Truly unresolvable (nothing in the preset, several models detected): the
    name goes upstream unchanged -- the backend may well serve it -- but the
    request is proxied without cache treatment, so the alias never becomes a
    cache namespace of its own. A WARNING marks the window."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}])
    sm.set_backend_slots(0, "m2", [{"id": 0}])

    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save = AsyncMock()
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save)

    data = _small_data()
    data["model"] = "default"
    with caplog.at_level("WARNING"):
        await _chat(data)

    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "default", "the backend still gets the name it asked for"
    assert "model_alias_unresolved" in caplog.text
    # No restore and no save: the request must not create an alias namespace.
    assert sm._last_saved == {}, "nothing may be saved under the alias"
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_unresolvable_alias_when_backend_id_unknown(sm, caplog, monkeypatch):
    """A backend restart can leave the router with no loaded model and an empty
    table: the same no-cache treatment, never a namespace of the alias."""
    client = sm.backends[0]["client"]
    client.resolve_model_id_cached = AsyncMock(return_value=None)
    client.get_model_id_cached = AsyncMock(return_value="unknown")
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]

    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save = AsyncMock()
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save)

    data = _small_data()
    data["model"] = "default"
    with caplog.at_level("WARNING"):
        await _chat(data)

    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "default"
    assert "model_alias_unresolved" in caplog.text
    assert sm._last_saved == {}, "nothing may be saved under the alias"
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolved_alias_keeps_cache_treatment(sm, caplog, monkeypatch):
    """The contrast: a resolved alias is cached like any real-name request, and
    the noisy WARNING is reserved for the unresolvable path."""
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}, {"id": 1}])
    client.resolve_model_id_cached = AsyncMock(return_value="m1")

    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save = AsyncMock()
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save)

    data = _small_data()
    data["model"] = "default"
    with caplog.at_level("WARNING"):
        await _chat(data)

    assert "model_alias_unresolved" not in caplog.text
    body = client.chat_completions.await_args.args[0]
    assert body["model"] == "m1"
    # Contrast: a resolved alias keeps the normal cache treatment, so the slot
    # is tracked under the real model id and not under the alias.
    assert all(m == "m1" for (_be, m) in sm._pools)
    assert not any(m == "default" for (_be, m) in sm._pools)


@pytest.mark.asyncio
async def test_unresolvable_alias_stays_uncached_on_every_request(
    sm, monkeypatch, caplog
):
    """Two requests with the same unresolvable alias.

    A plain backend reports its slots under whatever name it routes on, so the
    on-demand freshen of the alias would register a pool for it -- and the next
    request would take the alias for a real model id and cache under it. Two
    detected models keep the single-model fallback out of the way, so the alias
    really is unresolvable."""
    client = sm.backends[0]["client"]
    # A live backend: the freshen re-poll answers with its slot, as llama.cpp
    # does, instead of the fixture's "no re-poll" (None).
    client.get_slots = AsyncMock(return_value=[{"id": 0, "model": "m1"}])
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}, {"id": 1}])
    sm.set_backend_slots(0, "m2", [{"id": 0}])

    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save = AsyncMock()
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save)

    data = _small_data()
    data["model"] = "default"
    for _ in range(2):
        with caplog.at_level("WARNING"):
            await _chat(data)

    # Unresolved on every request, not just the first one.
    assert caplog.text.count("model_alias_unresolved") == 2
    assert sm._last_saved == {}, "nothing may be saved under the alias"
    save.assert_not_awaited()
    # The alias never becomes a pool, so it can never be mistaken for a model.
    assert not any(m == "default" for (_be, m) in sm._pools)
    assert sm.discovered_models() == {"m1", "m2"}


@pytest.mark.asyncio
async def test_cache_prompt_false_still_freshens(sm, monkeypatch):
    """cache_prompt: false is not the same as an unresolved alias: the name is
    real, only the reuse is off, so the on-demand slot refresh must still run
    (a stale slot_id wraps onto a live physical slot)."""
    client = sm.backends[0]["client"]
    client.get_slots = AsyncMock(return_value=[{"id": 0, "model": "m1"}])
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    sm.set_backend_slots(0, "m1", [{"id": 0}, {"id": 1}])

    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save = AsyncMock()
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save)

    data = _small_data()
    data["model"] = "m1"
    data["cache_prompt"] = False
    await _chat(data)

    client.get_slots.assert_awaited()
    assert client.chat_completions.await_args.args[0]["model"] == "m1"
    save.assert_not_awaited()
