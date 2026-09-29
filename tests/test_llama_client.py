# tests/test_llama_client.py

"""P0-4: model_id must be cached with a TTL and fetched with a short timeout."""

import asyncio
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
    c._models._at -= config.MODEL_ID_TTL + 1  # expire the cache
    assert await c.get_model_id_cached() == "m1"
    assert c.client.get.await_count == 2


@pytest.mark.asyncio
async def test_failure_falls_back_to_last_known_id():
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    assert await c.get_model_id_cached() == "m1"
    c._models._at -= config.MODEL_ID_TTL + 1  # expire
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
    c._models._at -= config.UNKNOWN_MODEL_ID_RETRY + 1
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


@pytest.mark.asyncio
async def test_concurrent_expired_callers_singleflight():
    """N concurrent callers on an expired cache must trigger exactly ONE fetch."""
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    await c.get_model_id_cached()
    c._models._at -= config.MODEL_ID_TTL + 1  # expire the cache

    calls = []

    async def counted():
        calls.append(1)
        await asyncio.sleep(0.05)
        return [{"id": "m1"}]

    c.get_models = counted
    results = await asyncio.gather(*[c.get_model_id_cached() for _ in range(8)])
    assert len(calls) == 1, f"expected exactly one fetch, got {len(calls)}"
    assert all(r == "m1" for r in results)


@pytest.mark.asyncio
async def test_singleflight_failure_propagates_and_retries():
    """A failed fetch must propagate to all waiters and be retried next call."""
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    await c.get_model_id_cached()
    c._models._at -= config.MODEL_ID_TTL + 1  # expire the cache

    async def failing():
        await asyncio.sleep(0.05)
        raise RuntimeError("backend down")

    c.get_models = failing
    with pytest.raises(RuntimeError):
        await asyncio.gather(*[c.get_model_id_cached() for _ in range(3)])

    # The in-flight state must be cleared: the next call retries the fetch.
    async def ok():
        await asyncio.sleep(0.01)
        return [{"id": "m1"}]

    c.get_models = ok
    assert await c.get_model_id_cached() == "m1"


@pytest.mark.asyncio
async def test_cancelled_fetcher_clears_inflight_and_allows_retry():
    """Cancelling the fetcher mid-flight must clear _models._inflight so a
    subsequent call does not hang on a dangling pending Future."""
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    await c.get_model_id_cached()
    c._models._at -= config.MODEL_ID_TTL + 1  # expire the cache

    async def slow():
        await asyncio.sleep(5)
        return [{"id": "m1"}]

    c.get_models = slow
    fetcher = asyncio.create_task(c.get_model_id_cached())
    await asyncio.sleep(0.05)  # let the fetcher start and create the Future
    assert c._models._inflight is not None
    fetcher.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fetcher

    assert c._models._inflight is None, "in-flight Future must be cleared on cancellation"

    # A subsequent call must not hang on the dangling Future.
    async def ok():
        await asyncio.sleep(0.01)
        return [{"id": "m1"}]

    c.get_models = ok
    assert await asyncio.wait_for(c.get_model_id_cached(), timeout=1.0) == "m1"


@pytest.mark.asyncio
async def test_cancelled_fetcher_waiters_survive_and_refetch():
    """Cancelling only the fetcher (its client disconnected) must not kill the
    concurrent waiters: they hold no slot and were not cancelled themselves,
    so they must re-trigger a fresh fetch and get the model id."""
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    await c.get_model_id_cached()
    c._models._at -= config.MODEL_ID_TTL + 1  # expire the cache

    fetches = 0

    async def slow_then_fast():
        nonlocal fetches
        fetches += 1
        if fetches == 1:
            await asyncio.sleep(5)  # first fetch: long enough to be cancelled
        await asyncio.sleep(0.01)
        return [{"id": "m1"}]

    c.get_models = slow_then_fast
    tasks = [asyncio.create_task(c.get_model_id_cached()) for _ in range(3)]
    await asyncio.sleep(0.05)  # fetcher starts; waiters await the shared Future
    assert c._models._inflight is not None
    tasks[0].cancel()  # only the fetcher's client disconnects

    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(results[0], asyncio.CancelledError)
    assert all(r == "m1" for r in results[1:]), f"waiters must survive: {results}"
    assert fetches == 2, f"waiters must re-trigger exactly one fresh fetch: {fetches}"


@pytest.mark.asyncio
async def test_fresh_cache_fast_path_no_fetch():
    """A fresh cache must return immediately without any fetch."""
    c = make_client()
    c.client.get = AsyncMock(return_value=fake_models_resp("m1"))
    assert await c.get_model_id_cached() == "m1"
    assert await c.get_model_id_cached() == "m1"
    assert c.client.get.await_count == 1


# --- slot pin --------------------------------------------------------------
# The contract of the slot pin: llama.cpp accepts the target slot in three
# places at once, and the caller's body must survive untouched.


def test_slot_pin_none_returns_body_unchanged():
    body = {"messages": [], "model": "m1"}
    out_body, out_query = LlamaClient._with_slot_id(body, None)
    assert out_query == {}
    assert out_body == body
    assert out_body is body, "no pin means no copy"


def test_slot_pin_sets_all_three_locations():
    body = {"messages": [], "model": "m1", "options": {"temperature": 0.4}}
    out_body, out_query = LlamaClient._with_slot_id(body, 3)

    assert out_body["_slot_id"] == 3
    assert out_body["slot_id"] == 3
    assert out_body["id_slot"] == 3
    assert out_body["options"] == {"temperature": 0.4, "slot_id": 3, "id_slot": 3}
    assert out_query == {"slot_id": 3, "id_slot": 3}


def test_slot_pin_creates_options_when_absent():
    out_body, _query = LlamaClient._with_slot_id({"messages": []}, 0)
    assert out_body["options"] == {"slot_id": 0, "id_slot": 0}


def test_slot_pin_treats_empty_options_as_absent():
    out_body, _query = LlamaClient._with_slot_id({"options": {}}, 1)
    assert out_body["options"] == {"slot_id": 1, "id_slot": 1}


def test_slot_pin_treats_null_options_as_absent():
    out_body, _query = LlamaClient._with_slot_id({"options": None}, 1)
    assert out_body["options"] == {"slot_id": 1, "id_slot": 1}


def test_slot_pin_does_not_mutate_caller_body():
    body = {"messages": [], "options": {"temperature": 0.4}}
    LlamaClient._with_slot_id(body, 2)
    assert body == {"messages": [], "options": {"temperature": 0.4}}


def test_slot_pin_overwrites_a_preexisting_pin():
    body = {"slot_id": 9, "id_slot": 9, "options": {"slot_id": 9}}
    out_body, out_query = LlamaClient._with_slot_id(body, 5)
    assert out_body["slot_id"] == 5
    assert out_body["id_slot"] == 5
    assert out_body["_slot_id"] == 5
    assert out_body["options"]["slot_id"] == 5
    assert out_query["slot_id"] == 5
