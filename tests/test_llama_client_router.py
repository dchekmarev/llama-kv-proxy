# tests/test_llama_client_router.py

"""Router-aware LlamaClient: model-scoped slot ops, loaded-model resolution,
router detection.

A router backend (llama-server --models-preset) requires a `model` parameter
on /slots operations and reports per-model load state via a `status` field in
/v1/models. A plain single-model backend has neither. The client must:
- add `?model=X` to slot ops only when a model is supplied;
- resolve the *loaded* model (not just the first entry);
- detect router mode from the presence of a `status` field.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llama_client import LlamaClient


def make_client():
    c = LlamaClient("http://be")
    c.client = MagicMock()
    return c


def resp(status_code, payload):
    r = MagicMock()
    r.status_code = status_code
    r.json = MagicMock(return_value=payload)
    r.raise_for_status = MagicMock()
    return r


def router_models_resp():
    return resp(
        200,
        {
            "data": [
                {"id": "qwen.fast", "status": {"value": "unloaded"}},
                {"id": "qwen.nvfp4", "status": {"value": "loaded"}},
            ]
        },
    )


def plain_models_resp(model_id):
    return resp(200, {"data": [{"id": model_id}]})


# --- get_slots -------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_slots_with_model_adds_param():
    c = make_client()
    c.client.get = AsyncMock(return_value=resp(200, [{"id": 0}]))
    slots = await c.get_slots(model="qwen.nvfp4")
    assert slots == [{"id": 0}]
    assert c.client.get.call_args.kwargs.get("params") == {"model": "qwen.nvfp4"}


@pytest.mark.asyncio
async def test_get_slots_without_model_sends_no_param():
    c = make_client()
    c.client.get = AsyncMock(return_value=resp(200, [{"id": 0}]))
    await c.get_slots()
    assert c.client.get.call_args.kwargs.get("params") is None


# --- save_slot / restore_slot ---------------------------------------------


@pytest.mark.asyncio
async def test_save_slot_with_model_in_body():
    """A router routes the save by the model in the BODY, not the query."""
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(200, {}))
    assert await c.save_slot(0, "abc", model="qwen.nvfp4") is True
    assert c.client.post.call_args.kwargs.get("params") == {"action": "save"}
    assert c.client.post.call_args.kwargs.get("json") == {
        "filename": "abc",
        "model": "qwen.nvfp4",
    }


@pytest.mark.asyncio
async def test_save_slot_without_model_omits_model():
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(200, {}))
    await c.save_slot(0, "abc")
    assert c.client.post.call_args.kwargs.get("params") == {"action": "save"}
    assert c.client.post.call_args.kwargs.get("json") == {"filename": "abc"}


@pytest.mark.asyncio
async def test_restore_slot_with_model_in_body():
    """A router routes the restore by the model in the BODY, not the query."""
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(200, {}))
    assert await c.restore_slot(0, "abc", model="qwen.nvfp4") is True
    assert c.client.post.call_args.kwargs.get("params") == {"action": "restore"}
    assert c.client.post.call_args.kwargs.get("json") == {
        "filename": "abc",
        "model": "qwen.nvfp4",
    }


@pytest.mark.asyncio
async def test_restore_slot_without_model_omits_model():
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(200, {}))
    await c.restore_slot(0, "abc")
    assert c.client.post.call_args.kwargs.get("params") == {"action": "restore"}
    assert c.client.post.call_args.kwargs.get("json") == {"filename": "abc"}


# --- delete_cache_file -----------------------------------------------------


@pytest.mark.asyncio
async def test_delete_cache_file_with_model_adds_param():
    c = make_client()
    c.client.delete = AsyncMock(return_value=resp(200, {}))
    assert await c.delete_cache_file("abc", model="qwen.nvfp4") is True
    assert c.client.delete.call_args.kwargs.get("params") == {
        "filename": "abc",
        "model": "qwen.nvfp4",
    }


@pytest.mark.asyncio
async def test_delete_cache_file_without_model_omits_param():
    c = make_client()
    c.client.delete = AsyncMock(return_value=resp(200, {}))
    await c.delete_cache_file("abc")
    assert c.client.delete.call_args.kwargs.get("params") == {"filename": "abc"}


# --- loaded-model resolution ----------------------------------------------


@pytest.mark.asyncio
async def test_get_model_id_prefers_loaded():
    c = make_client()
    c.client.get = AsyncMock(return_value=router_models_resp())
    assert await c.get_model_id() == "qwen.nvfp4"


@pytest.mark.asyncio
async def test_get_model_id_plain_falls_back_to_first():
    c = make_client()
    c.client.get = AsyncMock(return_value=plain_models_resp("m1"))
    assert await c.get_model_id() == "m1"


@pytest.mark.asyncio
async def test_get_loaded_model_returns_loaded():
    c = make_client()
    c.client.get = AsyncMock(return_value=router_models_resp())
    assert await c.get_loaded_model() == "qwen.nvfp4"


@pytest.mark.asyncio
async def test_get_loaded_model_none_on_failure():
    c = make_client()
    c.client.get = AsyncMock(side_effect=RuntimeError("backend down"))
    assert await c.get_loaded_model() is None


@pytest.mark.asyncio
async def test_get_loaded_model_plain_returns_first():
    c = make_client()
    c.client.get = AsyncMock(return_value=plain_models_resp("m1"))
    assert await c.get_loaded_model() == "m1"


# --- router detection ------------------------------------------------------


@pytest.mark.asyncio
async def test_is_router_true_when_status_present():
    c = make_client()
    c.client.get = AsyncMock(return_value=router_models_resp())
    assert await c.is_router() is True


@pytest.mark.asyncio
async def test_is_router_false_when_no_status():
    c = make_client()
    c.client.get = AsyncMock(return_value=plain_models_resp("m1"))
    assert await c.is_router() is False


@pytest.mark.asyncio
async def test_is_router_not_cached_on_failure():
    c = make_client()
    c.client.get = AsyncMock(side_effect=RuntimeError("backend down"))
    assert await c.is_router() is False
    # A transient failure must not permanently mark the backend as plain.
    assert c._is_router is None
    c.client.get = AsyncMock(return_value=router_models_resp())
    assert await c.is_router() is True


# --- health ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_reports_loaded_model():
    c = make_client()
    c.client.get = AsyncMock(return_value=router_models_resp())
    h = await c.health()
    assert h["ok"] is True
    assert h["model_id"] == "qwen.nvfp4"


@pytest.mark.asyncio
async def test_health_plain_reports_first_model():
    c = make_client()
    c.client.get = AsyncMock(return_value=plain_models_resp("m1"))
    h = await c.health()
    assert h["ok"] is True
    assert h["model_id"] == "m1"
