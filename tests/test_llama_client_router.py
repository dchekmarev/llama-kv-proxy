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

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llama_client import RESTORE_MISSING, LlamaClient


def make_client():
    c = LlamaClient("http://be")
    c.client = MagicMock()
    return c


def resp(status_code, payload):
    r = MagicMock()
    r.status_code = status_code
    r.json = MagicMock(return_value=payload)
    r.text = json.dumps(payload) if isinstance(payload, (dict, list)) else (payload or "")
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
async def test_save_slot_500_is_false():
    """A 500 (slot not ready / nothing to save) is a normal outcome: False."""
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(500, {}))
    assert await c.save_slot(0, "abc") is False


@pytest.mark.asyncio
async def test_save_slot_other_error_is_false_not_raise():
    """Any other non-2xx is a failed save: False, never raised (mirrors
    restore_slot), so callers can rely on a plain bool."""
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(502, {}))
    assert await c.save_slot(0, "abc") is False


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


@pytest.mark.asyncio
async def test_restore_slot_404_reports_missing():
    """A 404 means the cache file does not exist: report RESTORE_MISSING so the
    caller can drop the stale meta (M7)."""
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(404, {}))
    assert await c.restore_slot(0, "abc") == RESTORE_MISSING


@pytest.mark.asyncio
async def test_restore_slot_other_error_is_false():
    """A non-404 failure (e.g. 500) is a transient/other error: False, so the
    caller keeps the meta."""
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(500, {}))
    assert await c.restore_slot(0, "abc") is False


@pytest.mark.asyncio
async def test_erase_slot_sends_erase_action():
    """erase_slot clears a slot's KV via action=erase, routing by model."""
    c = make_client()
    c.client.post = AsyncMock(return_value=resp(200, {}))
    assert await c.erase_slot(3, model="m1") is True
    assert c.client.post.call_args.args[0] == "/slots/3"
    assert c.client.post.call_args.kwargs.get("params") == {"action": "erase"}
    assert c.client.post.call_args.kwargs.get("json") == {"model": "m1"}


@pytest.mark.asyncio
async def test_erase_slot_failure_does_not_raise():
    """A failing erase is best-effort: returns False, never raises."""
    c = make_client()
    c.client.post = AsyncMock(side_effect=Exception("boom"))
    assert await c.erase_slot(3) is False


# --- chat_completions non-stream error body ---------------------------------


@pytest.mark.asyncio
async def test_non_stream_http_error_surfaces_backend_body():
    """L2: a 4xx/5xx must not raise; the backend's body is surfaced so the
    caller sees the real error, not just the status code."""
    c = make_client()
    r = resp(500, {"error": "context length exceeded"})
    r.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=r)
    )
    c.client.post = AsyncMock(return_value=r)

    out = await c.chat_completions({"messages": []}, slot_id=0, stream=False)

    assert out["object"] == "error"
    assert "500" in out["message"]
    assert "context length exceeded" in out["raw"]
    assert out["status"] == 500


@pytest.mark.asyncio
async def test_non_stream_http_error_has_structured_status():
    """M-8: the error body must carry the backend status as a structured field
    so the caller can map 4xx vs 5xx without parsing the message text."""
    c = make_client()
    r = resp(400, {"error": "context length exceeded"})
    r.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=r)
    )
    c.client.post = AsyncMock(return_value=r)

    out = await c.chat_completions({"messages": []}, slot_id=0, stream=False)

    assert out["object"] == "error"
    assert out["status"] == 400


@pytest.mark.asyncio
async def test_non_stream_non_json_has_status_502():
    """M-8: a non-JSON body is a genuine upstream failure: status 502."""
    c = make_client()
    r = MagicMock()
    r.status_code = 200
    r.headers = {"content-type": "text/html"}
    r.text = "<html>oops</html>"
    r.raise_for_status = MagicMock()
    c.client.post = AsyncMock(return_value=r)

    out = await c.chat_completions({"messages": []}, slot_id=0, stream=False)

    assert out["object"] == "error"
    assert out["status"] == 502
    assert out["raw"] == "<html>oops</html>"


@pytest.mark.asyncio
async def test_non_stream_invalid_json_has_status_502():
    """M-8: an unparseable JSON body is a genuine upstream failure: 502."""
    c = make_client()
    r = MagicMock()
    r.status_code = 200
    r.headers = {"content-type": "application/json"}
    r.text = "{not json"
    r.json = MagicMock(side_effect=ValueError("bad json"))
    r.raise_for_status = MagicMock()
    c.client.post = AsyncMock(return_value=r)

    out = await c.chat_completions({"messages": []}, slot_id=0, stream=False)

    assert out["object"] == "error"
    assert out["status"] == 502


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
