# tests/test_provider_errors.py

"""P2-3: provider error bodies ({"object": "error", ...}) must be mapped to
HTTP 502 instead of being returned to the client as HTTP 200."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import hashing as hs
import slot_manager as sm_module
from slot_manager import SlotManager


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    client.restore_slot = AsyncMock(return_value=True)
    client.get_model_id_cached = AsyncMock(return_value="m1")
    client.get_loaded_model = AsyncMock(return_value="m1")
    client.chat_completions = AsyncMock(return_value={"choices": []})
    manager.set_clients([client])
    return manager


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "META_DIR", str(tmp_path))
    return tmp_path


async def _chat(sm, content):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    return await app_module.chat(FakeRequest(data))


@pytest.mark.asyncio
async def test_provider_error_body_maps_to_502(sm, meta_dir):
    """A provider error body must become HTTP 502 with the message."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"object": "error", "message": "backend exploded"}
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    body = json.loads(resp.body)
    assert body["error"] == "backend exploded"


@pytest.mark.asyncio
async def test_provider_error_without_message_maps_to_502(sm, meta_dir):
    """An error body without a message still becomes 502."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"object": "error"}
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    assert json.loads(resp.body)["error"]


@pytest.mark.asyncio
async def test_normal_body_stays_200(sm, meta_dir):
    """A normal completion body is still returned as 200."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"object": "chat.completion", "choices": []}
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 200
