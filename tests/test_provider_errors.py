# tests/test_provider_errors.py

"""P2-3: provider error bodies ({"object": "error", ...}) must be mapped to
HTTP 502 instead of being returned to the client as HTTP 200."""

import json
from unittest.mock import AsyncMock

import pytest

import app as app_module


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


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
