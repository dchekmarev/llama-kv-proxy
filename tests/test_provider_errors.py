# tests/test_provider_errors.py

"""P2-3: provider error bodies ({"object": "error", ...}) must be mapped to
HTTP 502 instead of being returned to the client as HTTP 200.

M-8: backend 4xx must surface as the same 4xx on BOTH the stream and
non-stream paths (client fault stays 4xx); 5xx / connect errors / non-JSON
bodies are genuine upstream failures and map to 502 on both paths."""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

import app as app_module


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class FakeStreamResp:
    """Mimics the httpx.Response parts used by the stream dispatch."""

    def __init__(self, status, body=b""):
        self.status_code = status
        self._body = body

    async def aread(self):
        return self._body

    async def aclose(self):
        pass


async def _chat(sm, content, stream=False):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
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


# --- M-8: consistent 4xx/5xx mapping across stream and non-stream ----------


@pytest.mark.asyncio
async def test_non_stream_error_4xx_status_passthrough(sm, meta_dir):
    """A backend 4xx (client fault) must stay 4xx on the non-stream path."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={
            "object": "error",
            "message": "provider returned HTTP 400: context length exceeded",
            "status": 400,
            "raw": "context length exceeded",
        }
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"] == "provider returned HTTP 400: context length exceeded"
    assert body["raw"] == "context length exceeded"


@pytest.mark.asyncio
async def test_non_stream_error_429_status_passthrough(sm, meta_dir):
    """A backend 429 must stay 429 on the non-stream path."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"object": "error", "message": "rate limited", "status": 429}
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_non_stream_error_5xx_status_maps_to_502(sm, meta_dir):
    """A backend 5xx is a genuine upstream failure: 502 on the non-stream path."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"object": "error", "message": "backend exploded", "status": 500}
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    assert json.loads(resp.body)["error"] == "backend exploded"


@pytest.mark.asyncio
async def test_stream_error_4xx_status_passthrough(sm, meta_dir):
    """A backend 4xx must stay 4xx on the stream path (already verbatim, must
    stay consistent with the non-stream path)."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value=FakeStreamResp(400, b'{"error": "context length exceeded"}')
    )

    resp = await _chat(sm, "small request", stream=True)

    assert resp.status_code == 400
    assert json.loads(resp.body)["error"] == '{"error": "context length exceeded"}'


@pytest.mark.asyncio
async def test_stream_error_5xx_status_maps_to_502(sm, meta_dir):
    """A backend 5xx is a genuine upstream failure: 502 on the stream path."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value=FakeStreamResp(500, b"backend exploded")
    )

    resp = await _chat(sm, "small request", stream=True)

    assert resp.status_code == 502
    assert json.loads(resp.body)["error"] == "backend exploded"


@pytest.mark.asyncio
async def test_connect_error_maps_to_502(sm, meta_dir):
    """A connect failure (no backend response at all) is an upstream failure:
    502 on the non-stream path."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        side_effect=httpx.ConnectError("connection refused")
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    assert "connection refused" in json.loads(resp.body)["error"]


@pytest.mark.asyncio
async def test_stream_connect_error_maps_to_502(sm, meta_dir):
    """A connect failure is an upstream failure: 502 on the stream path."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        side_effect=httpx.ConnectError("connection refused")
    )

    resp = await _chat(sm, "small request", stream=True)

    assert resp.status_code == 502
    assert "connection refused" in json.loads(resp.body)["error"]
