# tests/test_passthrough.py

"""Catch-all pass-through: unhandled paths are forwarded to the first backend
as-is (method, path, query, headers, body); the response is streamed back.

After the /proxy/* rename, the native llama.cpp paths (/slots, /health) are no
longer shadowed by the proxy's own endpoints and reach the backend."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module


class FakeRequest:
    def __init__(self, method, query="", body=b"", headers=None):
        self.method = method
        self.url = SimpleNamespace(query=query)
        self._body = body
        self.headers = headers or [("host", "test")]

    async def body(self):
        return self._body


def _mock_client(chunks, status=200, headers=None):
    """A mock LlamaClient whose httpx client returns a streaming upstream."""
    upstream = MagicMock()
    upstream.status_code = status
    upstream.headers = headers or {"content-type": "application/json"}

    async def aiter_bytes():
        for c in chunks:
            yield c

    upstream.aiter_bytes = aiter_bytes
    upstream.aclose = AsyncMock()

    client = MagicMock()
    built = object()
    client.client.build_request = MagicMock(return_value=built)
    client.client.send = AsyncMock(return_value=upstream)
    return client, built, upstream


def _setup(clients):
    app_module.app.state.clients = clients


async def _passthrough(path, method="GET", query="", body=b"", headers=None):
    req = FakeRequest(method, query, body, headers)
    return await app_module.passthrough(path, req)


@pytest.mark.asyncio
async def test_passthrough_forwards_method_path_query():
    """Method, path and the raw query string reach the backend unchanged."""
    client, built, _ = _mock_client([b'{"ok":true}'])
    _setup([client])

    resp = await _passthrough("metrics", query="x=1")

    args, kwargs = client.client.build_request.call_args
    assert args[0] == "GET"
    assert args[1] == "/metrics"
    assert kwargs["params"] == "x=1"
    assert client.client.send.await_args.args[0] is built
    assert resp.status_code == 200
    body = b"".join([c async for c in resp.body_iterator])
    assert body == b'{"ok":true}'


@pytest.mark.asyncio
async def test_passthrough_forwards_body_and_headers():
    """The raw body and client headers (except host) are forwarded."""
    client, _, _ = _mock_client([b"ok"])
    _setup([client])
    headers = [("host", "test"), ("x-custom", "42")]

    await _passthrough("tokenize", method="POST", body=b'{"content":"hi"}', headers=headers)

    kwargs = client.client.build_request.call_args.kwargs
    assert kwargs["content"] == b'{"content":"hi"}'
    assert kwargs["headers"]["x-custom"] == "42"
    assert "host" not in {k.lower() for k in kwargs["headers"]}


@pytest.mark.asyncio
async def test_passthrough_streams_chunks_and_closes_upstream():
    """Body chunks are yielded in order; the upstream response is closed after."""
    client, _, upstream = _mock_client([b"chunk1", b"chunk2"])
    _setup([client])

    resp = await _passthrough("completion")

    chunks = [c async for c in resp.body_iterator]
    assert chunks == [b"chunk1", b"chunk2"]
    upstream.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_passthrough_backend_error_status():
    """A backend error status (e.g. 404) is passed through, not masked."""
    client, _, _ = _mock_client([b"not found"], status=404)
    _setup([client])

    resp = await _passthrough("nope")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_passthrough_uses_first_backend():
    """Only the first backend is used (consistent with /v1/models)."""
    first, _, _ = _mock_client([b"first"])
    second, _, _ = _mock_client([b"second"])
    _setup([first, second])

    await _passthrough("metrics")

    first.client.build_request.assert_called_once()
    second.client.build_request.assert_not_called()


@pytest.mark.asyncio
async def test_native_backend_paths_reach_backend():
    """/slots and /health are no longer the proxy's own: they reach the backend."""
    client, _, _ = _mock_client([b"[]"])
    _setup([client])

    await _passthrough("slots", query="model=qwen.fast")

    args, kwargs = client.client.build_request.call_args
    assert args[1] == "/slots"
    assert kwargs["params"] == "model=qwen.fast"


def test_proxy_routes_do_not_shadow_native_paths():
    """The route table keeps /slots and /health free for the pass-through."""
    paths = {r.path for r in app_module.app.routes}
    assert "/slots" not in paths
    assert "/health" not in paths
    assert "/proxy/slots" in paths
    assert "/proxy/health" in paths
