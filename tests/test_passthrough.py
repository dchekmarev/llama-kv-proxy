# tests/test_passthrough.py

"""Catch-all pass-through: unhandled paths are forwarded to the first backend
as-is (method, path, query, headers, body); the response is streamed back.

After the /proxy/* rename, the native llama.cpp paths (/slots, /health) are no
longer shadowed by the proxy's own endpoints and reach the backend."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import app as app_module
from backend.slot_manager import SlotManager
from core import config


class StarletteLikeHeaders(dict):
    """Mimics Starlette Headers: iteration yields keys, not (key, value) pairs."""

    def __iter__(self):
        return iter(self.keys())


class FakeRequest:
    def __init__(self, method, query="", body=b"", headers=None):
        self.method = method
        self.url = SimpleNamespace(query=query)
        self._body = body
        self.headers = headers or StarletteLikeHeaders({"host": "test"})

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
    client.probe = AsyncMock(return_value=True)
    built = object()
    client.client.build_request = MagicMock(return_value=built)
    client.client.send = AsyncMock(return_value=upstream)
    return client, built, upstream


def _setup(clients):
    # Both are set by the lifespan together: the pass-through picks its client
    # through the health order, which reads the SlotManager's liveness verdicts
    # (a fresh SlotManager never polled = every backend up).
    app_module.app.state.clients = clients
    app_module.app.state.sm = SlotManager()
    app_module.app.state.sm.set_clients(clients)


async def _passthrough(path, method="GET", query="", body=b"", headers=None):
    req = FakeRequest(method, query, body, headers)
    return await app_module.passthrough(path, req)


@pytest.mark.asyncio
async def test_passthrough_forwards_method_path_query():
    """Method, path and the raw query string reach the backend unchanged."""
    client, built, _ = _mock_client([b'{"ok":true}'])
    _setup([client])

    resp = await _passthrough("tokenize", query="x=1")

    args, kwargs = client.client.build_request.call_args
    assert args[0] == "GET"
    assert args[1] == "/tokenize"
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
async def test_passthrough_strips_hop_by_hop_headers():
    """L4: length/hop-by-hop headers are not forwarded — httpx recomputes
    content-length and framing, so a stale client value would conflict."""
    client, _, _ = _mock_client([b"ok"])
    _setup([client])
    headers = [
        ("host", "test"),
        ("content-length", "100"),
        ("transfer-encoding", "chunked"),
        ("connection", "keep-alive"),
        ("content-type", "application/json"),
        ("x-custom", "42"),
    ]

    await _passthrough("tokenize", method="POST", body=b'{"content":"hi"}', headers=headers)

    fwd = {k.lower() for k in client.client.build_request.call_args.kwargs["headers"]}
    assert "content-length" not in fwd
    assert "transfer-encoding" not in fwd
    assert "connection" not in fwd
    assert "host" not in fwd
    assert "content-type" in fwd
    assert "x-custom" in fwd


@pytest.mark.asyncio
async def test_passthrough_starlette_style_headers():
    """Production shape: request.headers is a mapping iterating keys only."""
    client, _, _ = _mock_client([b"ok"])
    _setup([client])
    headers = StarletteLikeHeaders({"host": "test", "x-custom": "7"})

    await _passthrough("tokenize", headers=headers)

    kwargs = client.client.build_request.call_args.kwargs
    assert kwargs["headers"]["x-custom"] == "7"
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

    await _passthrough("tokenize")

    first.client.build_request.assert_called_once()
    second.client.build_request.assert_not_called()


@pytest.mark.asyncio
async def test_passthrough_skips_known_down_backend():
    """Health order: a backend the slot poll marked down is skipped, and when
    every backend is down the first one is used as before (HOLD governs)."""
    first, _, _ = _mock_client([b"first"])
    second, _, _ = _mock_client([b"second"])
    _setup([first, second])
    app_module.app.state.sm.set_backend_up(0, False)

    await _passthrough("tokenize")

    first.client.build_request.assert_not_called()
    second.client.build_request.assert_called_once()

    # All down: fall back to the first backend, never an empty pick.
    app_module.app.state.sm.set_backend_up(1, False)
    await _passthrough("tokenize")

    first.client.build_request.assert_called_once()


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


@pytest.mark.asyncio
async def test_passthrough_drops_upstream_content_encoding():
    """httpx decodes the body in aiter_bytes(), so the upstream
    content-encoding no longer describes what we send. Forwarding it makes
    every client that honours it (i.e. any browser, which always sends
    Accept-Encoding) fail to decode the pass-through body."""
    client, _, _ = _mock_client(
        [b"<html>ok</html>"],
        headers={"content-type": "text/html", "content-encoding": "gzip"},
    )
    _setup([client])

    resp = await _passthrough("")

    assert "content-encoding" not in {k.lower() for k in resp.headers}
    assert resp.headers["content-type"] == "text/html"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timeout"),
    ],
    ids=["connect-error", "connect-timeout"],
)
async def test_passthrough_connect_error_502_retry_after(exc):
    """A dead backend must not surface as an unhandled 500 from the
    pass-through: it is the same upstream failure as on the chat path,
    so 502 + Retry-After (timeouts included: ConnectTimeout is an
    httpx.HTTPError)."""
    client = MagicMock()
    client.probe = AsyncMock(return_value=True)
    client.client.build_request = MagicMock(return_value=object())
    client.client.send = AsyncMock(side_effect=exc)
    _setup([client])

    resp = await _passthrough("slots")

    assert resp.status_code == 502
    assert resp.headers["Retry-After"] == str(config.RETRY_AFTER_S)
    assert str(exc) in json.loads(resp.body)["error"]


@pytest.mark.asyncio
async def test_passthrough_backend_5xx_mirrored_without_retry_after():
    """Once the upstream response is open, its status is mirrored verbatim
    (README contract): a backend 5xx reaches the client as-is, without a
    proxy-generated Retry-After."""
    client, _, _ = _mock_client([b"boom"], status=503)
    _setup([client])

    resp = await _passthrough("completion")

    assert resp.status_code == 503
    assert "Retry-After" not in resp.headers


@pytest.mark.asyncio
async def test_passthrough_mid_stream_error_ends_cleanly():
    """A transport failure after the headers were sent ends the streamed
    response instead of escaping the generator as an unhandled traceback."""
    client, _, upstream = _mock_client([])

    async def _broken_aiter():
        yield b"chunk1"
        raise httpx.ReadError("connection reset")

    upstream.aiter_bytes = _broken_aiter
    _setup([client])

    resp = await _passthrough("completion")

    chunks = [c async for c in resp.body_iterator]
    assert chunks == [b"chunk1"], "the chunks before the failure must be delivered"
    upstream.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_passthrough_holds_while_backend_down_then_streams(monkeypatch):
    """Same hold as the chat path: a down backend is waited out before the
    upstream is opened, so the client sees a long TTFT, not a 502."""
    import chat_flow

    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 5.0)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.01)
    client, _, _ = _mock_client([b"ok"])
    client.probe = AsyncMock(side_effect=[False, True])
    _setup([client])

    resp = await _passthrough("completion")

    assert resp.status_code == 200
    assert client.probe.call_count == 2
    assert b"".join([c async for c in resp.body_iterator]) == b"ok"


@pytest.mark.asyncio
async def test_passthrough_hold_exhaustion_502_retry_after(monkeypatch):
    """When the backend does not come back within HOLD_TIMEOUT, the
    pass-through answers the chat path's contract: 502 + Retry-After."""
    import chat_flow

    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 0.05)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.01)
    client, _, _ = _mock_client([b"ok"])
    client.probe = AsyncMock(return_value=False)
    _setup([client])

    resp = await _passthrough("completion")

    assert resp.status_code == 502
    assert resp.headers["Retry-After"] == str(config.RETRY_AFTER_S)
