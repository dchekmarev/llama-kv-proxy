# tests/test_backend_retry.py

"""Proxy-side retry of a retryable backend failure before the response commits.

A stale pooled connection ("Server disconnected without sending a response"), a
backend restart or a transient upstream 5xx must be re-dispatched by the proxy
instead of bouncing the client with 502 + Retry-After. These tests pin:

- the retry happens and succeeds transparently (transport, provider 5xx, stream
  upstream 5xx), with the client seeing a single clean response;
- the retry budget is bounded by BACKEND_RETRY_ATTEMPTS, after which the
  original 502 + Retry-After contract resumes;
- a 4xx client fault is never retried (retrying would repeat the same fault);
- a mid-stream failure is never retried (the response is already committed);
- a retry erases the served slot's partial KV and is counted in metrics.
"""

import json
from unittest.mock import AsyncMock

import httpx
import pytest

import app as app_module
from core import config, promstats


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class FakeStreamResp:
    """A 200 upstream stream: ``aiter_raw`` chunks plus ``aclose``."""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    async def aiter_raw(self):
        for c in self._chunks:
            yield c

    async def aclose(self):
        self.closed = True


class MidStreamFailResp:
    """A 200 upstream whose stream dies after the first chunk.

    The first chunk is what commits the response, so the reader (not this
    module) sees the failure: it must not be retried.
    """

    status_code = 200

    def __init__(self, first, exc):
        self._first = first
        self._exc = exc
        self.closed = False

    async def aiter_raw(self):
        yield self._first
        raise self._exc

    async def aclose(self):
        self.closed = True


class ErrorStreamResp:
    """A non-200 upstream stream: status + body read before close."""

    def __init__(self, status, body=b""):
        self.status_code = status
        self._body = body
        self.closed = False

    async def aread(self):
        return self._body

    async def aclose(self):
        self.closed = True


OK_BODY = {"object": "chat.completion", "choices": [{"message": {"content": "hi"}}]}
OK_CHUNKS = [
    b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n',
    b"data: [DONE]\n\n",
]


async def _chat(sm, content, stream=False):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {"messages": [{"role": "user", "content": content}], "stream": stream}
    return await app_module.chat(FakeRequest(data))


async def _drain(resp):
    return b"".join([c async for c in resp.body_iterator])


@pytest.mark.asyncio
async def test_transport_error_is_retried_then_succeeds(sm, meta_dir):
    """A stale connection is transparent: the retry re-dispatches and wins."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        side_effect=[httpx.RemoteProtocolError("Server disconnected"), OK_BODY]
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 200
    assert json.loads(resp.body)["choices"][0]["message"]["content"] == "hi"
    assert client.chat_completions.await_count == 2
    # The failed attempt had reached the backend: its partial KV is erased.
    client.erase_slot.assert_awaited()


@pytest.mark.asyncio
async def test_transport_error_retries_are_bounded(sm, meta_dir):
    """When every attempt fails, the 502 + Retry-After contract resumes."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        side_effect=httpx.ConnectError("connection refused")
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    assert resp.headers["Retry-After"] == str(config.RETRY_AFTER_S)
    assert client.chat_completions.await_count == config.BACKEND_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_connect_error_is_retryable(sm, meta_dir):
    """A refused connect never reached the backend: it is retried."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        side_effect=[httpx.ConnectError("refused"), OK_BODY]
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 200
    assert client.chat_completions.await_count == 2


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("no answer"),
        httpx.WriteTimeout("upstream stuck"),
        httpx.ReadError("backend reset mid-body"),
        httpx.DecodingError("bad chunked body"),
    ],
)
@pytest.mark.asyncio
async def test_ambiguous_transport_failure_is_not_retried(sm, meta_dir, exc):
    """After the request was sent, the backend may have generated: no retry.

    A read/write timeout or a connection drop is ambiguous -- the backend may
    be mid/generating -- so it is answered 502 + Retry-After at once instead of
    recomputing the answer. Only a failure provably before the backend acted
    (connect/pool, stale pooled connection) is re-dispatched.
    """
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(side_effect=exc)

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    assert resp.headers["Retry-After"] == str(config.RETRY_AFTER_S)
    assert client.chat_completions.await_count == 1


@pytest.mark.asyncio
async def test_provider_5xx_is_retried_then_succeeds(sm, meta_dir):
    """A transient provider 5xx is retried before it is surfaced as 502."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        side_effect=[
            {"object": "error", "message": "boom", "status": 500},
            OK_BODY,
        ]
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 200
    assert client.chat_completions.await_count == 2


@pytest.mark.asyncio
async def test_client_fault_is_not_retried(sm, meta_dir):
    """A 4xx client fault is surfaced at once: a retry repeats the fault."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        return_value={"object": "error", "message": "bad", "status": 400}
    )

    resp = await _chat(sm, "small request")

    assert resp.status_code == 400
    assert client.chat_completions.await_count == 1


@pytest.mark.asyncio
async def test_stream_upstream_5xx_is_retried_then_succeeds(sm, meta_dir):
    """The stream path retries a 5xx answered before any byte was committed."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        side_effect=[ErrorStreamResp(500, b"boom"), FakeStreamResp(OK_CHUNKS)]
    )

    resp = await _chat(sm, "small request", stream=True)

    assert resp.status_code == 200
    body = await _drain(resp)
    assert b'"hi"' in body
    assert client.chat_completions.await_count == 2


@pytest.mark.asyncio
async def test_mid_stream_failure_is_not_retried(sm, meta_dir):
    """A failure after the first committed byte is a truncation, not a retry."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        side_effect=[
            MidStreamFailResp(OK_CHUNKS[0], httpx.ReadError("backend reset"))
        ]
    )

    resp = await _chat(sm, "small request", stream=True)

    assert resp.status_code == 200
    body = await _drain(resp)
    assert b'"error"' in body
    assert client.chat_completions.await_count == 1


@pytest.mark.asyncio
async def test_retry_is_counted(sm, meta_dir):
    """A proxy-issued retry is counted, by reason, for observability."""
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(
        side_effect=[httpx.ConnectError("connection refused"), OK_BODY]
    )

    await _chat(sm, "small request")

    counted = promstats.counter_sum(promstats.backend_retries_total, reason="transport")
    assert counted == 1
