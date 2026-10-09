# tests/test_hold.py

"""Hybrid hold: while the picked backend is down or unhealthy (5xx on
/health, e.g. model loading), the proxy holds the connection for up to
HOLD_TIMEOUT seconds instead of bouncing the client with 502 + Retry-After
right away. On exhaustion the usual 502 + Retry-After is answered; a kill
during the hold answers 499 immediately.

The slot stays held for the whole wait: while the backend is down its slots
are useless anyway, and releasing would only move the wait into the acquire
queue."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.responses import StreamingResponse

import app as app_module
import chat_flow
from core import config, promstats
from core.request_id import request_id_var
from obs import ui as ui_obs


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class FakeResp:
    """Mimics the httpx.Response parts used by the stream dispatch."""

    status_code = 200

    def __init__(self, chunks, delay=0.0):
        self._chunks = list(chunks)
        self._delay = delay
        self.closed = False

    async def aiter_raw(self):
        for c in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield c

    async def aclose(self):
        self.closed = True


async def _chat(sm, content, rid=None, stream=False):
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if rid is None:
        return await app_module.chat(FakeRequest(data))
    token = request_id_var.set(rid)
    try:
        return await app_module.chat(FakeRequest(data))
    finally:
        request_id_var.reset(token)


def _assert_all_free(sm):
    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"


# --- LlamaClient.probe ---------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_classification():
    """Serving = any non-5xx answer on /health (llama.cpp answers 503 while a
    model is loading); a transport failure means down."""
    from backend.llama_client import LlamaClient

    client = LlamaClient("http://127.0.0.1:9")
    try:
        for status, expected in [(200, True), (404, True), (503, False), (500, False)]:
            client.client.get = AsyncMock(
                return_value=MagicMock(status_code=status)
            )
            assert await client.probe() is expected

        # The probe must bound both phases: the connect like every other
        # call, the read by MODEL_ID_TIMEOUT (a wedged backend must not stall
        # one probe for REQUEST_TIMEOUT).
        kwargs = client.client.get.call_args.kwargs
        assert kwargs["timeout"].connect == config.CONNECT_TIMEOUT
        assert kwargs["timeout"].read == config.MODEL_ID_TIMEOUT

        client.client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        assert await client.probe() is False
        client.client.get = AsyncMock(side_effect=httpx.ReadTimeout("no answer"))
        assert await client.probe() is False
        # Soft read: a non-httpx failure (closed client, shutdown race) also
        # means "not serving", never an exception.
        client.client.get = AsyncMock(side_effect=RuntimeError("closed"))
        assert await client.probe() is False
    finally:
        await client.close()


# --- chat path ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_hold_when_backend_healthy(sm, meta_dir, monkeypatch):
    """A healthy backend (first probe True) dispatches immediately: exactly
    one probe, no waiting."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 5.0)
    client = sm.backends[0]["client"]
    client.probe = AsyncMock(return_value=True)

    resp = await _chat(sm, "small request")

    assert resp.status_code == 200
    assert client.probe.call_count == 1


@pytest.mark.asyncio
async def test_chat_holds_while_backend_down_then_succeeds(sm, meta_dir, monkeypatch):
    """A down backend is waited out: the request succeeds once the probe turns
    healthy, and the client never sees a 502."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 5.0)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.01)
    client = sm.backends[0]["client"]
    client.probe = AsyncMock(side_effect=[False, False, True])

    resp = await _chat(sm, "small request")

    assert resp.status_code == 200
    assert client.probe.call_count == 3
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_stream_holds_while_backend_down_then_streams(sm, meta_dir, monkeypatch):
    """The hold applies to streams too, before any response byte is sent."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 5.0)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.01)
    client = sm.backends[0]["client"]
    client.probe = AsyncMock(side_effect=[False, True])
    client.chat_completions = AsyncMock(
        return_value=FakeResp([b'data: {"choices": []}\n\n', b"data: [DONE]\n\n"])
    )

    resp = await _chat(sm, "small request", stream=True)

    assert isinstance(resp, StreamingResponse)
    assert client.probe.call_count == 2
    body = b"".join([c async for c in resp.body_iterator])
    assert b"data: [DONE]" in body
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_hold_exhaustion_502_retry_after(sm, meta_dir, monkeypatch):
    """When the backend does not come back within HOLD_TIMEOUT, the client
    gets the usual upstream-failure contract: 502 + Retry-After."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 0.05)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.01)
    client = sm.backends[0]["client"]
    client.probe = AsyncMock(return_value=False)

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    assert resp.headers["Retry-After"] == str(config.RETRY_AFTER_S)
    assert client.probe.call_count >= 2
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_hold_timeout_zero_is_one_probe_then_502(sm, meta_dir, monkeypatch):
    """HOLD_TIMEOUT=0 disables the wait: exactly one probe, then the usual
    502 + Retry-After (no sleep in between)."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 0.0)
    client = sm.backends[0]["client"]
    client.probe = AsyncMock(return_value=False)

    resp = await _chat(sm, "small request")

    assert resp.status_code == 502
    assert resp.headers["Retry-After"] == str(config.RETRY_AFTER_S)
    assert client.probe.call_count == 1
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_hold_exhaustion_counts_once_and_reports_ui(sm, meta_dir, monkeypatch):
    """The hold_timeout outcome is counted exactly once (not also as a
    generic error) and lands in the UI history as an error."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 0.05)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.01)
    client = sm.backends[0]["client"]
    client.probe = AsyncMock(return_value=False)

    resp = await _chat(sm, "small request", rid="hold-metric")

    assert resp.status_code == 502
    assert promstats.counter_sum(
        promstats.requests_total, model="m1", stream="false", outcome="hold_timeout"
    ) == 1.0
    assert promstats.counter_sum(
        promstats.requests_total, model="m1", stream="false", outcome="error"
    ) == 0.0
    assert ui_obs.registry.history[0].status == ui_obs.STATUS_ERROR
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_kill_during_hold_sleep_answers_499(sm, meta_dir, monkeypatch):
    """A kill landing in the inter-probe sleep (not in a probe) also unwinds
    to 499 immediately."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 30.0)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.5)
    client = sm.backends[0]["client"]
    first_probe_done = asyncio.Event()

    async def _probe():
        first_probe_done.set()
        return False

    client.probe = _probe

    task = asyncio.create_task(_chat(sm, "small request", rid="hold-kill-sleep"))
    await asyncio.wait_for(first_probe_done.wait(), timeout=5.0)
    await asyncio.sleep(0.1)  # the gate is now in its inter-probe sleep
    assert chat_flow.kill_request("hold-kill-sleep") is not None
    resp = await asyncio.wait_for(task, timeout=5.0)

    assert resp.status_code == 499
    assert json.loads(resp.body).get("error")
    # The kill landed before dispatch: nothing was generated, so the (down)
    # backend must not be touched at all -- an erase would block the 499 up
    # to CONNECT_TIMEOUT.
    client.erase_slot.assert_not_awaited()
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_kill_during_hold_answers_499(sm, meta_dir, monkeypatch):
    """A kill must not wait out the hold: 499 immediately, slot released."""
    monkeypatch.setattr(chat_flow, "HOLD_TIMEOUT", 30.0)
    monkeypatch.setattr(chat_flow, "HOLD_PROBE_INTERVAL_S", 0.05)
    client = sm.backends[0]["client"]
    entered = asyncio.Event()

    async def _probe():
        entered.set()
        return False

    client.probe = _probe

    task = asyncio.create_task(_chat(sm, "small request", rid="hold-kill"))
    await asyncio.wait_for(entered.wait(), timeout=5.0)
    assert chat_flow.kill_request("hold-kill") is not None
    resp = await asyncio.wait_for(task, timeout=5.0)

    assert resp.status_code == 499
    assert json.loads(resp.body).get("error")
    client.erase_slot.assert_not_awaited()
    _assert_all_free(sm)
