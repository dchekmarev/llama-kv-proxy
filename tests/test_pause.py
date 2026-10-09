# tests/test_pause.py

"""Proxy pause (POST /proxy/pause): hold new work, drain committed work, resume.

The invariants these tests guard:
- a pause request closes admission at once, before it activates, so no new work
  slips in while the drain is running;
- activation waits for every unit of committed work to finish (the drain);
- the timer starts at activation, a second request extends it and a manual
  resume (or the timer) reopens admission;
- an idle pause auto-expires without a background task.
"""

import asyncio
import json
import time

import pytest
from fastapi.responses import StreamingResponse

import app as app_module
import chat_flow
from app import routes as routes_mod
from core import pause as pause_mod
from core import promstats
from core.request_id import request_id_var


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class BodyRequest:
    def __init__(self, raw: bytes):
        self._raw = raw

    async def body(self) -> bytes:
        return self._raw


class GatedResp:
    """Fake httpx stream: yields chunks, optionally gated before the first.

    A gate makes the prefill hang so a test controls exactly when the backend
    produces its first byte.
    """

    status_code = 200

    def __init__(self, chunks, gate=None, entered=None):
        self._chunks = list(chunks)
        self._gate = gate
        self._entered = entered
        self.closed = False

    async def aiter_raw(self):
        if self._entered is not None:
            self._entered.set()
        if self._gate is not None:
            await self._gate.wait()
        for c in self._chunks:
            yield c

    async def aclose(self):
        self.closed = True


class ChunkGateResp:
    """Fake httpx stream whose second chunk waits on a gate (first is free)."""

    status_code = 200

    def __init__(self, first, rest, gate, first_event=None):
        self._first = first
        self._rest = list(rest)
        self._gate = gate
        self._first_event = first_event
        self.closed = False

    async def aiter_raw(self):
        yield self._first
        if self._first_event is not None:
            self._first_event.set()
        await self._gate.wait()
        for c in self._rest:
            yield c

    async def aclose(self):
        self.closed = True


async def _consume(resp):
    return b"".join([c async for c in resp.body_iterator])


async def _chat(sm, content, rid, stream=False):
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]
    data = {"messages": [{"role": "user", "content": content}]}
    if stream:
        data["stream"] = True
    token = request_id_var.set(rid)
    try:
        return await app_module.chat(FakeRequest(data))
    finally:
        request_id_var.reset(token)


def _assert_all_free(sm):
    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"


def test_initial_state_is_running():
    snap = pause_mod.snapshot()
    assert snap["paused"] is False
    assert snap["activated"] is False
    assert snap["phase"] == pause_mod.PHASE_OFF
    assert pause_mod.is_paused() is False


async def test_request_without_activity_activates_at_once():
    snap = await pause_mod.request(5, reason="maint")
    assert snap["activated"] is True
    assert snap["paused"] is True
    assert snap["phase"] == pause_mod.PHASE_ACTIVE
    assert snap["reason"] == "maint"
    assert snap["seconds"] == 5
    assert snap["activated_at"] is not None
    assert snap["expires_at"] > snap["activated_at"]
    assert pause_mod.is_paused() is True


async def test_wait_not_paused_returns_when_running():
    await asyncio.wait_for(pause_mod.wait_not_paused(), 1)


async def test_admission_holds_until_resume():
    await pause_mod.request(30)
    waiter = asyncio.create_task(pause_mod.wait_not_paused())
    await asyncio.sleep(0.01)
    assert not waiter.done()
    pause_mod.resume()
    await asyncio.wait_for(waiter, 1)


async def test_request_waits_for_the_drain():
    pause_mod.activity_begin()
    task = asyncio.create_task(pause_mod.request(30))
    await asyncio.sleep(0.01)
    assert not task.done(), "activation must wait for the in-flight work"
    pause_mod.activity_end()
    snap = await asyncio.wait_for(task, 1)
    assert snap["phase"] == pause_mod.PHASE_ACTIVE


async def test_snapshot_reports_the_draining_phase():
    pause_mod.activity_begin()
    task = asyncio.create_task(pause_mod.request(30))
    await asyncio.sleep(0.01)
    snap = pause_mod.snapshot()
    assert snap["paused"] is True, "admission is closed while draining"
    assert snap["activated"] is False
    assert snap["phase"] == pause_mod.PHASE_DRAINING
    pause_mod.activity_end()
    await asyncio.wait_for(task, 1)


async def test_extend_updates_the_deadline():
    first = await pause_mod.request(10)
    second = await pause_mod.request(20)
    assert second["activated_at"] == first["activated_at"]
    assert second["expires_at"] > first["expires_at"]
    assert promstats.counter_sum(promstats.pause_total, action="extend") == 1.0
    assert promstats.counter_sum(promstats.pause_total, action="activate") == 1.0


async def test_auto_expire_releases_the_hold():
    await pause_mod.request(0.05)
    start = time.monotonic()
    await asyncio.wait_for(pause_mod.wait_not_paused(), 2)
    assert time.monotonic() - start >= 0.04
    assert pause_mod.is_paused() is False
    assert promstats.counter_sum(promstats.pause_total, action="expire") == 1.0


async def test_resume_is_idempotent():
    await pause_mod.request(30)
    snap = pause_mod.resume()
    assert snap["phase"] == pause_mod.PHASE_OFF
    assert promstats.counter_sum(promstats.pause_total, action="resume") == 1.0
    snap = pause_mod.resume()
    assert snap["phase"] == pause_mod.PHASE_OFF
    assert promstats.counter_sum(promstats.pause_total, action="resume") == 1.0


async def test_hold_self_releases_at_expiry_after_draining():
    """A waiter that entered during draining must switch to the timer once the
    pause activates, and release when the timer elapses."""
    pause_mod.activity_begin()
    req = asyncio.create_task(pause_mod.request(0.05))
    await asyncio.sleep(0.01)
    waiter = asyncio.create_task(pause_mod.wait_not_paused())
    pause_mod.activity_end()
    await asyncio.wait_for(req, 1)
    assert pause_mod.is_paused() is True
    await asyncio.wait_for(waiter, 2)
    assert pause_mod.is_paused() is False


async def test_hold_rechecks_after_a_re_pause():
    """A waiter woken by a resume must re-hold if a new pause lands first."""
    await pause_mod.request(30)
    waiter = asyncio.create_task(pause_mod.wait_not_paused())
    await asyncio.sleep(0.01)
    pause_mod.resume()
    await pause_mod.request(30)
    await asyncio.sleep(0.05)
    assert not waiter.done()
    pause_mod.resume()
    await asyncio.wait_for(waiter, 1)


async def test_activity_context_manager_drains():
    with pause_mod.activity():
        task = asyncio.create_task(pause_mod.request(30))
        await asyncio.sleep(0.01)
        assert not task.done()
    await asyncio.wait_for(task, 1)


# --- endpoints -----------------------------------------------------------------


async def test_pause_endpoint_round_trip():
    resp = await app_module.pause_request(
        BodyRequest(json.dumps({"seconds": 30, "reason": "maint"}).encode())
    )
    assert resp.status_code == 200
    state = json.loads(resp.body)
    assert state["activated"] is True
    assert state["reason"] == "maint"
    assert (await app_module.pause_state())["paused"] is True
    await app_module.resume_request()
    assert (await app_module.pause_state())["phase"] == "off"


async def test_pause_endpoint_rejects_bad_seconds():
    resp = await app_module.pause_request(
        BodyRequest(json.dumps({"seconds": "soon"}).encode())
    )
    assert resp.status_code == 400
    assert pause_mod.is_paused() is False


# --- pipeline integration ------------------------------------------------------


async def test_chat_is_held_during_a_pause(sm, meta_dir):
    """A request that arrives while paused must not touch the backend until
    the pause ends."""
    started = asyncio.Event()

    async def backend_chat(body, slot_id=None, stream=False):
        started.set()
        return {"choices": []}

    sm.backends[0]["client"].chat_completions = backend_chat
    await pause_mod.request(30)
    task = asyncio.create_task(_chat(sm, "hi", "rid-held"))
    await asyncio.sleep(0.05)
    assert not started.is_set(), "no backend dispatch while paused"
    pause_mod.resume()
    resp = await asyncio.wait_for(task, 2)
    assert resp.status_code == 200
    assert started.is_set()
    _assert_all_free(sm)


async def test_pause_redispatches_an_in_flight_non_stream(sm, meta_dir):
    """A non-stream request interrupted before its response is re-run from
    scratch after the resume, not answered with a stale slot."""
    calls = 0
    first_started = asyncio.Event()

    async def slow_chat(body, slot_id=None, stream=False):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await asyncio.sleep(10)
        return {"choices": []}

    sm.backends[0]["client"].chat_completions = slow_chat
    task = asyncio.create_task(_chat(sm, "hi", "rid-redispatch"))
    await asyncio.wait_for(first_started.wait(), 1)

    # The pause request blocks until the first attempt drains (it is cancelled
    # via race_pause, which releases its slot), then activates.
    await asyncio.wait_for(pause_mod.request(30), 2)
    assert pause_mod.is_paused() is True
    _assert_all_free(sm)
    pause_mod.resume()

    resp = await asyncio.wait_for(task, 2)
    assert resp.status_code == 200
    assert calls == 2
    assert pause_mod.snapshot()["activity"] == 0
    _assert_all_free(sm)


async def test_pause_redispatches_a_stream_in_prefill(sm, meta_dir):
    """A pause that lands before the first backend byte aborts the dispatch
    (still no client-visible commit) and re-runs it: the client never gets
    headers for the dropped attempt and sees no disconnect."""
    gate = asyncio.Event()
    started = asyncio.Event()
    entered = asyncio.Event()
    responses = []

    async def chat(body, slot_id=None, stream=False):
        if not responses:
            started.set()
            resp = GatedResp([b"data: stale\n\n"], gate=gate, entered=entered)
        else:
            resp = GatedResp([b"data: fresh\n\n", b"data: [DONE]\n\n"])
        responses.append(resp)
        return resp

    sm.backends[0]["client"].chat_completions = chat
    task = asyncio.create_task(_chat(sm, "hi", "rid-stream", stream=True))
    await asyncio.wait_for(started.wait(), 1)
    await asyncio.wait_for(entered.wait(), 1)

    # Pause lands while the prefill hangs: race_pause aborts the attempt and
    # releases its slot, so request() drains and activates.
    await asyncio.wait_for(pause_mod.request(30), 2)
    assert task.done() is False, "the stream must not commit while paused"
    _assert_all_free(sm)
    pause_mod.resume()

    resp = await asyncio.wait_for(task, 2)
    assert resp.status_code == 200
    body = await _consume(resp)
    assert b"data: fresh" in body and b"data: stale" not in body
    assert len(responses) == 2
    assert responses[0].closed, "the aborted upstream must be closed"
    assert (
        promstats.counter_sum(
            promstats.requests_total,
            model="m1",
            stream="true",
            outcome="client_disconnect",
        )
        == 0.0
    )
    _assert_all_free(sm)


async def test_pause_after_first_chunk_drains_the_stream(sm, meta_dir):
    """Once the first byte is committed (prefetched) a pause drains the
    stream instead of cutting it: the whole response reaches the client."""
    gate = asyncio.Event()
    first_event = asyncio.Event()

    async def chat(body, slot_id=None, stream=False):
        return ChunkGateResp(
            b"data: one\n\n", [b"data: two\n\n", b"data: [DONE]\n\n"], gate, first_event
        )

    sm.backends[0]["client"].chat_completions = chat
    task = asyncio.create_task(_chat(sm, "hi", "rid-drain", stream=True))
    resp = await asyncio.wait_for(task, 2)
    assert isinstance(resp, StreamingResponse)
    reader = asyncio.create_task(_consume(resp))
    await asyncio.wait_for(first_event.wait(), 1)

    pause_task = asyncio.create_task(pause_mod.request(30))
    await asyncio.sleep(0.05)
    assert not pause_task.done(), "pause must wait for the committed stream"
    gate.set()
    await asyncio.wait_for(pause_task, 2)
    body = await reader
    assert b"data: one" in body and b"data: two" in body
    _assert_all_free(sm)


async def test_kill_during_prefill_answers_499(sm, meta_dir):
    """A kill before the first byte answers a plain 499 JSON, not a stream."""
    gate = asyncio.Event()
    started = asyncio.Event()
    entered = asyncio.Event()
    upstream = GatedResp([b"data: never\n\n"], gate=gate, entered=entered)

    async def chat(body, slot_id=None, stream=False):
        started.set()
        return upstream

    sm.backends[0]["client"].chat_completions = chat
    task = asyncio.create_task(_chat(sm, "hi", "rid-kill-stream", stream=True))
    await asyncio.wait_for(started.wait(), 1)
    # Wait until the prefill is actually reading, so the kill is handled by
    # the prefetch (not by the dispatch race) and closes the upstream.
    await asyncio.wait_for(entered.wait(), 1)
    assert chat_flow.kill_request("rid-kill-stream") is not None

    resp = await asyncio.wait_for(task, 2)
    assert resp.status_code == 499
    assert json.loads(resp.body).get("error")
    assert upstream.closed, "the killed upstream must be closed"
    _assert_all_free(sm)


async def test_kill_wins_over_a_pause_abort():
    """A kill that lands while a pause is unwinding the work is not lost: the
    pipeline must answer 499, not retry under a fresh token."""
    token = chat_flow.bind_kill("rid-race", "m1")

    async def work():
        raise chat_flow.PausedRetry()

    token.kill()
    with pytest.raises(chat_flow.RequestKilled):
        await chat_flow.race_kill(work(), token)


async def test_empty_stream_is_an_empty_200(sm, meta_dir):
    """A backend that closes before any chunk still commits an empty SSE."""
    async def chat(body, slot_id=None, stream=False):
        return GatedResp([])

    sm.backends[0]["client"].chat_completions = chat
    resp = await asyncio.wait_for(_chat(sm, "hi", "rid-empty", stream=True), 2)
    assert resp.status_code == 200
    assert await _consume(resp) == b""
    _assert_all_free(sm)


async def test_metrics_skips_the_backend_while_paused(sm, meta_dir, monkeypatch):
    called = False

    async def fake_collect(clients, model=None):
        nonlocal called
        called = True
        return "backend_metric 1\n"

    monkeypatch.setattr(routes_mod.metrics, "collect", fake_collect)
    await pause_mod.request(30)
    resp = await app_module.metrics_endpoint()
    assert called is False
    assert b"llama_kv_proxy" in resp.body

    pause_mod.resume()
    resp = await app_module.metrics_endpoint()
    assert called is True
    assert b"backend_metric 1" in resp.body
