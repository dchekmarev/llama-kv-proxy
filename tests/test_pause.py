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

import app as app_module
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


async def _chat(sm, content, rid):
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]
    token = request_id_var.set(rid)
    try:
        return await app_module.chat(
            FakeRequest({"messages": [{"role": "user", "content": content}]})
        )
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
