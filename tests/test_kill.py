# tests/test_kill.py

"""Operator-initiated cancellation: a request must be killable by its rid both
while it is parked in a slot queue and while it is generating.

The invariants these tests guard are the ones a kill can break:
- the killed request answers 499 and nothing else (no half-open slot, no
  background save of a truncated answer);
- the slot is released exactly once, and the waiter queue stays FIFO;
- a slot whose generation was cut short is no longer allowed to claim it holds
  the saved conversation (else the next same-key request skips its restore and
  continues on a wrong context);
- a stream that was killed says so (an SSE error event), so the client can tell
  a kill from a truncated stream.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow
from core import promstats
from core.request_id import request_id_var
from obs import ui as ui_obs


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class FakeResp:
    """Mimics the httpx.Response parts used by the stream reader."""

    status_code = 200

    def __init__(self, chunks, delay=0.0):
        self._chunks = list(chunks)
        self._delay = delay
        self.closed = False
        self.started = asyncio.Event()

    async def aiter_raw(self):
        for c in self._chunks:
            self.started.set()
            if self._delay:
                await asyncio.sleep(self._delay)
            yield c

    async def aclose(self):
        self.closed = True


async def _chat(sm, content, rid, stream=False):
    """Run one request pipeline under a known correlation id."""
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]
    token = request_id_var.set(rid)
    try:
        return await app_module.chat(
            FakeRequest(
                {
                    "messages": [{"role": "user", "content": content}],
                    "stream": stream,
                }
            )
        )
    finally:
        request_id_var.reset(token)


def _assert_all_free(sm):
    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"


def _body(resp):
    return json.loads(resp.body)


# --- token and registry --------------------------------------------------------


def test_kill_unknown_rid_reports_nothing():
    assert chat_flow.kill_request("nope") is None


def test_bind_then_kill_reports_the_stage():
    token = chat_flow.bind_kill("r1", "m1")
    assert chat_flow.active_kills() == [
        {"rid": "r1", "model": "m1", "stage": chat_flow.STAGE_QUEUED}
    ]
    assert chat_flow.kill_request("r1") == chat_flow.STAGE_QUEUED
    assert token.killed and token.event.is_set()
    assert chat_flow.kill_request("r1") == chat_flow.STAGE_QUEUED, "idempotent"


def test_kill_runs_the_bound_abort_once():
    token = chat_flow.bind_kill("r2", "m1")
    calls: list[int] = []
    token.bind_abort(lambda: calls.append(1))
    chat_flow.kill_request("r2")
    chat_flow.kill_request("r2")
    assert calls == [1]


def test_abort_bound_after_the_kill_runs_at_once():
    """The kill can land between two stages; the abort bound afterwards must
    fire immediately, or the new stage would never learn about it."""
    token = chat_flow.bind_kill("r3", "m1")
    chat_flow.kill_request("r3")
    calls: list[int] = []
    token.bind_abort(lambda: calls.append(1))
    assert calls == [1]


def test_unregister_is_idempotent_and_hides_the_request():
    chat_flow.bind_kill("r4", "m1")
    chat_flow.unregister_kill("r4")
    chat_flow.unregister_kill("r4")
    assert chat_flow.kill_request("r4") is None
    assert chat_flow.active_kills() == []


# --- race_kill -----------------------------------------------------------------


async def test_race_kill_without_a_token_is_a_plain_await():
    async def work():
        return "done"

    assert await chat_flow.race_kill(work(), None) == "done"


async def test_race_kill_returns_the_result_when_the_work_wins():
    token = chat_flow.bind_kill("r5", "m1")

    async def work():
        await asyncio.sleep(0)
        return "done"

    assert await chat_flow.race_kill(work(), token) == "done"
    assert not token.killed
    chat_flow.unregister_kill("r5")


async def test_race_kill_cancels_the_work_and_raises():
    token = chat_flow.bind_kill("r6", "m1")
    unwound = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            unwound.set()
            raise

    task = asyncio.create_task(chat_flow.race_kill(work(), token))
    await asyncio.sleep(0.01)
    chat_flow.kill_request("r6")
    with pytest.raises(chat_flow.RequestKilled):
        await task
    assert unwound.is_set(), "the killed work must be cancelled, not abandoned"
    chat_flow.unregister_kill("r6")


async def test_race_kill_after_the_work_is_a_noop():
    token = chat_flow.bind_kill("r7", "m1")

    async def work():
        return "done"

    assert await chat_flow.race_kill(work(), token) == "done"
    # The request is over: a late kill must not raise into nobody.
    chat_flow.kill_request("r7")
    chat_flow.unregister_kill("r7")


# --- queued request ------------------------------------------------------------


async def test_kill_queued_request_unparks_it_without_leaking(sm, meta_dir):
    """A request parked in the slot queue must be killable: it answers 499,
    leaves the queue, and the slot it never got stays with its holder."""
    holder_started = asyncio.Event()
    release_holder = asyncio.Event()

    async def holding_chat(body, slot_id=None, stream=False):
        holder_started.set()
        await release_holder.wait()
        return {"choices": []}

    sm.backends[0]["client"].chat_completions = holding_chat
    holder = asyncio.create_task(_chat(sm, "holder", "rid-holder"))
    await asyncio.wait_for(holder_started.wait(), 1)
    model = holder_started and "m1"
    assert model == "m1", "the fixture resolves the model to m1"

    queued = asyncio.create_task(_chat(sm, "queued", "rid-queued"))
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert len(sm._waiters.get("m1", ())) == 1, "the second request must be parked"

    assert chat_flow.kill_request("rid-queued") == chat_flow.STAGE_QUEUED
    resp = await asyncio.wait_for(queued, 1)
    assert resp.status_code == 499
    assert _body(resp)["error"] == "killed"
    assert not sm._waiters.get("m1"), "a killed waiter must unregister"
    # The killed request never owned a slot, so nothing of its own leaked; the
    # holder still owns the only one.
    assert sm._locks[(0, "m1", 0)].locked()

    release_holder.set()
    assert (await asyncio.wait_for(holder, 1)).status_code == 200
    _assert_all_free(sm)


async def test_kill_queued_request_keeps_the_queue_fifo(sm, meta_dir):
    """Dropping a waiter must not hand its slot to the corpse nor skip the
    live waiters behind it."""
    holder_started = asyncio.Event()
    release_holder = asyncio.Event()

    async def holding_chat(body, slot_id=None, stream=False):
        holder_started.set()
        await release_holder.wait()
        return {"choices": []}

    sm.backends[0]["client"].chat_completions = holding_chat
    holder = asyncio.create_task(_chat(sm, "holder", "rid-holder"))
    await asyncio.wait_for(holder_started.wait(), 1)

    first = asyncio.create_task(_chat(sm, "first", "rid-first"))
    second = asyncio.create_task(_chat(sm, "second", "rid-second"))
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert len(sm._waiters.get("m1", ())) == 2

    chat_flow.kill_request("rid-first")
    assert (await asyncio.wait_for(first, 1)).status_code == 499
    assert len(sm._waiters.get("m1", ())) == 1, "only the killed one left"

    release_holder.set()
    assert (await asyncio.wait_for(holder, 1)).status_code == 200
    # The survivor gets the freed slot.
    assert (await asyncio.wait_for(second, 2)).status_code == 200
    _assert_all_free(sm)


# --- in-flight non-stream request ---------------------------------------------


async def test_kill_in_flight_non_stream(sm, meta_dir):
    started = asyncio.Event()

    async def hanging_chat(body, slot_id=None, stream=False):
        started.set()
        await asyncio.sleep(10)
        return {"choices": []}

    sm.backends[0]["client"].chat_completions = hanging_chat
    task = asyncio.create_task(_chat(sm, "small", "rid-run"))
    await asyncio.wait_for(started.wait(), 1)

    assert chat_flow.kill_request("rid-run") == chat_flow.STAGE_GENERATING
    resp = await asyncio.wait_for(task, 1)
    assert resp.status_code == 499
    assert _body(resp)["error"] == "killed"
    _assert_all_free(sm)
    assert not chat_flow._BG_SAVE_TASKS, "a killed request must save nothing"
    assert promstats.counter_sum(
        promstats.requests_total, model="m1", stream="false", outcome="killed"
    ) == 1.0
    assert ui_obs.registry.history[0].status == ui_obs.STATUS_CANCELLED


async def test_kill_in_flight_forgets_the_slot_cache_record(sm, meta_dir, monkeypatch):
    """A slot cut mid-generation holds the prefix plus the tokens llama.cpp
    already produced, so it must not keep claiming it holds exactly the saved
    conversation: the next same-key request has to restore again."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    monkeypatch.setattr(
        chat_flow, "_select_restore_candidate", AsyncMock(return_value="k" * 32)
    )
    started = asyncio.Event()

    async def hanging_chat(body, slot_id=None, stream=False):
        started.set()
        await asyncio.sleep(10)
        return {"choices": []}

    sm.backends[0]["client"].chat_completions = hanging_chat
    task = asyncio.create_task(_chat(sm, "small", "rid-run"))
    await asyncio.wait_for(started.wait(), 1)

    g = (0, "m1", 0)
    assert sm._last_saved.get(g) == "k" * 32, "the restore recorded the key"

    chat_flow.kill_request("rid-run")
    assert (await asyncio.wait_for(task, 1)).status_code == 499
    assert g not in sm._last_saved, (
        "the killed slot must lose its saved-key record: a later same-key "
        "request would otherwise skip the restore and run on a wrong context"
    )


async def test_kill_unknown_rid_route_returns_404(sm, meta_dir):
    resp = await app_module.kill_request("never-existed")
    assert resp.status_code == 404
    assert _body(resp) == {"error": "unknown request id"}


async def test_kill_route_kills_an_active_request(sm, meta_dir):
    started = asyncio.Event()

    async def hanging_chat(body, slot_id=None, stream=False):
        started.set()
        await asyncio.sleep(10)
        return {"choices": []}

    sm.backends[0]["client"].chat_completions = hanging_chat
    task = asyncio.create_task(_chat(sm, "small", "rid-route"))
    await asyncio.wait_for(started.wait(), 1)

    listed = await app_module.requests_state()
    assert [r["rid"] for r in listed["requests"]] == ["rid-route"]

    resp = await app_module.kill_request("rid-route")
    assert resp.status_code == 200
    assert _body(resp) == {"rid": "rid-route", "killed": True, "stage": "generating"}
    assert (await asyncio.wait_for(task, 1)).status_code == 499
    assert (await app_module.requests_state())["requests"] == []


# --- in-flight stream ---------------------------------------------------------


async def test_kill_in_flight_stream(sm, meta_dir):
    resp = FakeResp([b"data: a\n\n"] + [b"data: b\n\n"] * 40, delay=0.02)
    sm.backends[0]["client"].chat_completions = AsyncMock(return_value=resp)
    out = await _chat(sm, "small", "rid-stream", stream=True)
    assert out.status_code == 200

    chunks = out.body_iterator
    await anext(chunks)  # the stream is live now
    await asyncio.wait_for(resp.started.wait(), 1)

    assert chat_flow.kill_request("rid-stream") == chat_flow.STAGE_GENERATING
    rest = [c async for c in chunks]

    events = b"".join(rest)
    assert b"killed by operator" in events, "a killed stream must say so to its client"
    assert b"[DONE]" not in events, "a killed stream must not look complete"
    assert resp.closed
    _assert_all_free(sm)
    sm.backends[0]["client"].save_slot.assert_not_awaited()
    assert promstats.counter_sum(
        promstats.requests_total, model="m1", stream="true", outcome="killed"
    ) == 1.0
    assert ui_obs.registry.history[0].status == ui_obs.STATUS_CANCELLED


async def test_kill_before_the_reader_exists_closes_the_response(sm, meta_dir):
    """A kill that lands while the backend response is being opened must not
    leave a reader task that never ran (its finally owns the slot release)."""
    resp = FakeResp([b"data: a\n\n"])
    token = chat_flow.bind_kill("rid-early", "m1")
    chat_flow.kill_request("rid-early")
    with pytest.raises(chat_flow.RequestKilled):
        await chat_flow.start_stream_task(
            resp,
            (0, "m1", 0),
            "k" * 16,
            "prefix",
            ["b"],
            "m1",
            sm,
            is_big=False,
            rid="rid-early",
            kill_token=token,
        )
    assert resp.closed
    assert not chat_flow._READER_TASKS
