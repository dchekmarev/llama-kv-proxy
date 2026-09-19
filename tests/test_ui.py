# tests/test_ui.py

"""The /proxy/ui/ live dashboard: registry lifecycle, hook gating, the SSE
broadcaster, and the four routes (page, state, request, events)."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow
import config
import ui as ui_obs
from request_id import request_id_var
from ui import Registry, format_prompt, format_prompt_tail


# --- format_prompt -----------------------------------------------------------


def test_format_prompt_roles_and_content():
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert format_prompt(msgs) == "[system] be brief\n\n[user] hi"


def test_format_prompt_edge_cases():
    assert format_prompt(None) == ""
    assert format_prompt([]) == ""
    # Non-dict entries are skipped; missing role/content tolerated.
    assert format_prompt([42, {"content": "x"}, {"role": "user"}]) == (
        "[?] x\n\n[user] "
    )
    # Non-string content is JSON-encoded.
    assert format_prompt([{"role": "user", "content": [1, 2]}]) == (
        '[user] [1, 2]'
    )


def test_format_prompt_tail_keeps_newest_messages():
    msgs = [
        {"role": "system", "content": "S" * 50},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "latest question"},
    ]
    # A transcript that fits is returned whole (same as format_prompt).
    assert format_prompt_tail(msgs, 10_000) == format_prompt(msgs)
    # Over the limit: only the newest whole messages that fit, in order.
    tail = format_prompt_tail(msgs, 30)
    assert tail == "[user] latest question"
    # A middle budget keeps the last two turns but drops the head.
    tail2 = format_prompt_tail(msgs, 60)
    assert tail2 == "[assistant] old answer\n\n[user] latest question"
    # The newest message is always kept, even alone over the limit.
    big = [{"role": "system", "content": "x"}, {"role": "user", "content": "y" * 100}]
    assert format_prompt_tail(big, 10) == "[user] " + "y" * 100
    # Empty input.
    assert format_prompt_tail(None, 10) == ""
    assert format_prompt_tail([], 10) == ""


# --- registry lifecycle ------------------------------------------------------


def _start(reg: Registry, rid: str = "r1") -> None:
    reg.start(
        rid,
        model="m1",
        stream=True,
        n_words=10,
        is_big=True,
        key="k" * 40,
        messages=[{"role": "user", "content": "hello"}],
    )


def test_registry_lifecycle():
    reg = Registry()
    _start(reg)
    info = reg.active["r1"]
    assert info.status == ui_obs.STATUS_QUEUED
    assert info.key == "k" * 16, "key is truncated to 16 chars"
    assert info.prompt_preview == "[user] hello"
    assert reg.full_prompt("r1") == "[user] hello"

    reg.slot("r1", 0, "m1", 3)
    assert info.status == ui_obs.STATUS_GENERATING
    assert info.slot == {"backend": 0, "model": "m1", "id": 3}

    reg.ttft("r1", 0.5)
    reg.ttft("r1", 0.9)  # first value wins
    assert info.ttft == 0.5

    reg.tokens("r1", "he", "")
    reg.tokens("r1", "llo", "why")
    assert info.n_chars == 5
    assert info.tail == "hello"
    assert info.tail_reason == "why"

    reg.usage("r1", {"prompt_tokens": 2})
    reg.usage("r1", {"prompt_tokens": 99})  # first value wins
    assert info.usage == {"prompt_tokens": 2}

    reg.end("r1", status=ui_obs.STATUS_DONE)
    assert "r1" not in reg.active
    assert reg.full_prompt("r1") == "[user] hello", "full prompt kept in history"
    assert len(reg.history) == 1
    done = reg.history[0]
    assert done.status == ui_obs.STATUS_DONE
    assert done.ended_at is not None

    # End is idempotent: a second call is a no-op.
    reg.end("r1", status=ui_obs.STATUS_ERROR)
    assert len(reg.history) == 1
    assert reg.history[0].status == ui_obs.STATUS_DONE

    snap = reg.snapshot()
    assert snap["active"] == []
    assert snap["history"][0]["rid"] == "r1"
    assert "tail" not in snap["history"][0], "history rows carry no tail"


def test_registry_tail_capped():
    reg = Registry(tail_max=10)
    _start(reg)
    reg.tokens("r1", "a" * 50, "")
    info = reg.active["r1"]
    assert info.n_chars == 50
    assert info.tail == "a" * 10


def test_registry_history_eviction():
    reg = Registry(history_max=2)
    for i in range(3):
        rid = f"r{i}"
        _start(reg, rid)
        reg.end(rid)
    assert [i.rid for i in reg.history] == ["r2", "r1"], "oldest evicted"


def test_registry_response_tail():
    reg = Registry()
    _start(reg)
    reg.tokens("r1", "abc", "why")
    assert reg.response_tail("r1") == ("abc", "why")
    reg.end("r1")
    assert reg.response_tail("r1") == ("abc", "why"), "kept in history"
    assert reg.response_tail("nope") is None


def test_registry_unknown_rid_noop():
    reg = Registry()
    reg.slot("nope", 0, "m1", 1)
    reg.tokens("nope", "x", "")
    reg.ttft("nope", 0.1)
    reg.usage("nope", {})
    reg.end("nope")
    assert reg.active == {} and len(reg.history) == 0


# --- hook gating -------------------------------------------------------------


def test_hooks_noop_when_ui_disabled(monkeypatch):
    monkeypatch.setattr(config, "UI_ENABLED", False)
    ui_obs.req_start(
        "r1", model="m1", stream=True, n_words=1, is_big=False, key="k",
        messages=[{"role": "user", "content": "x"}],
    )
    ui_obs.req_tokens("r1", "abc", "")
    ui_obs.req_end("r1")
    assert ui_obs.registry.active == {}
    assert len(ui_obs.registry.history) == 0


def test_hooks_never_raise(monkeypatch):
    # A registry that blows up must not propagate into the request path.
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ui_obs.registry, "start", boom)
    ui_obs.req_start(
        "r1", model="m1", stream=True, n_words=1, is_big=False, key="k",
        messages=None,
    )


# --- broadcaster -------------------------------------------------------------


@pytest.mark.asyncio
async def test_broadcaster_pushes_token_batches():
    reg = Registry()
    q = reg.subscribe()
    try:
        _start(reg)
        reg.tokens("r1", "hello", "think")
        # The start event is pushed synchronously; the token batch arrives on
        # the broadcaster's 200 ms cycle — collect until it shows up.
        events: list[dict] = []
        while not any(e["type"] == "tokens" for e in events):
            events.extend(await asyncio.wait_for(q.get(), timeout=2))
        assert "start" in [e["type"] for e in events]
        tok = next(e for e in events if e["type"] == "tokens")
        assert tok["content"] == "hello"
        assert tok["reasoning"] == "think"
        # Pending buffers are drained: a second cycle has nothing to push.
        assert reg.active["r1"].pending == ""
    finally:
        reg.unsubscribe(q)
    assert reg._broadcaster is None, "last unsubscribe cancels the broadcaster"


@pytest.mark.asyncio
async def test_broadcaster_survives_slow_viewer():
    reg = Registry()
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    reg._subs.add(q)
    reg._ensure_broadcaster()
    try:
        _start(reg)
        for _ in range(5):
            reg.tokens("r1", "x" * 2000, "")
            await asyncio.sleep(0.25)
        # The slow viewer's queue must be bounded, never grown unbounded.
        assert q.qsize() <= 1
    finally:
        reg._subs.discard(q)
        reg._broadcaster.cancel()
        reg._broadcaster = None


# --- routes ------------------------------------------------------------------


@pytest.fixture()
def ui_app(sm):
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]
    return app_module.app


@pytest.mark.asyncio
async def test_ui_page_route(ui_app):
    resp = await app_module.ui_page_route()
    assert resp.status_code == 200
    assert "text/html" in resp.media_type
    assert "proxy/ui/events" in resp.body.decode()


@pytest.mark.asyncio
async def test_ui_state_route(ui_app, sm):
    sm.set_backend_slots(
        0, "m1", [{"id": 0, "state": "busy"}, {"id": 1, "state": "free"}]
    )
    ui_obs.registry.start(
        "r1", model="m1", stream=True, n_words=5, is_big=False, key="k" * 20,
        messages=[{"role": "user", "content": "hi"}],
    )
    ui_obs.registry.slot("r1", 0, "m1", 0)
    resp = await app_module.ui_state()
    body = json.loads(resp.body)
    assert body["active"][0]["rid"] == "r1"
    assert body["active"][0]["slot"] == {"backend": 0, "model": "m1", "id": 0}
    assert isinstance(body["history"], list)
    # Slot rows carry the busy request id when a live request holds them.
    busy = [s for s in body["slots"] if s["busy_rid"] == "r1"]
    assert busy, "the slot held by r1 must be marked busy"


@pytest.mark.asyncio
async def test_ui_request_route(ui_app):
    ui_obs.registry.start(
        "r1", model="m1", stream=True, n_words=5, is_big=False, key="k",
        messages=[{"role": "user", "content": "full prompt here"}],
    )
    ui_obs.registry.tokens("r1", "the answer", "hmm")
    ok = await app_module.ui_request("r1")
    assert ok.status_code == 200
    body = json.loads(ok.body)
    assert body["prompt"] == "[user] full prompt here"
    assert body["response"] == "the answer"
    assert body["response_reasoning"] == "hmm"

    missing = await app_module.ui_request("nope")
    assert missing.status_code == 404

    ui_obs.registry.end("r1")
    kept = await app_module.ui_request("r1")
    assert kept.status_code == 200, "history requests keep the full prompt"
    body = json.loads(kept.body)
    assert body["prompt"] == "[user] full prompt here"
    assert body["response"] == "the answer", "history keeps the response tail"


@pytest.mark.asyncio
async def test_ui_events_first_chunk_is_snapshot(ui_app):
    resp = await app_module.ui_events()
    it = resp.body_iterator
    try:
        first = await asyncio.wait_for(anext(it), timeout=2)
        if isinstance(first, bytes):
            first = first.decode()
        payload = json.loads(first.removeprefix("data: ").strip())
        assert payload["type"] == "snapshot"
        assert "active" in payload and "history" in payload and "slots" in payload
    finally:
        await it.aclose()


@pytest.mark.asyncio
async def test_ui_disabled_returns_404(ui_app, monkeypatch):
    # app.py imports the flag by name, so patch it there (ui.py reads
    # config.UI_ENABLED live, which the hook-gating test covers).
    monkeypatch.setattr(app_module, "UI_ENABLED", False)
    assert (await app_module.ui_page_route()).status_code == 404
    assert (await app_module.ui_state()).status_code == 404
    assert (await app_module.ui_request("r1")).status_code == 404
    assert (await app_module.ui_events()).status_code == 404


# --- integration: a real stream feeds the registry ---------------------------


@pytest.mark.asyncio
async def test_stream_request_lands_in_ui_history(sm, meta_dir, monkeypatch):
    """A streamed chat request shows up live and ends up in the UI history
    with its tokens, usage, and a done status."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 10**9)
    sse = [
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"llo"}}]}\n\n',
        (
            b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,'
            b'"total_tokens":3}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]

    class FakeResp:
        status_code = 200

        async def aiter_raw(self):
            for c in sse:
                yield c

        async def aclose(self):
            pass

    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(return_value=FakeResp())
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    class FakeRequest:
        async def json(self):
            return {
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }

    # Direct endpoint calls bypass the middleware that would set the rid.
    token = request_id_var.set("rid_ui_test")
    try:
        resp = await app_module.chat(FakeRequest())
        assert resp.status_code == 200
        chunks = [c async for c in resp.body_iterator]
        assert chunks == sse
    finally:
        request_id_var.reset(token)

    # The request ran to completion inside the reader task.
    await asyncio.sleep(0.1)
    assert ui_obs.registry.active == {}
    assert len(ui_obs.registry.history) == 1
    done = ui_obs.registry.history[0]
    assert done.status == ui_obs.STATUS_DONE
    assert done.tail == "hello"
    assert done.usage == {
        "prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3
    }
    assert done.ttft is not None
    assert done.slot is not None
