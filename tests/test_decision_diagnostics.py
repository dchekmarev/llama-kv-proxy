# tests/test_decision_diagnostics.py

"""Diagnostics added to pinpoint full-prompt reprocessing on the backend:
- _stream_usage_of: extracts the final llama.cpp SSE usage/timings chunk
  (response.json usage is the ground truth for cached_tokens vs the restored
  prefix);
- _snapshot_slot: records the slot's KV state right after a restore so an
  "ok but n_past==0" restore is visible in decision.json;
- decision.json content threaded from the restore decision through the save
  outcome."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import chat_flow
import reqlog


@pytest.fixture()
def reqlog_dir(tmp_path, monkeypatch):
    """Point reqlog at a fresh temp dir with a high group cap."""
    monkeypatch.setattr(reqlog, "REQUEST_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(reqlog, "REQUEST_LOG_MAX_GROUPS", 100)
    return tmp_path


async def _drain(timeout=3.0):
    """Wait until all log-write and background save/prefix tasks finished."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        tasks = list(reqlog._LOG_TASKS) + list(chat_flow._BG_SAVE_TASKS)
        if not tasks:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("log tasks did not finish in time")


def _usage_sse(extra_usage=None, extra_timings=None):
    usage = {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45}
    usage.update(extra_usage or {})
    timings = {"prompt_n": 40, "prompt_per_second": 99.9}
    timings.update(extra_timings or {})
    return json.dumps(
        {"models": [], "choices": [], "usage": usage, "timings": timings}
    )


# --- _stream_usage_of --------------------------------------------------------


def test_stream_usage_ignores_content_chunks():
    assert chat_flow._stream_usage_of('data: {"choices":[{"delta":{"content":"hi"}}]') is None
    assert chat_flow._stream_usage_of("data: [DONE]") is None
    assert chat_flow._stream_usage_of("whatever") is None
    assert chat_flow._stream_usage_of("data: not-json{") is None


def test_stream_usage_extracts_final_chunk():
    usage, timings = chat_flow._stream_usage_of(f"data: {_usage_sse(extra_usage={'cached_tokens': 35})}")
    assert usage["cached_tokens"] == 35
    assert usage["total_tokens"] == 45
    assert timings["prompt_per_second"] == 99.9


def test_stream_usage_returns_empty_timings_when_absent():
    chunk = 'data: ' + json.dumps({"usage": {"total_tokens": 45}})
    usage, timings = chat_flow._stream_usage_of(chunk)
    assert usage == {"total_tokens": 45}
    assert timings == {}


def test_stream_usage_rejects_empty_usage():
    assert chat_flow._stream_usage_of('data: {"usage": {}}') is None
    assert chat_flow._stream_usage_of('data: {}') is None


# --- _snapshot_slot ----------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_slot_records_kv_state():
    client = MagicMock()
    client.get_slots = AsyncMock(
        return_value=[{"id": 0, "state": 1, "n_past": 0, "n_tokens": 0}]
    )
    snap = await chat_flow._snapshot_slot(client, 0, "m1")
    assert snap == {"state": 1, "n_past": 0, "n_tokens": 0}
    client.get_slots.assert_awaited_once_with(model="m1")


@pytest.mark.asyncio
async def test_snapshot_slot_returns_none_when_get_slots_fails():
    client = MagicMock()
    client.get_slots = AsyncMock(side_effect=RuntimeError("boom"))
    assert await chat_flow._snapshot_slot(client, 0, "m1") is None


# --- decision.json content via the small stream path --------------------------


class _FakeResp:
    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_raw(self):
        for c in self._chunks:
            yield c

    async def aclose(self):
        pass


_SSE = [
    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
    (
        b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,'
        b'"total_tokens":3,"cached_tokens":2},"timings":{"prompt_per_second":1}}\n\n'
    ),
    b"data: [DONE]\n\n",
]


def _decision(sm, is_big=False):
    return {
        "is_big": is_big,
        "n_words": 3,
        "words_threshold": 500,
        "model": "m1",
        "restore": {
            "candidate_key": None,
            "candidate_ratio": None,
            "used_key": None,
            "outcome": None,
            "stale_meta_dropped": False,
        },
        "wait_inflight_save": False,
        "erase_done": False,
        "slot": {"backend": 0, "model": "m1", "id": 0},
        "slot_before_chat": None,
    }


@pytest.mark.asyncio
async def test_decision_and_usage_written_for_small_stream(reqlog_dir, sm, meta_dir):
    g = (0, "m1", 0)
    lock = sm._lock_for(g)
    await lock.acquire()
    gen = await chat_flow.start_stream_task(
        _FakeResp(_SSE),
        g,
        "k" * 16,
        "prefix",
        ["b"],
        "m1",
        sm,
        False,
        ["h"],
        [sm.backends[0]["client"]],
        [{"role": "user", "content": "hi"}],
        "rid1",
        "123",
        _decision(sm),
    )
    received = [c async for c in gen]
    assert received == _SSE

    await _drain()

    files = {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in reqlog_dir.iterdir()
        if reqlog._GROUP_RE.match(p.name) and not p.name.endswith(".tmp")
    }
    resp = files["123.rid1.response.json"]
    assert resp["usage"]["cached_tokens"] == 2
    assert resp["timings"] == {"prompt_per_second": 1}

    dec = files["123.rid1.decision.json"]
    assert dec["is_big"] is False
    assert dec["restore"]["outcome"] is None
    assert dec["save"] == {"attempted": False}
    assert "_emitted" not in dec


# --- decision emitted on error paths (slot owned, then failure) ---------------


class _FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


@pytest.mark.asyncio
async def test_decision_written_on_provider_error(reqlog_dir, sm, meta_dir):
    """A provider error after slot acquire must still persist the
    restore-phase decision: the diagnostics are needed to debug failures."""
    import app as app_module

    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"object": "error", "status": 500, "message": "boom"}
    )
    app_module.app.state.sm = sm
    app_module.app.state.clients = [sm.backends[0]["client"]]
    from request_id import request_id_var

    token = request_id_var.set("rid_err")
    try:
        data = {"messages": [{"role": "user", "content": "hello"}]}
        resp = await app_module.chat(_FakeRequest(data))
    finally:
        request_id_var.reset(token)
    assert resp.status_code == 502
    await _drain()

    files = {
        p.name: json.loads(p.read_text(encoding="utf-8"))
        for p in reqlog_dir.iterdir()
        if reqlog._GROUP_RE.match(p.name) and not p.name.endswith(".tmp")
    }
    dec = next(v for k, v in files.items() if k.endswith(".rid_err.decision.json"))
    assert dec["save"] == {"attempted": False}
    assert dec["slot_before_chat"] is None
    assert dec["restore"]["candidate_key"] is None