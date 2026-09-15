# tests/test_reqlog.py

"""Request/response/prefix logging with group rotation:

- rotation keeps only the newest N groups (all their files), leaves foreign
  files untouched, and is disabled at N=0;
- a small non-stream request writes request/response/prefix files;
- a big non-stream request gets its prefix.json from the background save;
- a stream request writes response (assembled) + raw (SSE) + prefix;
- the request file is written even when the provider errors;
- logging is a no-op when the directory is disabled.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow
import reqlog
from request_id import request_id_var


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class FakeResp:
    """Mimics the httpx.Response parts used by the reader: aiter_raw + aclose."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    async def aiter_raw(self):
        for c in self._chunks:
            yield c

    async def aclose(self):
        self.closed = True


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


def _groups(d):
    """Map group prefix (ts.rid) -> set of file types present."""
    groups: dict[str, set] = {}
    for p in d.iterdir():
        if p.name.endswith(".tmp"):
            continue
        m = reqlog._GROUP_RE.match(p.name)
        if m:
            groups.setdefault(f"{m.group(1)}.{m.group(2)}", set()).add(m.group(3))
    return groups


def _read(d, name):
    return json.loads((d / name).read_text(encoding="utf-8"))


# --- rotation unit tests ----------------------------------------------------


def test_rotate_keeps_newest_groups(reqlog_dir):
    for i in range(150):
        for t in ("request", "response", "prefix"):
            (reqlog_dir / f"{1000 + i}.{i:04d}.{t}.json").write_text("{}")
    (reqlog_dir / "foreign.txt").write_text("keep me")
    reqlog._rotate()
    groups = _groups(reqlog_dir)
    assert len(groups) == 100
    # ts 1000..1149 -> the 100 newest (1050..1149) survive.
    assert all(1050 <= int(g.split(".")[0]) <= 1149 for g in groups)
    assert (reqlog_dir / "foreign.txt").exists()


def test_rotate_disabled_at_zero(reqlog_dir, monkeypatch):
    monkeypatch.setattr(reqlog, "REQUEST_LOG_MAX_GROUPS", 0)
    for i in range(5):
        (reqlog_dir / f"{i}.r.request.json").write_text("{}")
    reqlog._rotate()
    assert len(_groups(reqlog_dir)) == 5


def test_log_file_noop_when_disabled(reqlog_dir, monkeypatch):
    monkeypatch.setattr(reqlog, "REQUEST_LOG_DIR", "")
    reqlog.log_file("request", "rid", "123", {"a": 1})
    assert not reqlog._LOG_TASKS
    assert list(reqlog_dir.iterdir()) == []


# --- non-stream e2e ---------------------------------------------------------


async def _chat(sm, content, stream=False, rid="rid123"):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    token = request_id_var.set(rid)
    try:
        data = {
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        }
        return await app_module.chat(FakeRequest(data))
    finally:
        request_id_var.reset(token)


async def test_small_json_writes_group(sm, reqlog_dir):
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"choices": [{"message": {"role": "assistant", "content": "world"}}]}
    )
    resp = await _chat(sm, "hello")
    assert resp.status_code == 200
    await _drain()

    groups = _groups(reqlog_dir)
    (g,) = groups
    assert g.endswith(".rid123")
    assert groups[g] == {"request", "response", "prefix"}

    assert _read(reqlog_dir, f"{g}.request.json") == {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    assert _read(reqlog_dir, f"{g}.response.json")["choices"][0]["message"][
        "content"
    ] == "world"
    prefix = _read(reqlog_dir, f"{g}.prefix.json")
    # The next message will be matched against prompt + assistant response.
    assert "user:hello" in prefix["prefix"]
    assert "assistant:world" in prefix["prefix"]
    assert len(prefix["key"]) == 64


async def test_big_json_prefix_from_bg_save(sm, meta_dir, reqlog_dir, monkeypatch):
    """Big requests reuse the background save's computed saved values."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", AsyncMock(return_value=True))
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"choices": [{"message": {"role": "assistant", "content": "ans"}}]}
    )
    resp = await _chat(sm, "hello world")
    assert resp.status_code == 200
    await _drain()

    groups = _groups(reqlog_dir)
    (g,) = groups
    assert groups[g] == {"request", "response", "prefix"}
    prefix = _read(reqlog_dir, f"{g}.prefix.json")
    assert "user:hello world" in prefix["prefix"]
    assert "assistant:ans" in prefix["prefix"]


async def test_request_logged_on_provider_error(sm, reqlog_dir):
    """The request file lands even when the provider fails (no response file)."""
    sm.backends[0]["client"].chat_completions = AsyncMock(
        return_value={"object": "error", "status": 500, "message": "boom"}
    )
    resp = await _chat(sm, "hello")
    assert resp.status_code == 502
    await _drain()

    groups = _groups(reqlog_dir)
    (g,) = groups
    assert groups[g] == {"request"}


# --- stream e2e -------------------------------------------------------------


_SSE = [
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\ndata: [DONE]\n\n',
]


async def _stream(sm, is_big, rid="rid123"):
    g = (0, "model", 0)
    lock = sm._lock_for(g)
    await lock.acquire()
    gen = await chat_flow.start_stream_task(
        FakeResp(_SSE),
        g,
        "k" * 16,
        "prefix",
        ["b"],
        "model",
        sm,
        is_big,
        ["h"],
        [sm.backends[0]["client"]],
        [{"role": "user", "content": "hi"}],
        rid,
        "123",
    )
    received = [c async for c in gen]
    await _drain()
    return received


async def test_small_stream_writes_group(sm, reqlog_dir):
    received = await _stream(sm, is_big=False)
    assert received == _SSE
    (grp,) = _groups(reqlog_dir)
    assert grp == "123.rid123"
    assert _groups(reqlog_dir)[grp] == {"response", "raw", "prefix"}

    resp = _read(reqlog_dir, f"{grp}.response.json")
    assert resp["content"] == "Hello"
    assert resp["completed"] is True
    assert resp["error"] is None
    raw = _read(reqlog_dir, f"{grp}.raw.json")
    assert raw["sse"].startswith('data: {"choices"')
    assert "[DONE]" in raw["sse"]
    prefix = _read(reqlog_dir, f"{grp}.prefix.json")
    assert "user:hi" in prefix["prefix"]
    assert "assistant:Hello" in prefix["prefix"]


async def test_big_stream_prefix_from_bg_save(sm, meta_dir, reqlog_dir):
    """A big completed stream gets its prefix.json from the background save."""
    await _stream(sm, is_big=True)
    groups = _groups(reqlog_dir)
    (grp,) = groups
    assert groups[grp] == {"response", "raw", "prefix"}
    prefix = _read(reqlog_dir, f"{grp}.prefix.json")
    assert "assistant:Hello" in prefix["prefix"]
    sm.backends[0]["client"].save_slot.assert_awaited_once()


# --- rotation through the write path ----------------------------------------


async def test_write_path_rotates_to_max_groups(reqlog_dir, monkeypatch):
    monkeypatch.setattr(reqlog, "REQUEST_LOG_MAX_GROUPS", 2)
    for i in range(3):
        reqlog.log_file("request", f"r{i}", str(1000 + i), {"i": i})
    await _drain()
    groups = _groups(reqlog_dir)
    assert groups == {
        "1001.r1": {"request"},
        "1002.r2": {"request"},
    }
