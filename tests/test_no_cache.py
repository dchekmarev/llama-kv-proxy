# tests/test_no_cache.py

"""A client opts a request out of the KV cache by sending cache_prompt:false
in the body. Such a request is proxied onto a free/oldest slot untouched:
no restore search, no pre-chat erase, and no save of bin/meta — for both the
stream and non-stream paths. Absent or true keeps the normal big/small cache
behavior (an explicit true is NOT a no-cache signal)."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


async def _chat(sm, content, cache_prompt, stream=False):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if cache_prompt is not None:
        data["cache_prompt"] = cache_prompt
    return await app_module.chat(FakeRequest(data)), client


def _assert_all_free(sm):
    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_no_cache_big_json_untouched_slot(sm, meta_dir, monkeypatch):
    """cache_prompt:false on a big request: no restore, no erase, no save."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    # Spy on acquire to prove no restore key was requested from the slot mgr.
    acquired: dict = {}
    orig = sm.acquire_for_request

    async def spy(model, restore_key=None, resolve_restore_key=None):
        acquired["restore_key"] = restore_key
        return await orig(model, restore_key, resolve_restore_key=resolve_restore_key)

    sm.acquire_for_request = spy

    resp, client = await _chat(sm, "hello world", cache_prompt=False)

    assert resp.status_code == 200
    assert acquired["restore_key"] is None, "no-cache must not request a restore"
    client.erase_slot.assert_not_awaited()
    await _pump()
    save_mock.assert_not_awaited()
    body = client.chat_completions.await_args.args[0]
    assert body["cache_prompt"] is False
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_cache_prompt_true_big_still_saves(sm, meta_dir, monkeypatch):
    """An explicit cache_prompt:true is not a no-cache signal: a big request
    still saves the KV cache (only an explicit false disables it)."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    resp, client = await _chat(sm, "hello world", cache_prompt=True)

    assert resp.status_code == 200
    await _pump()
    save_mock.assert_awaited_once()
    body = client.chat_completions.await_args.args[0]
    assert body["cache_prompt"] is True
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_no_cache_big_stream_no_save_no_erase(sm, meta_dir, monkeypatch):
    """The stream path honors no-cache too: no save handoff, no erase."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    sse = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        (
            b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,'
            b'"total_tokens":3},"timings":{"prompt_per_second":1}}\n\n'
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

    sm.backends[0]["client"].chat_completions = AsyncMock(return_value=FakeResp())

    resp, client = await _chat(sm, "hello world", cache_prompt=False, stream=True)
    assert resp.status_code == 200
    chunks = [c async for c in resp.body_iterator]
    assert chunks == sse
    client.erase_slot.assert_not_awaited()
    await _pump()
    save_mock.assert_not_awaited()
    _assert_all_free(sm)
