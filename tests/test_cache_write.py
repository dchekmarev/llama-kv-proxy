# tests/test_cache_write.py

"""cache_write:false keeps the restore side of the KV cache and only disables
saving the result. cache_prompt:false remains the full opt-out (no restore, no
save). Neither control flag participates in the cache key."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow
import hashing as hs
from core import config


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


def _body(content, **flags):
    data = {"messages": [{"role": "user", "content": content}], "stream": False}
    data.update(flags)
    return data


async def _chat(sm, data):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    return await app_module.chat(FakeRequest(data)), client


def _write_meta(content):
    prefix = hs.raw_prefix([{"role": "user", "content": content}])
    key = hs.prefix_key_sha256("m1\n" + prefix)
    blocks = hs.block_hashes_from_text(prefix, hs.WORDS_PER_BLOCK)
    hs.write_meta(key, prefix, blocks, hs.WORDS_PER_BLOCK, "m1")
    return key


def _assert_all_free(sm):
    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_cache_write_false_restores_but_does_not_save(
    sm, meta_dir, monkeypatch
):
    """A big request with cache_write:false still restores; nothing is saved."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)
    key = _write_meta("hello world")

    resp, client = await _chat(sm, _body("hello world", cache_write=False))

    assert resp.status_code == 200
    client.restore_slot.assert_awaited_once()
    _slot_id, restored_key = client.restore_slot.call_args.args[:2]
    assert restored_key == key
    client.erase_slot.assert_not_awaited()
    await _pump()
    save_mock.assert_not_awaited()
    body = client.chat_completions.await_args.args[0]
    assert "cache_write" not in body, "cache_write must not leak to the backend"
    assert body["cache_prompt"] is True
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_cache_write_false_miss_leaves_slot_untouched(
    sm, meta_dir, monkeypatch
):
    """No restore candidate: no save and no pre-chat erase either."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    resp, client = await _chat(sm, _body("hello world", cache_write=False))

    assert resp.status_code == 200
    client.restore_slot.assert_not_awaited()
    client.erase_slot.assert_not_awaited()
    await _pump()
    save_mock.assert_not_awaited()
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_cache_prompt_false_wins_over_cache_write(
    sm, meta_dir, monkeypatch
):
    """cache_prompt:false is the full opt-out even with cache_write:true."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)
    _write_meta("hello world")

    resp, client = await _chat(
        sm, _body("hello world", cache_prompt=False, cache_write=True)
    )

    assert resp.status_code == 200
    client.restore_slot.assert_not_awaited()
    client.erase_slot.assert_not_awaited()
    await _pump()
    save_mock.assert_not_awaited()
    body = client.chat_completions.await_args.args[0]
    assert body["cache_prompt"] is False
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_default_request_still_saves(sm, meta_dir, monkeypatch):
    """Neither flag present: a big request saves as before."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    resp, _client = await _chat(sm, _body("hello world"))

    assert resp.status_code == 200
    await _pump()
    save_mock.assert_awaited_once()
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_cache_write_false_stream_does_not_save(
    sm, meta_dir, monkeypatch
):
    """The stream reader honors cache_write:false: no save handoff."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    sse = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        (
            b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,'
            b'"total_tokens":3}}\n\n'
        ),
        b"data: [DONE]\n\n",
    ]

    class FakeResp:
        status_code = 200

        async def aiter_raw(self):
            for chunk in sse:
                yield chunk

        async def aclose(self):
            pass

    sm.backends[0]["client"].chat_completions = AsyncMock(return_value=FakeResp())

    resp, client = await _chat(
        sm,
        {
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": True,
            "cache_write": False,
        },
    )

    assert resp.status_code == 200
    chunks = [chunk async for chunk in resp.body_iterator]
    assert chunks == sse
    client.erase_slot.assert_not_awaited()
    await _pump()
    save_mock.assert_not_awaited()
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_cache_write_in_options_is_not_forwarded(
    sm, meta_dir, monkeypatch
):
    """The control flag is stripped from the root and from options."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    data = _body("hello world", cache_write=False)
    data["options"] = {"cache_write": False}

    resp, client = await _chat(sm, data)

    assert resp.status_code == 200
    body = client.chat_completions.await_args.args[0]
    assert "cache_write" not in body
    assert "cache_write" not in body["options"]
    _assert_all_free(sm)


def test_control_flags_are_not_part_of_the_cache_key():
    """Only render-affecting params enter the key: the two control flags must
    not fork the cache namespace."""
    messages = [{"role": "user", "content": "hello world"}]
    plain = chat_flow._render_ctx_of({})
    flagged = chat_flow._render_ctx_of(
        {"cache_prompt": False, "cache_write": False}
    )
    assert "cache_prompt" not in flagged
    assert "cache_write" not in flagged

    _p, key_plain, _b, _h, _n = hs.request_prefix_values(
        messages, "m1", config.WORDS_PER_BLOCK, render_ctx=plain
    )
    _p, key_flagged, _b, _h, _n = hs.request_prefix_values(
        messages, "m1", config.WORDS_PER_BLOCK, render_ctx=flagged
    )
    assert key_plain == key_flagged
