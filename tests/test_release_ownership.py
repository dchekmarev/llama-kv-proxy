# tests/test_release_ownership.py

"""P2-4: the slot must be released exactly once per request on every path —
success, provider error, exception, and cancellation. A second release is a
time bomb: between two releases another request may have acquired the slot,
and the second release would free someone else's lock."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import app as app_module


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


async def _chat(sm, content, stream=False):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    return await app_module.chat(FakeRequest(data))


def _assert_all_free(sm):
    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"


@pytest.mark.asyncio
async def test_non_stream_success_releases_slot(sm, meta_dir):
    await _chat(sm, "small")
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_non_stream_exception_releases_slot(sm, meta_dir):
    sm.backends[0]["client"].chat_completions = AsyncMock(
        side_effect=RuntimeError("boom")
    )
    resp = await _chat(sm, "small")
    assert resp.status_code == 500
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_cancellation_releases_slot(sm, meta_dir):
    """A cancelled request (BaseException, not caught by except Exception)
    must still release the slot."""

    async def slow_chat(body, slot_id=None, stream=False):
        await asyncio.sleep(10)

    sm.backends[0]["client"].chat_completions = slow_chat
    task = asyncio.create_task(_chat(sm, "small"))
    await asyncio.sleep(0.05)  # let it acquire the slot and enter chat
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_stream_provider_error_releases_slot(sm, meta_dir):
    class ErrResp:
        status_code = 500

        async def aread(self):
            return b"backend down"

        async def aclose(self):
            pass

    sm.backends[0]["client"].chat_completions = AsyncMock(return_value=ErrResp())
    resp = await _chat(sm, "small", stream=True)
    assert resp.status_code == 500
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_release_called_exactly_once(sm, meta_dir):
    """Regression guard: no path may release the same slot twice."""
    calls: list = []
    original = sm.release

    def counting_release(g):
        calls.append(g)
        original(g)

    sm.release = counting_release
    await _chat(sm, "small")

    assert len(calls) == 1, f"slot released {len(calls)} times: {calls}"
