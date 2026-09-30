# tests/test_bg_save.py

"""Non-stream big-request save+meta runs in a background task:

- the JSON response is returned WITHOUT waiting for the .bin disk write;
- the slot stays held until the background save completes, then is released
  by the task (exactly once, mirroring the stream reader's ownership);
- a save failure still releases the slot and is logged, never propagated
  to the client (the response was already sent);
- small requests are unchanged: no background save, slot released in finally.
"""

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


def _assert_all_locked(sm):
    for g, lock in sm._locks.items():
        assert lock.locked(), f"slot {g} released too early"


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_big_json_response_returns_before_slow_save(
    sm, meta_dir, monkeypatch, caplog
):
    """The response must not wait for the background .bin write."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_save(*_args, **_kwargs):
        started.set()
        await release.wait()
        return True

    save_mock = AsyncMock(side_effect=slow_save)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    with caplog.at_level("WARNING", logger="chat_flow"):
        resp = await _chat(sm, "hello world")

    # Let the scheduled task start (it then blocks on `release`).
    await _pump(0.05)
    # The response came back while the save is still running: it was
    # scheduled as a background task, not awaited inline.
    try:
        assert resp.status_code == 200
        assert started.is_set(), "the background save must have started"
        assert not release.is_set(), "the response must not wait for the save"
        _assert_all_locked(sm)  # the slot stays held until the save completes
    finally:
        # Always unblock the save so a failed assertion cannot hang the loop.
        release.set()
    await _pump()
    save_mock.assert_awaited_once()
    _assert_all_free(sm)  # the background task released the slot
    assert not any("background_save_error" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_big_json_release_called_exactly_once(sm, meta_dir, monkeypatch):
    """The outer finally must not release a slot owned by the save task."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    calls: list = []
    original = sm.release

    def counting_release(g):
        calls.append(g)
        original(g)

    sm.release = counting_release
    await _chat(sm, "hello world")
    await _pump()

    assert len(calls) == 1, f"slot released {len(calls)} times: {calls}"
    _assert_all_free(sm)


@pytest.mark.asyncio
async def test_big_json_save_failure_releases_slot_and_logs(
    sm, meta_dir, monkeypatch, caplog
):
    """A save exception must not leak the slot or crash the task silently."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)

    async def failing_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(chat_flow, "_save_and_write_meta", failing_save)

    with caplog.at_level("WARNING", logger="chat_flow"):
        resp = await _chat(sm, "hello world")

    assert resp.status_code == 200, "the client must not see the save failure"
    await _pump()
    _assert_all_free(sm)
    assert any(
        "background_save_error" in r.message for r in caplog.records
    ), "the save failure must be logged"


@pytest.mark.asyncio
async def test_bg_save_registers_inflight_before_response_hash(
    sm, meta_dir, monkeypatch
):
    """The in-flight entry must exist from the start of _background_save,
    covering the response-hash phase: a fast continuation sent right after
    the JSON response must be able to wait for the save instead of missing
    it and reprocessing the whole prompt."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    # The real save would register/finish the entry itself; mock it so the
    # test observes only _background_save's own registration and cleanup.
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", AsyncMock(return_value=True))
    chat_flow._INFLIGHT_SAVES.clear()

    observed: dict = {}

    async def spy_saved_values(*args, **_kwargs):
        observed["inflight_keys"] = list(chat_flow._INFLIGHT_SAVES)
        return args[3], args[4], args[5]  # fallback prefix, blocks, hashes

    monkeypatch.setattr(chat_flow, "_saved_conversation_values", spy_saved_values)

    try:
        resp = await _chat(sm, "hello world")
        await _pump()
        assert resp.status_code == 200
        assert observed["inflight_keys"], (
            "the in-flight save must be registered before the response-hash step"
        )
        assert not chat_flow._INFLIGHT_SAVES, (
            "no stale in-flight entry may remain after the save completes"
        )
        _assert_all_free(sm)
    finally:
        chat_flow._INFLIGHT_SAVES.clear()


@pytest.mark.asyncio
async def test_small_json_no_background_save(sm, meta_dir, monkeypatch):
    """Small requests are unchanged: no save, slot released in the finally."""
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)

    resp = await _chat(sm, "small")

    assert resp.status_code == 200
    _assert_all_free(sm)
    await _pump()
    save_mock.assert_not_awaited()
