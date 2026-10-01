# tests/test_inflight_save_wait.py

"""Continuation requests must not miss a restore because the previous
message's save is still in flight.

The client treats [DONE] as the end of the response and sends the next
message immediately, but the previous message's meta only lands after its
.bin write finishes. The proxy keeps an in-flight-save registry (request key
-> completion event); a continuation whose restore search misses waits for a
matching in-flight save (its key is a prefix of the continuation's request)
and re-runs the search.

Covered:
- registry register/finish semantics (event set on finish, cleared on every
  exit path, unknown keys are a no-op, re-registration is idempotent);
- the wait: no match -> immediate False, completion -> True, timeout ->
  False, disabled (timeout 0) -> False;
- end-to-end: M's save in flight -> M+1 waits and restores from M's key;
- end-to-end: the wait is bounded -> a stuck save does not hang the request.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow
import hashing as hs
from core import config


@pytest.fixture(autouse=True)
def clean_registry():
    """The registry is module state: no entry may leak between tests."""
    chat_flow._INFLIGHT_SAVES.clear()
    yield
    chat_flow._INFLIGHT_SAVES.clear()


# --- registry unit tests ---------------------------------------------------


def test_register_finish_sets_event_and_clears_entry():
    key = "k1"
    chat_flow._register_inflight_save(key)
    ev = chat_flow._INFLIGHT_SAVES[key]
    assert not ev.is_set()
    chat_flow._finish_inflight_save(key)
    assert ev.is_set()
    assert key not in chat_flow._INFLIGHT_SAVES


def test_register_is_idempotent():
    """A second registration must not replace the live event: waiters may
    already be awaiting it (e.g. _background_save registers before
    _save_and_write_meta re-registers the same key)."""
    key = "k1"
    chat_flow._register_inflight_save(key)
    ev = chat_flow._INFLIGHT_SAVES[key]
    chat_flow._register_inflight_save(key)
    assert chat_flow._INFLIGHT_SAVES[key] is ev
    chat_flow._finish_inflight_save(key)
    assert key not in chat_flow._INFLIGHT_SAVES


def test_finish_unknown_key_is_noop():
    chat_flow._finish_inflight_save("no-such-key")  # must not raise


# --- wait unit tests ---------------------------------------------------------


async def test_wait_no_inflight_returns_false():
    assert await chat_flow._wait_for_inflight_save(["a", "b"]) is False


async def test_wait_no_match_returns_false():
    chat_flow._register_inflight_save("other")
    assert await chat_flow._wait_for_inflight_save(["a", "b"]) is False


async def test_wait_completion_returns_true():
    ev = asyncio.Event()
    chat_flow._INFLIGHT_SAVES["a"] = ev

    async def finish_later():
        await asyncio.sleep(0.05)
        ev.set()

    task = asyncio.create_task(finish_later())
    try:
        assert await chat_flow._wait_for_inflight_save(["a", "b"]) is True
    finally:
        await task


async def test_wait_timeout_returns_false(monkeypatch):
    monkeypatch.setattr(chat_flow, "SAVE_WAIT_TIMEOUT", 0.05)
    chat_flow._register_inflight_save("a")  # never finished
    assert await chat_flow._wait_for_inflight_save(["a"]) is False


async def test_wait_disabled_returns_false(monkeypatch):
    monkeypatch.setattr(chat_flow, "SAVE_WAIT_TIMEOUT", 0)
    chat_flow._register_inflight_save("a")
    assert await chat_flow._wait_for_inflight_save(["a"]) is False


# --- end-to-end: continuation waits for the in-flight save ------------------


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


async def _chat(sm, messages):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {"messages": messages, "stream": False}
    return await app_module.chat(FakeRequest(data))


def _assert_all_free(sm):
    for g, lock in sm._locks.items():
        assert not lock.locked(), f"slot {g} leaked"


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


def _msgs_m():
    return [{"role": "user", "content": "first message"}]


def _msgs_m1():
    return _msgs_m() + [
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second message"},
    ]


def _key_m():
    """Cache key of M's request (a strict prefix of M+1's request)."""
    _prefix, key, _blocks, _hashes, _n = hs.request_prefix_values(
        _msgs_m(), "m1", config.WORDS_PER_BLOCK
    )
    return key


def _write_m_meta_as_save_would(key_m):
    """Write M's meta exactly as _save_and_write_meta would (prompt+response)."""
    saved_prefix, saved_blocks, saved_hashes = hs.saved_conversation_values(
        _msgs_m(), "reply", "m1", config.WORDS_PER_BLOCK
    )
    hs.write_meta(
        key_m,
        saved_prefix,
        saved_blocks,
        config.WORDS_PER_BLOCK,
        "m1",
        None,
        None,
        saved_hashes,
    )


async def test_continuation_waits_for_inflight_save_and_restores(
    sm, meta_dir, monkeypatch
):
    """M+1 must wait for M's in-flight save and then restore from M's key."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    monkeypatch.setattr(chat_flow, "SAVE_WAIT_TIMEOUT", 5.0)
    # M+1's own save must not touch the disk or the registry.
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", AsyncMock(return_value=True))

    key_m = _key_m()
    ev = asyncio.Event()
    chat_flow._INFLIGHT_SAVES[key_m] = ev  # M's save in flight, meta absent

    async def finish_save_later():
        await asyncio.sleep(0.1)
        _write_m_meta_as_save_would(key_m)
        ev.set()

    task = asyncio.create_task(finish_save_later())
    try:
        resp = await _chat(sm, _msgs_m1())
    finally:
        await task
        chat_flow._INFLIGHT_SAVES.pop(key_m, None)

    assert resp.status_code == 200
    client = sm.backends[0]["client"]
    client.restore_slot.assert_awaited_once()
    _slot_id, restored_key = client.restore_slot.call_args.args[:2]
    assert restored_key == key_m, "M+1 must restore from M's in-flight save key"
    client.erase_slot.assert_not_awaited()  # a successful restore skips the erase
    await _pump()
    _assert_all_free(sm)


async def test_inflight_save_timeout_proceeds_without_restore(
    sm, meta_dir, monkeypatch, caplog
):
    """A stuck save must not hang the continuation: bounded wait, then miss."""
    monkeypatch.setattr(chat_flow, "BIG_THRESHOLD_WORDS", 1)
    monkeypatch.setattr(chat_flow, "SAVE_WAIT_TIMEOUT", 0.1)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", AsyncMock(return_value=True))

    key_m = _key_m()
    chat_flow._INFLIGHT_SAVES[key_m] = asyncio.Event()  # never finished

    try:
        with caplog.at_level("WARNING", logger="chat_flow"):
            resp = await _chat(sm, _msgs_m1())
    finally:
        chat_flow._INFLIGHT_SAVES.pop(key_m, None)

    assert resp.status_code == 200
    client = sm.backends[0]["client"]
    client.restore_slot.assert_not_awaited()
    client.erase_slot.assert_awaited()  # no restore -> the slot is erased
    assert any(
        "inflight_save_wait_timeout" in r.message for r in caplog.records
    ), "the timed-out wait must be logged"
    await _pump()
    _assert_all_free(sm)
