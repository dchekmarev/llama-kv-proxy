# tests/test_response_meta.py

"""Response-aware meta: the assistant response is included in
saved_prefix_hashes while the .bin key remains the request prompt key."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import chat_flow
import hashing as hs
import slot_manager as sm_module
from slot_manager import SlotManager


class FakeResp:
    def __init__(self, chunks, delay=0.0):
        self._chunks = list(chunks)
        self._delay = delay
        self.closed = False

    async def aiter_raw(self):
        for c in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield c

    async def aclose(self):
        self.closed = True


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    manager.set_clients([client])
    return manager


async def _acquire(sm, g=(0, "model", 0)):
    lock = sm._lock_for(g)
    await lock.acquire()
    return lock


async def _pump(seconds=0.3):
    await asyncio.sleep(seconds)


def test_assistant_content_normal():
    out = {"choices": [{"message": {"content": "hi"}}]}
    assert chat_flow._assistant_content(out) == ("hi", "", "reasoning_content")


def test_assistant_content_missing_choices():
    assert chat_flow._assistant_content({"choices": []}) == ("", "", "reasoning_content")


def test_assistant_content_missing_message():
    assert chat_flow._assistant_content({"choices": [{}]}) == ("", "", "reasoning_content")


def test_assistant_content_none_content():
    out = {"choices": [{"message": {"content": None}}]}
    assert chat_flow._assistant_content(out) == ("", "", "reasoning_content")


def test_assistant_content_non_string():
    out = {"choices": [{"message": {"content": ["a", "b"]}}]}
    assert chat_flow._assistant_content(out) == ("['a', 'b']", "", "reasoning_content")


def test_append_stream_content_delta():
    parts = []
    chat_flow._append_stream_content(
        'data: {"choices": [{"delta": {"content": "he"}}]}', parts
    )
    assert parts == ["he"]


def test_append_stream_content_message_fallback():
    parts = []
    chat_flow._append_stream_content(
        'data: {"choices": [{"message": {"content": "lo"}}]}', parts
    )
    assert parts == ["lo"]


def test_append_stream_content_ignores_done_and_invalid():
    parts = []
    chat_flow._append_stream_content("data: [DONE]", parts)
    chat_flow._append_stream_content("data: not-json", parts)
    chat_flow._append_stream_content('data: {"choices": []}', parts)
    assert parts == []


@pytest.mark.asyncio
async def test_saved_conversation_values_empty_response_fallback():
    fallback = ("fp", ["fb"], ["fh"])
    result = await chat_flow._saved_conversation_values(
        [{"role": "user", "content": "a"}], "", "m1", *fallback
    )
    assert result == fallback


@pytest.mark.asyncio
async def test_saved_conversation_values_with_response():
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = await chat_flow._saved_conversation_values(
        messages, "b", "m1", "fp", ["fb"], ["fh"]
    )
    expected = messages + [{"role": "assistant", "content": "b"}]
    assert prefix == hs.raw_prefix(expected)
    assert blocks == hs.block_hashes_from_text(prefix, hs.WORDS_PER_BLOCK)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1")


@pytest.mark.asyncio
async def test_stream_saves_response_extended_meta(sm, monkeypatch):
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)
    g = (0, "model", 0)
    await _acquire(sm, g)
    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\ndata: [DONE]\n\n',
    ]
    resp = FakeResp(chunks)
    messages = [{"role": "user", "content": "hi"}]

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True, messages=messages
    )
    _ = [c async for c in gen]
    await _pump(0.2)

    save_mock.assert_awaited_once()
    args = save_mock.await_args.args
    expected_messages = messages + [{"role": "assistant", "content": "Hello"}]
    assert args[3] == "k" * 16
    assert args[4] == hs.raw_prefix(expected_messages)
    assert args[5] == hs.block_hashes_from_text(args[4], hs.WORDS_PER_BLOCK)
    assert args[6] == []
    assert args[7] == "model"
    assert args[8] == hs.prefix_hashes_from_messages(expected_messages, "model")


@pytest.mark.asyncio
async def test_stream_parses_sse_split_across_chunks(sm, monkeypatch):
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)
    g = (0, "model", 0)
    await _acquire(sm, g)
    chunks = [
        b'data: {"choices": [{"delta": {"con',
        b'tent": "split"}}]}\n\n',
    ]
    resp = FakeResp(chunks)
    messages = [{"role": "user", "content": "hi"}]

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True, messages=messages
    )
    _ = [c async for c in gen]
    await _pump(0.2)

    save_mock.assert_awaited_once()
    args = save_mock.await_args.args
    expected_messages = messages + [{"role": "assistant", "content": "split"}]
    assert args[8] == hs.prefix_hashes_from_messages(expected_messages, "model")


@pytest.mark.asyncio
async def test_stream_without_response_falls_back_to_prompt_meta(sm, monkeypatch):
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)
    g = (0, "model", 0)
    await _acquire(sm, g)
    resp = FakeResp([b"not sse"])
    messages = [{"role": "user", "content": "hi"}]

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True, messages=messages
    )
    _ = [c async for c in gen]
    await _pump(0.2)

    save_mock.assert_awaited_once()
    args = save_mock.await_args.args
    assert args[4] == "prefix"
    assert args[5] == ["b"]
    assert args[8] == []


@pytest.mark.asyncio
async def test_stream_falls_back_when_saved_values_fail(sm, monkeypatch):
    save_mock = AsyncMock(return_value=True)

    async def _raise(*_args, **_kwargs):
        raise RuntimeError("hashing failed")

    monkeypatch.setattr(chat_flow, "_saved_conversation_values", _raise)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)
    g = (0, "model", 0)
    await _acquire(sm, g)
    chunks = [b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n']
    resp = FakeResp(chunks)
    messages = [{"role": "user", "content": "hi"}]

    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True, messages=messages
    )
    _ = [c async for c in gen]
    await _pump(0.2)

    save_mock.assert_awaited_once()
    args = save_mock.await_args.args
    assert args[4] == "prefix"
    assert args[5] == ["b"]
    assert args[8] == []
