# tests/test_chat_flow_hashing_offloop.py

"""M-1: chat_flow must compute the request prefix values off the event loop
(via request_prefix_values_async) and must not re-tokenize the prefix a
second time for the word count."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow
import hashing as hs


def _big_data(n_msgs: int = 3) -> dict:
    """A request big enough (word count > threshold) to exercise the
    prefix-hash path."""
    return {
        "messages": [
            {"role": "user", "content": f"message {i} " + "word " * 200}
            for i in range(n_msgs)
        ],
        "stream": False,
    }


@pytest.mark.asyncio
async def test_chat_flow_computes_prefix_values_off_loop(sm, monkeypatch):
    """The heavy hashing/tokenization runs in the thread pool: the event loop
    stays responsive and the values are produced by request_prefix_values."""
    info = {}

    def blocking_values(messages, model_id, wpb, include_reasoning=False):
        info["thread"] = threading.current_thread().ident
        time.sleep(0.3)
        return (
            hs.raw_prefix(messages, include_reasoning),
            hs.prefix_key_sha256(
                model_id + "\n" + hs.raw_prefix(messages, include_reasoning)
            ),
            [],
            [],
            10_000,
        )

    monkeypatch.setattr(hs, "request_prefix_values", blocking_values)
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(return_value={"choices": []})
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(10):
            ticks += 1
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    resp = await chat_flow.chat_flow(sm, [client], _big_data())
    await hb

    assert resp.status_code == 200
    assert info["thread"] != threading.main_thread().ident, (
        "prefix values must be computed off the event loop thread"
    )
    assert ticks >= 3, f"event loop was blocked during hashing: {ticks} heartbeats"


@pytest.mark.asyncio
async def test_chat_flow_does_not_retokenize_for_word_count(sm, monkeypatch):
    """The word count must come from the single request_prefix_values pass,
    not from a second words_from_text call on the prefix."""
    words_calls = []
    # Spy the tokenizer (delegating to the real one): the single
    # request_prefix_values pass tokenizes once; a redundant second call from
    # chat_flow for the word count would show up as an extra entry.
    real_words = hs.words_from_text

    def spy_words(text):
        words_calls.append(text)
        return real_words(text)

    monkeypatch.setattr(hs, "words_from_text", spy_words)

    calls = []
    real_values = hs.request_prefix_values

    def counting_values(messages, model_id, wpb, include_reasoning=False):
        calls.append((messages, model_id, wpb))
        return real_values(messages, model_id, wpb, include_reasoning)

    monkeypatch.setattr(hs, "request_prefix_values", counting_values)
    client = sm.backends[0]["client"]
    client.chat_completions = AsyncMock(return_value={"choices": []})
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]

    resp = await chat_flow.chat_flow(sm, [client], _big_data())

    assert resp.status_code == 200
    assert len(calls) == 1, "request_prefix_values must run exactly once"
    # The only tokenization is the one inside request_prefix_values; chat_flow
    # itself must not call words_from_text again for the word count.
    assert len(words_calls) == 1
    assert words_calls[0] == hs.raw_prefix(_big_data()["messages"])
