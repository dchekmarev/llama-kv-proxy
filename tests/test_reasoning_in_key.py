# tests/test_reasoning_in_key.py

"""M-4: reasoning_content/reasoning are omitted from the KV-cache key, so two
requests with identical content but different reasoning traces hash to the
same key. Behind REASONING_IN_KEY (off by default): the reasoning fields join
the per-message key parts and the saved assistant response, so distinct
reasoning traces get distinct cache keys. With the flag off, keys must stay
byte-identical to the legacy behavior."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import chat_flow
import hashing as hs
import slot_manager as sm_module
from slot_manager import SlotManager


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class FakeResp:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    async def aiter_raw(self):
        for c in self._chunks:
            yield c

    async def aclose(self):
        self.closed = True


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    client.restore_slot = AsyncMock(return_value=True)
    client.erase_slot = AsyncMock(return_value=True)
    client.get_model_id_cached = AsyncMock(return_value="m1")
    client.get_loaded_model = AsyncMock(return_value="m1")
    client.chat_completions = AsyncMock(return_value={"choices": []})
    manager.set_clients([client])
    return manager


# --- hashing: _message_part / request_prefix_values -------------------------


def test_flag_off_reasoning_ignored_byte_identical():
    """Flag off: a message carrying reasoning fields hashes exactly like the
    same message without them (legacy behavior, no churn)."""
    with_r = [{"role": "assistant", "content": "b", "reasoning_content": "R"}]
    without_r = [{"role": "assistant", "content": "b"}]
    assert hs._message_part(with_r[0]) == hs._message_part(without_r[0])
    assert hs.raw_prefix(with_r) == hs.raw_prefix(without_r)
    v_off = hs.request_prefix_values(with_r, "m1", 100)
    v_ref = hs.request_prefix_values(without_r, "m1", 100)
    assert v_off == v_ref


def test_flag_on_different_reasoning_different_keys():
    """Flag on: identical content, different reasoning_content -> different
    parts, prefix and cache key."""
    base = [{"role": "user", "content": "same content"}]
    a = base + [{"role": "assistant", "content": "b", "reasoning_content": "R1"}]
    b = base + [{"role": "assistant", "content": "b", "reasoning_content": "R2"}]
    assert hs._message_parts(a, True) != hs._message_parts(b, True)
    assert hs.raw_prefix(a, True) != hs.raw_prefix(b, True)
    pa, ka, _, ha, _ = hs.request_prefix_values(a, "m1", 100, True)
    pb, kb, _, hb, _ = hs.request_prefix_values(b, "m1", 100, True)
    assert ka != kb
    assert ha[-1] != hb[-1]
    assert pa != pb


def test_flag_on_same_reasoning_same_key():
    """Flag on: determinism — same reasoning always yields the same key."""
    m1 = [{"role": "assistant", "content": "b", "reasoning_content": "R"}]
    m2 = [{"role": "assistant", "content": "b", "reasoning_content": "R"}]
    assert hs.request_prefix_values(m1, "m1", 100, True) == hs.request_prefix_values(
        m2, "m1", 100, True
    )


def test_flag_on_reasoning_only_message_contributes_part():
    """Flag on: a message with only reasoning (no content) contributes a part;
    flag off it contributes nothing."""
    msg = {"role": "assistant", "content": None, "reasoning_content": "R"}
    assert hs._message_part(msg, True) != ""
    assert hs._message_part(msg) == ""


def test_flag_on_reasoning_field_variant_included():
    """Flag on: the `reasoning` field variant is included too."""
    a = [{"role": "assistant", "content": "b", "reasoning": "R1"}]
    b = [{"role": "assistant", "content": "b", "reasoning": "R2"}]
    assert hs.raw_prefix(a, True) != hs.raw_prefix(b, True)
    assert "R1" in hs._message_part(a[0], True)


def test_flag_on_dual_field_single_precedence():
    """Flag on: a message carrying BOTH reasoning fields contributes exactly one
    reasoning field to the part — reasoning_content wins over reasoning (the
    same single-field precedence the save side applies via _reasoning_of), so
    the request-side normalization is the inverse of the save-side one."""
    dual = {"role": "assistant", "content": "b", "reasoning_content": "A", "reasoning": "B"}
    winner = {"role": "assistant", "content": "b", "reasoning_content": "A"}
    part = hs._message_part(dual, True)
    assert part == hs._message_part(winner, True)
    assert "reasoning_content=\"A\"" in part
    assert "reasoning=\"B\"" not in part


def test_flag_on_dual_field_empty_winner_falls_through_to_nothing():
    """Flag on: an empty-string reasoning_content takes precedence over a
    non-empty reasoning (mirroring _reasoning_of's `is not None` precedence),
    and because the winner is empty it carries no key material — the part is
    byte-identical to the message with no reasoning at all (the save side drops
    the empty winner and never falls through to the other field)."""
    dual = {"role": "assistant", "content": "b", "reasoning_content": "", "reasoning": "B"}
    bare = {"role": "assistant", "content": "b"}
    assert hs._message_part(dual, True) == hs._message_part(bare, True)


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_flag_on_empty_string_reasoning_treated_as_absent(field):
    """Flag on: an explicitly-present empty-string reasoning carries no key
    material — the part is byte-identical to the message without the field
    (the save side drops empty reasoning, so both sides must agree)."""
    with_empty = {"role": "assistant", "content": "b", field: ""}
    without = {"role": "assistant", "content": "b"}
    assert hs._message_part(with_empty, True) == hs._message_part(without, True)
    # A message with only an empty reasoning contributes no part at all.
    only_empty = {"role": "assistant", "content": None, field: ""}
    assert hs._message_part(only_empty, True) == ""
    # Same for a message that already takes the extra path (tool_calls).
    tc = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
        field: "",
    }
    tc_no_field = dict(tc)
    del tc_no_field[field]
    assert hs._message_part(tc, True) == hs._message_part(tc_no_field, True)
    # Full request values: with-empty == without-field.
    base = [{"role": "user", "content": "same content"}]
    assert hs.request_prefix_values(
        base + [with_empty], "m1", 100, True
    ) == hs.request_prefix_values(base + [without], "m1", 100, True)


def test_flag_on_plain_messages_unchanged():
    """Flag on: messages without reasoning fields keep the legacy parts, so
    non-reasoning conversations do not churn."""
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                }
            ],
        },
    ]
    assert hs.request_prefix_values(msgs, "m1", 100) == hs.request_prefix_values(
        msgs, "m1", 100, True
    )


@pytest.mark.asyncio
async def test_request_prefix_values_async_threads_flag():
    msgs = [{"role": "assistant", "content": "b", "reasoning_content": "R"}]
    off = await hs.request_prefix_values_async(msgs, "m1", 100)
    on = await hs.request_prefix_values_async(msgs, "m1", 100, True)
    assert off[1] != on[1]
    assert off[0] == hs.raw_prefix(msgs)
    assert on[0] == hs.raw_prefix(msgs, True)


# --- hashing: saved_conversation_values -------------------------------------


def test_saved_values_flag_on_includes_reasoning():
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = hs.saved_conversation_values(
        messages, "b", "m1", 100, True, "R"
    )
    expected = messages + [
        {"role": "assistant", "content": "b", "reasoning_content": "R"}
    ]
    assert prefix == hs.raw_prefix(expected, True)
    assert blocks == hs.block_hashes_from_text(prefix, 100)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1", True)


def test_saved_values_flag_off_ignores_reasoning():
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = hs.saved_conversation_values(
        messages, "b", "m1", 100, False, "R"
    )
    expected = messages + [{"role": "assistant", "content": "b"}]
    assert prefix == hs.raw_prefix(expected)
    assert blocks == hs.block_hashes_from_text(prefix, 100)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1")


def test_saved_values_flag_on_reasoning_only_response():
    """Flag on: an empty-content response with reasoning still extends the
    saved conversation (the reasoning trace is what must match)."""
    messages = [{"role": "user", "content": "a"}]
    prefix, _blocks, hashes = hs.saved_conversation_values(
        messages, "", "m1", 100, True, "R"
    )
    expected = messages + [
        {"role": "assistant", "content": "", "reasoning_content": "R"}
    ]
    assert prefix == hs.raw_prefix(expected, True)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1", True)


def test_saved_values_flag_on_preserves_reasoning_field_name():
    """Flag on: the saved assistant message carries the reasoning under the
    field name the backend used (not normalized to reasoning_content)."""
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = hs.saved_conversation_values(
        messages, "b", "m1", 100, True, "R", reasoning_field="reasoning"
    )
    expected = messages + [{"role": "assistant", "content": "b", "reasoning": "R"}]
    assert prefix == hs.raw_prefix(expected, True)
    assert blocks == hs.block_hashes_from_text(prefix, 100)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1", True)


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_saved_values_continuation_key_matches_echoed_field(field):
    """The client receives the backend body unchanged and echoes the reasoning
    field back on the continuation request: the echoed request key must equal
    the saved conversation's last prefix hash for BOTH field variants."""
    messages = [{"role": "user", "content": "a"}]
    _prefix, _blocks, hashes = hs.saved_conversation_values(
        messages, "b", "m1", 100, True, "R", reasoning_field=field
    )
    echoed = messages + [{"role": "assistant", "content": "b", field: "R"}]
    _p, key, _b, req_hashes, _w = hs.request_prefix_values(echoed, "m1", 100, True)
    assert hashes[-1] == key
    assert req_hashes[-1] == key


@pytest.mark.parametrize(
    "dual",
    [
        # diff-value: the backend emitted both fields with different traces.
        {"reasoning_content": "A", "reasoning": "B"},
        # same-value: the backend mirrored one trace into both fields.
        {"reasoning_content": "A", "reasoning": "A"},
    ],
    ids=["diff-value", "same-value"],
)
def test_saved_values_dual_field_continuation_matches_echo(dual):
    """A dual-field response (both reasoning_content and reasoning present):
    the save side stores only the precedence winner (reasoning_content), and
    the client echoes the WHOLE dual-field message back on the continuation
    request. The echoed request key must equal the saved meta's last prefix
    hash — the request side must apply the same single-field precedence."""
    messages = [{"role": "user", "content": "a"}]
    _prefix, _blocks, hashes = hs.saved_conversation_values(
        messages, "b", "m1", 100, True, dual["reasoning_content"]
    )
    echoed = messages + [
        {
            "role": "assistant",
            "content": "b",
            "reasoning_content": dual["reasoning_content"],
            "reasoning": dual["reasoning"],
        }
    ]
    _p, key, _b, req_hashes, _w = hs.request_prefix_values(echoed, "m1", 100, True)
    assert hashes[-1] == key
    assert req_hashes[-1] == key


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_saved_values_empty_reasoning_continuation_matches_echo(field):
    """A backend that emits an empty reasoning field (e.g. a Qwen3/llama.cpp
    non-thinking turn) saves a meta without the field; the client still
    echoes the empty field back on the continuation request. The echoed
    request key must equal the saved meta's last prefix hash ("" carries no
    key material on either side)."""
    messages = [{"role": "user", "content": "a"}]
    _prefix, _blocks, hashes = hs.saved_conversation_values(
        messages, "b", "m1", 100, True, "", reasoning_field=field
    )
    echoed = messages + [{"role": "assistant", "content": "b", field: ""}]
    _p, key, _b, req_hashes, _w = hs.request_prefix_values(echoed, "m1", 100, True)
    assert hashes[-1] == key
    assert req_hashes[-1] == key


# --- chat_flow: response extraction ------------------------------------------


def test_assistant_content_returns_reasoning():
    out = {
        "choices": [
            {"message": {"content": "hi", "reasoning_content": "R"}}
        ]
    }
    assert chat_flow._assistant_content(out) == ("hi", "R", "reasoning_content")


def test_assistant_content_reasoning_field_fallback():
    out = {"choices": [{"message": {"content": "hi", "reasoning": "R"}}]}
    assert chat_flow._assistant_content(out) == ("hi", "R", "reasoning")


def test_assistant_content_reasoning_content_takes_precedence():
    out = {
        "choices": [
            {"message": {"content": "hi", "reasoning_content": "A", "reasoning": "B"}}
        ]
    }
    assert chat_flow._assistant_content(out) == ("hi", "A", "reasoning_content")


def test_assistant_content_no_reasoning():
    out = {"choices": [{"message": {"content": "hi"}}]}
    assert chat_flow._assistant_content(out) == ("hi", "", "reasoning_content")
    assert chat_flow._assistant_content({"choices": []}) == ("", "", "reasoning_content")


def test_append_stream_content_captures_reasoning():
    parts: list[str] = []
    reasoning: list[str] = []
    field = chat_flow._append_stream_content(
        'data: {"choices": [{"delta": {"reasoning_content": "Let me"}}]}',
        parts,
        reasoning,
    )
    assert field == "reasoning_content"
    chat_flow._append_stream_content(
        'data: {"choices": [{"delta": {"content": "he"}}]}', parts, reasoning
    )
    assert parts == ["he"]
    assert reasoning == ["Let me"]


def test_append_stream_content_captures_reasoning_field_variant():
    """A `reasoning`-field backend streams reasoning in delta.reasoning; it
    must be captured too (not just delta.reasoning_content)."""
    parts: list[str] = []
    reasoning: list[str] = []
    field = chat_flow._append_stream_content(
        'data: {"choices": [{"delta": {"reasoning": "Let me"}}]}',
        parts,
        reasoning,
    )
    assert parts == []
    assert reasoning == ["Let me"]
    assert field == "reasoning"


def test_append_stream_content_reasoning_content_takes_precedence():
    """Parity with the non-stream path: reasoning_content wins over reasoning."""
    parts: list[str] = []
    reasoning: list[str] = []
    field = chat_flow._append_stream_content(
        'data: {"choices": [{"delta": {"reasoning_content": "A", "reasoning": "B"}}]}',
        parts,
        reasoning,
    )
    assert reasoning == ["A"]
    assert field == "reasoning_content"


def test_append_stream_content_reasoning_message_fallback():
    """A final message chunk (no delta) carries the reasoning: it must be
    captured via the message fallback, like content is."""
    parts: list[str] = []
    reasoning: list[str] = []
    field = chat_flow._append_stream_content(
        'data: {"choices": [{"message": {"content": "hi", "reasoning": "R"}}]}',
        parts,
        reasoning,
    )
    assert parts == ["hi"]
    assert reasoning == ["R"]
    assert field == "reasoning"


def test_append_stream_content_no_reasoning_returns_none():
    parts: list[str] = []
    reasoning: list[str] = []
    assert (
        chat_flow._append_stream_content(
            'data: {"choices": [{"delta": {"content": "he"}}]}', parts, reasoning
        )
        is None
    )
    assert reasoning == []


def test_append_stream_content_no_reasoning_sink():
    """Flag-off path: no sink given, reasoning deltas are dropped."""
    parts: list[str] = []
    chat_flow._append_stream_content(
        'data: {"choices": [{"delta": {"reasoning_content": "R", "content": "he"}}]}',
        parts,
    )
    assert parts == ["he"]


# --- chat_flow: _saved_conversation_values -----------------------------------


@pytest.mark.asyncio
async def test_saved_conversation_values_flag_on_includes_reasoning(monkeypatch):
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", True)
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = await chat_flow._saved_conversation_values(
        messages, "b", "m1", "fp", ["fb"], ["fh"], "R"
    )
    expected = messages + [
        {"role": "assistant", "content": "b", "reasoning_content": "R"}
    ]
    assert prefix == hs.raw_prefix(expected, True)
    assert blocks == hs.block_hashes_from_text(prefix, hs.WORDS_PER_BLOCK)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1", True)


@pytest.mark.asyncio
async def test_saved_conversation_values_flag_off_ignores_reasoning(monkeypatch):
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", False)
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = await chat_flow._saved_conversation_values(
        messages, "b", "m1", "fp", ["fb"], ["fh"], "R"
    )
    expected = messages + [{"role": "assistant", "content": "b"}]
    assert prefix == hs.raw_prefix(expected)
    assert blocks == hs.block_hashes_from_text(prefix, hs.WORDS_PER_BLOCK)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1")


@pytest.mark.asyncio
async def test_saved_conversation_values_flag_on_reasoning_only_no_fallback(
    monkeypatch,
):
    """Flag on: empty content + reasoning must not take the prompt-only
    fallback (the reasoning trace is what continuation matching needs)."""
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", True)
    messages = [{"role": "user", "content": "a"}]
    fallback = ("fp", ["fb"], ["fh"])
    result = await chat_flow._saved_conversation_values(
        messages, "", "m1", *fallback, "R"
    )
    assert result != fallback
    expected = messages + [
        {"role": "assistant", "content": "", "reasoning_content": "R"}
    ]
    assert result[0] == hs.raw_prefix(expected, True)


@pytest.mark.asyncio
async def test_saved_conversation_values_flag_off_empty_response_fallback(
    monkeypatch,
):
    """Flag off: empty content + reasoning takes the prompt-only fallback
    (legacy behavior)."""
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", False)
    messages = [{"role": "user", "content": "a"}]
    fallback = ("fp", ["fb"], ["fh"])
    result = await chat_flow._saved_conversation_values(
        messages, "", "m1", *fallback, "R"
    )
    assert result == fallback


@pytest.mark.asyncio
async def test_saved_conversation_values_flag_on_empty_reasoning_echo_matches(
    monkeypatch,
):
    """Backend body with reasoning_content: "" (non-thinking turn): the save
    path drops the empty trace, and the continuation request that echoes the
    empty field back must still match the saved last prefix hash."""
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", True)
    out = {"choices": [{"message": {"content": "b", "reasoning_content": ""}}]}
    content, reasoning, field = chat_flow._assistant_content(out)
    assert (content, reasoning, field) == ("b", "", "reasoning_content")
    messages = [{"role": "user", "content": "a"}]
    _prefix, _blocks, hashes = await chat_flow._saved_conversation_values(
        messages, content, "m1", "fp", ["fb"], ["fh"], reasoning, field
    )
    echoed = messages + [{"role": "assistant", "content": "b", "reasoning_content": ""}]
    _p, key, _b, req_hashes, _w = hs.request_prefix_values(echoed, "m1", 100, True)
    assert hashes[-1] == key
    assert req_hashes[-1] == key


@pytest.mark.asyncio
async def test_saved_conversation_values_flag_on_reasoning_field_variant(
    monkeypatch,
):
    """The reasoning field name the backend used is threaded into the saved
    conversation (the echoed continuation key must match)."""
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", True)
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = await chat_flow._saved_conversation_values(
        messages, "b", "m1", "fp", ["fb"], ["fh"], "R", "reasoning"
    )
    expected = messages + [{"role": "assistant", "content": "b", "reasoning": "R"}]
    assert prefix == hs.raw_prefix(expected, True)
    assert blocks == hs.block_hashes_from_text(prefix, hs.WORDS_PER_BLOCK)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1", True)


# --- chat_flow: end-to-end key computation -----------------------------------


async def _chat_keys(
    sm, monkeypatch, flag: bool, reasonings: list[str]
) -> tuple[list[str], list[bool]]:
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", flag)
    keys: list[str] = []
    flags: list[bool] = []
    real = hs.request_prefix_values_async

    async def spy(messages, model_id, wpb, include_reasoning=False):
        result = await real(messages, model_id, wpb, include_reasoning)
        keys.append(result[1])
        flags.append(include_reasoning)
        return result

    monkeypatch.setattr(hs, "request_prefix_values_async", spy)
    for rc in reasonings:
        data = {
            "messages": [
                {"role": "user", "content": "same content", "reasoning_content": rc}
            ],
            "stream": False,
        }
        await app_module.chat(FakeRequest(data))
    return keys, flags


@pytest.mark.asyncio
async def test_chat_flow_flag_on_distinct_reasoning_distinct_keys(sm, monkeypatch):
    keys, flags = await _chat_keys(sm, monkeypatch, True, ["think A", "think B"])
    assert flags == [True, True], "the flag must be threaded into the key"
    assert len(keys) == 2
    assert keys[0] != keys[1]


@pytest.mark.asyncio
async def test_chat_flow_flag_off_same_key_for_different_reasoning(
    sm, monkeypatch
):
    keys, flags = await _chat_keys(sm, monkeypatch, False, ["think A", "think B"])
    assert flags == [False, False]
    assert keys[0] == keys[1], "flag off: reasoning must not affect the key"


# --- chat_flow: stream save ---------------------------------------------------


async def _stream_save(sm, monkeypatch, flag: bool, reasoning_field: str = "reasoning_content"):
    save_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(chat_flow, "_save_and_write_meta", save_mock)
    monkeypatch.setattr(chat_flow, "REASONING_IN_KEY", flag)
    g = (0, "model", 0)
    lock = sm._lock_for(g)
    await lock.acquire()
    chunks = [
        (
            b'data: {"choices": [{"delta": {"'
            + reasoning_field.encode()
            + b'": "Let me think."}}]}\n\n'
        ),
        b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\ndata: [DONE]\n\n',
    ]
    resp = FakeResp(chunks)
    messages = [{"role": "user", "content": "hi"}]
    gen = await chat_flow.start_stream_task(
        resp, g, "k" * 16, "prefix", ["b"], "model", sm, is_big=True, messages=messages
    )
    _ = [c async for c in gen]
    await asyncio.sleep(0.2)
    save_mock.assert_awaited_once()
    return save_mock.await_args.args


@pytest.mark.asyncio
async def test_stream_flag_on_saved_prefix_includes_reasoning(sm, monkeypatch):
    args = await _stream_save(sm, monkeypatch, True)
    expected = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello", "reasoning_content": "Let me think."},
    ]
    assert args[4] == hs.raw_prefix(expected, True)
    assert args[5] == hs.block_hashes_from_text(args[4], hs.WORDS_PER_BLOCK)
    assert args[8] == hs.prefix_hashes_from_messages(expected, "model", True)


@pytest.mark.asyncio
async def test_stream_flag_off_saved_prefix_drops_reasoning(sm, monkeypatch):
    args = await _stream_save(sm, monkeypatch, False)
    expected = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    assert args[4] == hs.raw_prefix(expected)
    assert args[5] == hs.block_hashes_from_text(args[4], hs.WORDS_PER_BLOCK)
    assert args[8] == hs.prefix_hashes_from_messages(expected, "model")


@pytest.mark.asyncio
async def test_stream_flag_on_reasoning_field_backend_saved_prefix_matches(
    sm, monkeypatch
):
    """A `reasoning`-field backend: the captured reasoning must be saved under
    `reasoning` (the field the client received and echoes back), so the
    continuation request key matches the saved meta."""
    args = await _stream_save(sm, monkeypatch, True, "reasoning")
    expected = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello", "reasoning": "Let me think."},
    ]
    assert args[4] == hs.raw_prefix(expected, True)
    assert args[5] == hs.block_hashes_from_text(args[4], hs.WORDS_PER_BLOCK)
    assert args[8] == hs.prefix_hashes_from_messages(expected, "model", True)
