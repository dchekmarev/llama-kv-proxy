# tests/test_request_prefix_values.py

"""M-1: request_prefix_values must produce byte-identical values to the
previous per-value computation (raw_prefix / prefix_key_sha256 /
block_hashes_from_text / words_from_text / prefix_hashes_from_messages), so
existing cache entries keep matching."""

import hashlib

import hashing as hs


def _old_prefix_hashes(messages: list[dict], model_id: str) -> list[str]:
    """The previous O(n^2) implementation, kept here as the reference."""
    hashes: list[str] = []
    for k in range(1, len(messages or []) + 1):
        h = hs.prefix_key_sha256(model_id + "\n" + hs.raw_prefix(messages[:k]))
        if not hashes or hashes[-1] != h:
            hashes.append(h)
    return hashes


def _msgs(contents: list[tuple[str, str]]) -> list[dict]:
    return [{"role": role, "content": c} for role, c in contents]


def test_matches_previous_implementation_plain():
    msgs = _msgs(
        [
            ("system", "You are a helpful assistant."),
            ("user", "Tell me about the history of computing, in detail."),
            ("assistant", "Computing began with mechanical calculators..."),
            ("user", "And what about the transistor era?"),
            ("assistant", "The transistor, invented in 1947..."),
        ]
    )
    prefix, key, blocks, hashes, n_words = hs.request_prefix_values(msgs, "m1", 100)

    old_prefix = hs.raw_prefix(msgs)
    assert prefix == old_prefix
    assert key == hs.prefix_key_sha256("m1\n" + old_prefix)
    assert blocks == hs.block_hashes_from_text(old_prefix, 100)
    assert hashes == _old_prefix_hashes(msgs, "m1")
    assert n_words == len(hs.words_from_text(old_prefix))
    # the last prefix hash is the cache key
    assert hashes[-1] == key


def test_matches_previous_implementation_empty_and_cjk():
    msgs = _msgs(
        [
            ("user", ""),  # empty message: skipped, no duplicate hash
            ("user", "你好世界这是一段中文文本"),
            ("assistant", ""),
            ("user", "hello 世界 ok 123"),
        ]
    )
    prefix, key, blocks, hashes, n_words = hs.request_prefix_values(msgs, "m2", 3)

    old_prefix = hs.raw_prefix(msgs)
    assert prefix == old_prefix
    assert key == hs.prefix_key_sha256("m2\n" + old_prefix)
    assert blocks == hs.block_hashes_from_text(old_prefix, 3)
    assert hashes == _old_prefix_hashes(msgs, "m2")
    assert n_words == len(hs.words_from_text(old_prefix))
    assert hashes[-1] == key


def test_matches_previous_implementation_tool_calls():
    msgs: list[dict] = [
        {"role": "user", "content": "weather in paris"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny, 22C"},
    ]
    prefix, key, blocks, hashes, n_words = hs.request_prefix_values(msgs, "m3", 100)

    old_prefix = hs.raw_prefix(msgs)
    assert prefix == old_prefix
    assert key == hs.prefix_key_sha256("m3\n" + old_prefix)
    assert blocks == hs.block_hashes_from_text(old_prefix, 100)
    assert hashes == _old_prefix_hashes(msgs, "m3")
    assert n_words == len(hs.words_from_text(old_prefix))


def test_matches_previous_implementation_large():
    # A large conversation: the incremental hashes must equal the O(n^2)
    # reference for every k, not just the last one.
    msgs = _msgs(
        [("user" if i % 2 == 0 else "assistant", f"message number {i} with some text")
         for i in range(40)]
    )
    prefix, key, blocks, hashes, n_words = hs.request_prefix_values(msgs, "m1", 100)

    assert hashes == _old_prefix_hashes(msgs, "m1")
    assert hashes[-1] == key
    assert blocks == hs.block_hashes_from_text(prefix, 100)
    assert n_words == len(hs.words_from_text(prefix))


def test_empty_messages():
    prefix, key, blocks, hashes, n_words = hs.request_prefix_values([], "m1", 100)
    assert prefix == ""
    assert key == hs.prefix_key_sha256("m1\n")
    assert blocks == []
    assert hashes == []
    assert n_words == 0


def test_message_part_computed_once_per_message(monkeypatch):
    # M-1: request_prefix_values must reuse the per-message normalization
    # (content strip, tool_calls JSON round-trip) instead of re-normalizing
    # each message for the incremental prefix hashes.
    calls: list[dict] = []
    real_part = hs._message_part

    def counting_part(msg: dict, include_reasoning: bool = False) -> str:
        calls.append(msg)
        return real_part(msg, include_reasoning)

    monkeypatch.setattr(hs, "_message_part", counting_part)
    msgs: list[dict] = [
        {"role": "user", "content": "  weather in paris  "},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "sunny, 22C"},
    ]
    prefix, key, _blocks, hashes, _n_words = hs.request_prefix_values(msgs, "m1", 100)
    assert len(calls) == len(msgs), "each message must be normalized exactly once"
    assert prefix == hs.raw_prefix(msgs)
    assert hashes == _old_prefix_hashes(msgs, "m1")
    assert hashes[-1] == key


def test_block_hashes_are_sha256_of_word_blocks():
    # Guard the block format explicitly: sha256 of wpb words joined by space.
    prefix, _key, blocks, _hashes, n_words = hs.request_prefix_values(
        _msgs([("user", "a b c d e")]), "m1", 2
    )
    words = hs.words_from_text(prefix)
    assert n_words == len(words) == 6
    expected = [
        hashlib.sha256(" ".join(words[i : i + 2]).encode("utf-8")).hexdigest()
        for i in range(0, len(words), 2)
    ]
    assert blocks == expected
