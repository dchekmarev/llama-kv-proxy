# tests/test_prefix_hashes.py

"""Per-message prefix hashes: H(messages[:k]) for k=1..n, where
H(x) = sha256(model_id + "\\n" + raw_prefix(x)). The last hash equals the
cache key for the full conversation, so a meta is findable by any prefix."""

import hashing as hs


def _msgs(*contents, role="user"):
    return [{"role": role, "content": c} for c in contents]


def test_last_hash_equals_key():
    msgs = _msgs("hello", "world")
    hashes = hs.prefix_hashes_from_messages(msgs, "m1")
    key = hs.prefix_key_sha256("m1\n" + hs.raw_prefix(msgs))
    assert hashes[-1] == key


def test_one_hash_per_nonempty_message():
    msgs = _msgs("a", "b", "c")
    hashes = hs.prefix_hashes_from_messages(msgs, "m1")
    assert len(hashes) == 3
    # each hash is the key for the prefix up to that message
    assert hashes[0] == hs.prefix_key_sha256("m1\n" + hs.raw_prefix(msgs[:1]))
    assert hashes[1] == hs.prefix_key_sha256("m1\n" + hs.raw_prefix(msgs[:2]))
    assert hashes[2] == hs.prefix_key_sha256("m1\n" + hs.raw_prefix(msgs[:3]))


def test_empty_message_produces_no_duplicate_hash():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": ""},  # empty -> skipped, no new hash
        {"role": "user", "content": "b"},
    ]
    hashes = hs.prefix_hashes_from_messages(msgs, "m1")
    assert len(hashes) == 2
    # still ends with the full-conversation key
    assert hashes[-1] == hs.prefix_key_sha256("m1\n" + hs.raw_prefix(msgs))


def test_roles_are_part_of_the_prefix():
    m_sys = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "y"},
    ]
    m_usr = [
        {"role": "user", "content": "x"},
        {"role": "user", "content": "y"},
    ]
    h_sys = hs.prefix_hashes_from_messages(m_sys, "m1")
    h_usr = hs.prefix_hashes_from_messages(m_usr, "m1")
    # same content, different role -> different first prefix hash
    assert h_sys[0] != h_usr[0]


def test_model_id_scopes_the_hashes():
    msgs = _msgs("a")
    h1 = hs.prefix_hashes_from_messages(msgs, "m1")
    h2 = hs.prefix_hashes_from_messages(msgs, "m2")
    assert h1[0] != h2[0]


def test_single_message():
    msgs = _msgs("only")
    hashes = hs.prefix_hashes_from_messages(msgs, "m1")
    assert len(hashes) == 1
    assert hashes[0] == hs.prefix_key_sha256("m1\n" + hs.raw_prefix(msgs))
