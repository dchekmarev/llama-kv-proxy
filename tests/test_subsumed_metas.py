# tests/test_subsumed_metas.py

"""delete_subsumed_metas: remove metas whose FULL conversation is a strict
prefix of the new conversation (the new one is a continuation of the old).

A meta M is subsumed when M.key (its full-conversation hash) appears among the
new conversation's prefix hashes. M.key always equals M's last prefix hash.
"""

import pytest

import hashing as hs


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "META_DIR", str(tmp_path))
    return tmp_path


def _write(key, prefix_hashes, model_id="m1"):
    # A meta's key is its full-conversation hash == the last prefix hash.
    assert key == prefix_hashes[-1]
    hs.write_meta(key, "p", [], 100, model_id, prefix_hashes=prefix_hashes)


def test_deletes_strict_prefix(meta_dir):
    _write("h_ab", ["h_a", "h_ab"])  # conv [a, b]
    _write("h_abc", ["h_a", "h_ab", "h_abc"])  # conv [a, b, c]
    deleted = hs.delete_subsumed_metas("h_abc", ["h_a", "h_ab", "h_abc"], "m1")
    assert deleted == ["h_ab"]


def test_keeps_non_prefix(meta_dir):
    """A branching meta (shares a prefix but diverges) is kept."""
    _write("h_abx", ["h_a", "h_ab", "h_abx"])  # conv [a, b, x]
    _write("h_abc", ["h_a", "h_ab", "h_abc"])  # conv [a, b, c]
    deleted = hs.delete_subsumed_metas("h_abc", ["h_a", "h_ab", "h_abc"], "m1")
    assert deleted == []


def test_keeps_self(meta_dir):
    _write("h_abc", ["h_a", "h_ab", "h_abc"])
    deleted = hs.delete_subsumed_metas("h_abc", ["h_a", "h_ab", "h_abc"], "m1")
    assert deleted == []


def test_model_filter(meta_dir):
    """Metas of other models are not touched."""
    _write("h_ab", ["h_a", "h_ab"], model_id="m2")
    _write("h_abc", ["h_a", "h_ab", "h_abc"], model_id="m1")
    deleted = hs.delete_subsumed_metas("h_abc", ["h_a", "h_ab", "h_abc"], "m1")
    assert deleted == []


def test_deletes_all_strict_prefixes(meta_dir):
    _write("h_a", ["h_a"])
    _write("h_ab", ["h_a", "h_ab"])
    _write("h_abc", ["h_a", "h_ab", "h_abc"])
    deleted = hs.delete_subsumed_metas("h_abc", ["h_a", "h_ab", "h_abc"], "m1")
    assert sorted(deleted) == ["h_a", "h_ab"]


def test_empty_prefix_hashes_delete_nothing(meta_dir):
    _write("h_ab", ["h_a", "h_ab"])
    deleted = hs.delete_subsumed_metas("h_ab", [], "m1")
    assert deleted == []
