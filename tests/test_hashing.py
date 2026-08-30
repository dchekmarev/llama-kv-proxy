# tests/test_hashing.py

"""P1-3: restore candidates must be ranked by the fraction of the REQUEST
covered by the cache (lcp / len(req_blocks)), not by the fraction of the
shorter sequence — otherwise a short fully-matching candidate (old ratio 1.0)
always evicts a longer, more useful one."""

import glob
import json
import os

import pytest

import hashing as hs


def _blocks(n, seed="x"):
    return [f"{seed}{i}" for i in range(n)]


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "META_DIR", str(tmp_path))
    return tmp_path


def _write(key, blocks):
    hs.write_meta(key, "p", blocks, 100, "m1")


def test_long_partial_candidate_beats_short_full_candidate(meta_dir):
    """1-block candidate (old ratio 1.0) vs 50-block candidate on a
    100-block request: the longer one must win."""
    req = _blocks(100, "r")
    _write("short", req[:1])
    _write("long", req[:50])

    cand = hs.find_best_restore_candidate([], req, 100, 0.005, "m1")

    assert cand is not None
    key, ratio = cand
    assert key == "long", "the longer, more useful candidate must win"
    assert ratio == pytest.approx(0.5)


def test_short_full_candidate_rejected_at_default_threshold(meta_dir):
    """A candidate covering 1% of the request must not pass LCP_TH=0.6."""
    req = _blocks(100, "r")
    _write("short", req[:1])

    cand = hs.find_best_restore_candidate([], req, 100, 0.6, "m1")

    assert cand is None


def test_equal_length_behavior_unchanged(meta_dir):
    """Equal-length candidate: ratio is lcp/len as before the fix."""
    req = _blocks(100, "r")
    cand_blocks = req[:70] + _blocks(30, "z")  # 70 match, then diverge
    _write("same_len", cand_blocks)

    cand = hs.find_best_restore_candidate([], req, 100, 0.6, "m1")

    assert cand is not None
    key, ratio = cand
    assert key == "same_len"
    assert ratio == pytest.approx(0.7)


def test_empty_request_returns_none(meta_dir):
    """No request blocks: no candidate, no division by zero."""
    _write("some", _blocks(10))

    cand = hs.find_best_restore_candidate([], [], 100, 0.6, "m1")

    assert cand is None


def test_scan_all_meta_skips_files_deleted_during_scan(meta_dir, monkeypatch):
    """H2: a meta file removed between glob and getmtime must not crash the
    scan — the surviving files are still returned, the vanished one is skipped."""
    _write("real", _blocks(3))
    real_files = glob.glob(os.path.join(str(meta_dir), "*.meta.json"))
    phantom = os.path.join(str(meta_dir), "phantom.meta.json")
    monkeypatch.setattr(hs.glob, "glob", lambda pattern: real_files + [phantom])

    metas = hs.scan_all_meta()

    keys = {m.get("key") for m in metas}
    assert "real" in keys
    assert "phantom" not in keys


def test_candidate_size_prefers_bin_size():
    """L1: bin_size (bytes) takes precedence over the char-based estimates."""
    assert (
        hs._candidate_size({"bin_size": 1000, "prefix_bytes": 500, "prefix_len": 100})
        == 1000
    )


def test_candidate_size_uses_prefix_bytes_when_no_bin():
    """L1: without bin_size, the byte-length estimate is used (not char count)."""
    assert hs._candidate_size({"prefix_bytes": 500, "prefix_len": 100}) == 500


def test_candidate_size_falls_back_to_prefix_len():
    """L1: old metas without prefix_bytes fall back to the char count."""
    assert hs._candidate_size({"prefix_len": 100}) == 100


def test_write_meta_stores_prefix_bytes(meta_dir):
    """L1: write_meta records the UTF-8 byte length so the tie-break compares
    bytes, not chars (a CJK prefix has more bytes than chars)."""
    key = "a" * 64
    hs.write_meta(key, "你好世界", ["b"], 100, "m1")
    path = os.path.join(str(meta_dir), f"{key}.meta.json")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["prefix_len"] == 4
    assert meta["prefix_bytes"] == len("你好世界".encode())


def test_write_meta_stores_saved_prefix_hashes(meta_dir):
    key = "a" * 64
    hs.write_meta(
        key,
        "p",
        ["b"],
        100,
        "m1",
        prefix_hashes=["h_prompt"],
        saved_prefix_hashes=["h_prompt", "h_response"],
    )
    path = os.path.join(str(meta_dir), f"{key}.meta.json")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["prefix_hashes"] == ["h_prompt"]
    assert meta["saved_prefix_hashes"] == ["h_prompt", "h_response"]


def test_saved_conversation_values_appends_assistant_response():
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = hs.saved_conversation_values(messages, "b", "m1", 100)
    expected = messages + [{"role": "assistant", "content": "b"}]
    assert prefix == hs.raw_prefix(expected)
    assert blocks == hs.block_hashes_from_text(prefix, 100)
    assert hashes == hs.prefix_hashes_from_messages(expected, "m1")


def test_saved_conversation_values_empty_response_keeps_prompt():
    messages = [{"role": "user", "content": "a"}]
    prefix, blocks, hashes = hs.saved_conversation_values(messages, "", "m1", 100)
    assert prefix == hs.raw_prefix(messages)
    assert blocks == hs.block_hashes_from_text(prefix, 100)
    assert hashes == hs.prefix_hashes_from_messages(messages, "m1")
