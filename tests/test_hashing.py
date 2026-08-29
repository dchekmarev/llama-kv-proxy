# tests/test_hashing.py

"""P1-3: restore candidates must be ranked by the fraction of the REQUEST
covered by the cache (lcp / len(req_blocks)), not by the fraction of the
shorter sequence — otherwise a short fully-matching candidate (old ratio 1.0)
always evicts a longer, more useful one."""

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

    cand = hs.find_best_restore_candidate(req, 100, 0.005, "m1")

    assert cand is not None
    key, ratio = cand
    assert key == "long", "the longer, more useful candidate must win"
    assert ratio == pytest.approx(0.5)


def test_short_full_candidate_rejected_at_default_threshold(meta_dir):
    """A candidate covering 1% of the request must not pass LCP_TH=0.6."""
    req = _blocks(100, "r")
    _write("short", req[:1])

    cand = hs.find_best_restore_candidate(req, 100, 0.6, "m1")

    assert cand is None


def test_equal_length_behavior_unchanged(meta_dir):
    """Equal-length candidate: ratio is lcp/len as before the fix."""
    req = _blocks(100, "r")
    cand_blocks = req[:70] + _blocks(30, "z")  # 70 match, then diverge
    _write("same_len", cand_blocks)

    cand = hs.find_best_restore_candidate(req, 100, 0.6, "m1")

    assert cand is not None
    key, ratio = cand
    assert key == "same_len"
    assert ratio == pytest.approx(0.7)


def test_empty_request_returns_none(meta_dir):
    """No request blocks: no candidate, no division by zero."""
    _write("some", _blocks(10))

    cand = hs.find_best_restore_candidate([], 100, 0.6, "m1")

    assert cand is None
