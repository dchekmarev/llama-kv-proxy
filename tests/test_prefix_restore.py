# tests/test_prefix_restore.py

"""Two-tier restore candidate selection:

1. Tier 1: precise message-boundary prefix match (metas with prefix_hashes).
   Longest common prefix first, smallest size on ties.
2. Tier 2: fallback to block-based LCP when Tier 1 finds nothing above the
   threshold. Old metas (blocks only, no prefix_hashes) stay usable here.
"""

import pytest

import hashing as hs


def _write_prefix(key, prefix_hashes, bin_size=None):
    hs.write_meta(key, "p", [], 100, "m1", prefix_hashes=prefix_hashes, bin_size=bin_size)


def _write_blocks(key, blocks):
    hs.write_meta(key, "p", blocks, 100, "m1")


def test_tier1_prefix_match(meta_dir):
    """A meta whose conversation extends the request is found by prefix hashes."""
    req_hashes = ["h_a", "h_ab"]
    _write_prefix("meta_abc", ["h_a", "h_ab", "h_abc"], bin_size=300)
    cand = hs.find_best_restore_candidate(req_hashes, [], 100, 0.0, "m1")
    assert cand is not None
    key, ratio = cand
    assert key == "meta_abc"
    assert ratio == pytest.approx(1.0)


def test_tier1_longest_prefix_then_smallest_size(meta_dir):
    """Two metas share the request's prefix; the smallest size wins."""
    req_hashes = ["h_a", "h_ab"]
    _write_prefix("big", ["h_a", "h_ab", "h_abc_big"], bin_size=500)
    _write_prefix("small", ["h_a", "h_ab", "h_abc_small"], bin_size=100)
    cand = hs.find_best_restore_candidate(req_hashes, [], 100, 0.0, "m1")
    assert cand is not None
    assert cand[0] == "small"


def test_tier1_beats_tier2(meta_dir):
    """When both a prefix match and a block match exist, the prefix match wins."""
    req_hashes = ["h_a", "h_ab"]
    req_blocks = ["b1", "b2"]
    _write_prefix("prefix_meta", ["h_a", "h_ab"], bin_size=100)
    _write_blocks("block_meta", ["b1", "b2", "b3"])
    cand = hs.find_best_restore_candidate(req_hashes, req_blocks, 100, 0.0, "m1")
    assert cand is not None
    assert cand[0] == "prefix_meta"


def test_tier2_fallback_when_no_prefix_match(meta_dir):
    """No prefix_hashes on any meta -> fall back to block LCP."""
    req_blocks = ["b1", "b2", "b3"]
    _write_blocks("short", ["b1"])
    _write_blocks("long", ["b1", "b2"])
    cand = hs.find_best_restore_candidate([], req_blocks, 100, 0.0, "m1")
    assert cand is not None
    assert cand[0] == "long"


def test_old_metas_match_via_tier2(meta_dir):
    """Old metas (blocks only, no prefix_hashes) are matched by the LCP fallback."""
    req_blocks = ["b1", "b2"]
    _write_blocks("old", ["b1", "b2", "b3"])
    cand = hs.find_best_restore_candidate([], req_blocks, 100, 0.0, "m1")
    assert cand is not None
    assert cand[0] == "old"


def test_tier1_threshold_rejects_short_match(meta_dir):
    """A prefix match below the threshold is rejected (no block fallback)."""
    req_hashes = ["h_a", "h_ab", "h_abc", "h_abcd"]
    _write_prefix("short", ["h_a", "h_ab"], bin_size=100)
    cand = hs.find_best_restore_candidate(req_hashes, [], 100, 0.6, "m1")
    assert cand is None


def test_model_filter(meta_dir):
    """Only metas of the same model are considered."""
    req_hashes = ["h_a"]
    hs.write_meta("other", "p", [], 100, "m2", prefix_hashes=["h_a"])
    _write_prefix("same", ["h_a"], bin_size=100)
    cand = hs.find_best_restore_candidate(req_hashes, [], 100, 0.0, "m1")
    assert cand is not None
    assert cand[0] == "same"


def test_tier1_uses_saved_prefix_hashes(meta_dir):
    """A continuation request matches the response-extended saved hashes even
    when the prompt-only hashes are below the threshold."""
    req_hashes = ["h_a", "h_ab", "h_abc", "h_abcd"]
    hs.write_meta(
        "meta_saved",
        "p",
        [],
        100,
        "m1",
        prefix_hashes=["h_a", "h_ab"],
        saved_prefix_hashes=["h_a", "h_ab", "h_abc"],
        bin_size=100,
    )
    cand = hs.find_best_restore_candidate(req_hashes, [], 100, 0.6, "m1")
    assert cand is not None
    assert cand[0] == "meta_saved"
    assert cand[1] == pytest.approx(0.75)


def test_tier1_saved_hashes_beat_prompt_only_hashes(meta_dir):
    req_hashes = ["h_a", "h_ab", "h_abc", "h_abcd"]
    hs.write_meta(
        "prompt_only",
        "p",
        [],
        100,
        "m1",
        prefix_hashes=["h_a", "h_ab"],
        bin_size=100,
    )
    hs.write_meta(
        "saved",
        "p",
        [],
        100,
        "m1",
        prefix_hashes=["h_a", "h_ab"],
        saved_prefix_hashes=["h_a", "h_ab", "h_abc"],
        bin_size=200,
    )
    cand = hs.find_best_restore_candidate(req_hashes, [], 100, 0.6, "m1")
    assert cand is not None
    assert cand[0] == "saved"
