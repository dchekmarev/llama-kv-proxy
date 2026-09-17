# tests/test_meta_index.py

"""In-memory restore index (B+A): unit tests for MetaIndex plus integration
tests for the flag-gated search path, live index updates on write/delete,
validate-on-hit, reconcile, startup rebuild, and the app reconcile loop."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import hashing as hs
import meta_index
import promstats

# ---- fixtures / helpers -----------------------------------------------------

def _touch(meta_dir, key):
    """Create an empty {key}.meta.json so validate-on-hit passes."""
    with open(os.path.join(str(meta_dir), f"{key}.meta.json"), "w") as f:
        f.write("{}")


@pytest.fixture
def fresh_index(monkeypatch):
    """A clean index and tier counters (flag left at its default, off)."""
    monkeypatch.setattr(hs, "_index", meta_index.MetaIndex())
    promstats.reset()
    yield hs._index


@pytest.fixture
def index_on(fresh_index, monkeypatch):
    """Enable the index flag with a fresh index."""
    monkeypatch.setattr(hs, "META_INDEX_ENABLED", True)
    yield fresh_index


# ---- _hash_key --------------------------------------------------------------

def test_hash_key_real_hex_is_raw_bytes():
    h = "ab" * 32
    assert meta_index._hash_key(h) == bytes.fromhex(h)
    assert len(meta_index._hash_key(h)) == 32


def test_hash_key_synthetic_falls_back_to_utf8():
    assert meta_index._hash_key("h_a") == b"h_a"


# ---- add / search (unit) ----------------------------------------------------

def test_search_hit_and_ratio(meta_dir):
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "k1")
    idx.add("k1", "m1", 100, ["h1", "h2", "h3"])
    assert idx.search(["h1", "h2", "h3", "h4"], 0.5, "m1", str(meta_dir)) == ("k1", 3 / 4)


def test_search_longest_prefix_wins(meta_dir):
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "short")
    _touch(meta_dir, "long")
    idx.add("short", "m1", 100, ["h1", "h2"])
    idx.add("long", "m1", 200, ["h1", "h2", "h3", "h4"])
    assert idx.search(["h1", "h2", "h3", "h4", "h5"], 0.5, "m1", str(meta_dir)) == (
        "long",
        4 / 5,
    )


def test_search_below_threshold_none(meta_dir):
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "k1")
    idx.add("k1", "m1", 100, ["h1"])
    # ratio 1/4 = 0.25 < 0.5
    assert idx.search(["h1", "x2", "x3", "x4"], 0.5, "m1", str(meta_dir)) is None


def test_search_model_mismatch_none(meta_dir):
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "k1")
    idx.add("k1", "m1", 100, ["h1", "h2"])
    assert idx.search(["h1", "h2"], 0.5, "m2", str(meta_dir)) is None


def test_search_empty_index_none(meta_dir):
    idx = meta_index.MetaIndex()
    assert idx.search(["h1"], 0.5, "m1", str(meta_dir)) is None
    assert idx.search([], 0.5, "m1", str(meta_dir)) is None


def test_search_tie_break_smallest_size(meta_dir):
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "big")
    _touch(meta_dir, "small")
    # Same 2-message prefix (equal LCP for a request matching up to h2); the
    # smallest cache wins the tie.
    idx.add("big", "m1", 500, ["h1", "h2", "h3b"])
    idx.add("small", "m1", 100, ["h1", "h2", "h3s"])
    assert idx.search(["h1", "h2"], 0.5, "m1", str(meta_dir)) == ("small", 1.0)


# ---- remove (unit) ----------------------------------------------------------

def test_remove_drops_key(meta_dir):
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "k1")
    idx.add("k1", "m1", 100, ["h1", "h2"])
    idx.remove("k1")
    assert idx.is_empty()
    assert idx.search(["h1", "h2"], 0.5, "m1", str(meta_dir)) is None


def test_remove_repoints_shared_hash_to_survivor(meta_dir):
    """Removing the smaller owner of a shared hash must re-point it to the
    next-smallest survivor, not drop the hash (the regression the owner-sets
    fix)."""
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "a")
    _touch(meta_dir, "b")
    idx.add("a", "m1", 100, ["h1", "h2a"])
    idx.add("b", "m1", 200, ["h1", "h2b"])
    idx.remove("a")
    # h1 was owned by both; after removing a it must resolve to b.
    assert idx.search(["h1"], 0.5, "m1", str(meta_dir)) == ("b", 1.0)
    assert idx.search(["h1", "h2b", "h3"], 0.5, "m1", str(meta_dir)) == ("b", 2 / 3)


def test_remove_last_owner_deletes_hash(meta_dir):
    idx = meta_index.MetaIndex()
    _touch(meta_dir, "a")
    idx.add("a", "m1", 100, ["h1"])
    idx.remove("a")
    assert idx.stats()["index_hashes"] == 0


# ---- reconcile / clear / rebuild / stats (unit) -----------------------------

def test_reconcile_keys_drops_ghosts():
    idx = meta_index.MetaIndex()
    idx.add("a", "m1", 100, ["h1"])
    idx.add("b", "m1", 100, ["h2"])
    dropped = idx.reconcile_keys({"b"})
    assert dropped == ["a"]
    assert len(idx) == 1
    assert idx.stats()["index_metas"] == 1


def test_clear_empties_index():
    idx = meta_index.MetaIndex()
    idx.add("a", "m1", 100, ["h1", "h2"])
    idx.clear()
    assert idx.is_empty()
    assert idx.stats() == {"index_metas": 0, "index_hashes": 0}


def test_rebuild_from_replaces_contents():
    idx = meta_index.MetaIndex()
    idx.add("stale", "m1", 100, ["old"])
    idx.rebuild_from(
        [
            {"key": "a", "model_id": "m1", "size": 10, "hashes": ["h1"]},
            {"key": "b", "model_id": "m1", "size": 20, "hashes": ["h2"]},
        ]
    )
    assert len(idx) == 2
    assert "stale" not in idx._meta


# ---- integration: parity with the on-disk scan ------------------------------

@pytest.mark.asyncio
async def test_parity_index_matches_disk(meta_dir, index_on):
    """For a single-wpb cache, the index search must return exactly what the
    on-disk two-tier scan returns (the index is a faithful tier-1 mirror)."""
    wpb, model, th = 100, "m1", 0.6
    metas = [
        ("h_a2", ["h_a1", "h_a2"], 300),
        ("h_b4", ["h_a1", "h_a2", "h_a3", "h_b4"], 400),
        ("h_d4", ["h_a1", "h_a2", "h_a3", "h_d4"], 350),
        ("h_c2", ["h_c1", "h_c2"], 100),
    ]
    for key, ph, size in metas:
        hs.write_meta(key, "p", [], wpb, model, prefix_hashes=ph, bin_size=size)

    requests = [
        ["h_a1", "h_a2", "h_a3", "h_b4", "h_x5"],  # continues b4
        ["h_a1", "h_a2", "h_a3", "h_d4", "h_y5"],  # continues d4
        ["h_c1", "h_c2", "h_z3"],  # continues c2
        ["h_a1", "h_a2", "h_a3"],  # exact a2 / prefix of b4,d4 -> tie -> a2
        ["h_q1", "h_q2"],  # unrelated -> none
    ]
    disk = {
        i: hs.find_best_restore_candidate(req, [], wpb, th, model)
        for i, req in enumerate(requests)
    }
    assert await hs.rebuild_index_async() == 4
    for i, req in enumerate(requests):
        idx = await hs.find_best_restore_candidate_async(req, [], wpb, th, model)
        assert idx == disk[i], f"request {i}: index={idx} disk={disk[i]}"


@pytest.mark.asyncio
async def test_parity_randomized(meta_dir, index_on):
    """Randomized parity: many random prefix-of-master metas and requests must
    give identical index vs disk results."""
    import random

    rng = random.Random(42)
    wpb, model, th = 100, "m1", 0.6
    master = [f"m{i}" for i in range(15)]
    for i in range(8):
        length = rng.randint(1, 15)
        hs.write_meta(
            f"meta{i}", "p", [], wpb, model, prefix_hashes=master[:length],
            bin_size=rng.randint(50, 500),
        )
    assert await hs.rebuild_index_async() == 8
    extras = [f"e{i}" for i in range(20)]
    for _ in range(40):
        length = rng.randint(1, 15)
        req = master[:length] + rng.sample(extras, rng.randint(0, 3))
        disk = hs.find_best_restore_candidate(req, [], wpb, th, model)
        idx = await hs.find_best_restore_candidate_async(req, [], wpb, th, model)
        assert idx == disk, f"req={req}: index={idx} disk={disk}"


# ---- integration: live index updates ----------------------------------------

@pytest.mark.asyncio
async def test_inflight_save_updates_index_live(meta_dir, index_on):
    await hs.write_meta_async("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    # No rebuild: the write updated the index directly.
    assert len(index_on) == 1
    res = await hs.find_best_restore_candidate_async(
        ["h_a1", "h_a2", "h_x3"], [], 100, 0.6, "m1"
    )
    assert res == ("h_a2", 2 / 3)


@pytest.mark.asyncio
async def test_subsumed_delete_updates_index(meta_dir, index_on):
    await hs.write_meta_async("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    await hs.write_meta_async(
        "h_a3", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2", "h_a3"]
    )
    assert len(index_on) == 2

    deleted = await hs.delete_subsumed_metas_async("h_a3", ["h_a1", "h_a2", "h_a3"], "m1")
    assert deleted == ["h_a2"]
    # a2 (the shorter, subsumed meta) is gone; a3 remains and now owns the
    # shared h_a1 hash.
    assert len(index_on) == 1
    res = await hs.find_best_restore_candidate_async(
        ["h_a1", "h_a2", "h_a3", "h_x4"], [], 100, 0.6, "m1"
    )
    assert res == ("h_a3", 3 / 4)


@pytest.mark.asyncio
async def test_delete_meta_async_updates_index(meta_dir, index_on):
    await hs.write_meta_async("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    assert await hs.delete_meta_async("h_a2") is True
    assert index_on.is_empty()


# ---- integration: validate-on-hit + reconcile -------------------------------

@pytest.mark.asyncio
async def test_validate_on_hit_skips_ghost(meta_dir, index_on):
    await hs.write_meta_async("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    # Simulate an out-of-band delete (bin_cache reconcile/clean).
    os.remove(os.path.join(str(meta_dir), "h_a2.meta.json"))
    res = await hs.find_best_restore_candidate_async(
        ["h_a1", "h_a2", "h_x3"], [], 100, 0.6, "m1"
    )
    # The ghost is skipped on hit; the disk fallback finds nothing.
    assert res is None
    assert index_on.is_empty(), "validate-on-hit must drop the ghost from the index"


@pytest.mark.asyncio
async def test_reconcile_index_async_drops_ghost(meta_dir, index_on):
    await hs.write_meta_async("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    os.remove(os.path.join(str(meta_dir), "h_a2.meta.json"))
    dropped = await hs.reconcile_index_async()
    assert dropped == ["h_a2"]
    assert index_on.is_empty()


@pytest.mark.asyncio
async def test_rebuild_index_async_counts(meta_dir, index_on):
    hs.write_meta("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    hs.write_meta("h_c2", "p", [], 100, "m1", prefix_hashes=["h_c1", "h_c2"])
    assert await hs.rebuild_index_async() == 2
    assert index_on.stats()["index_metas"] == 2


# ---- integration: flag off = legacy path ------------------------------------

@pytest.mark.asyncio
async def test_flag_off_uses_disk_and_index_untouched(meta_dir, fresh_index):
    await hs.write_meta_async("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    # Flag off: the write must not touch the index.
    assert fresh_index.is_empty()
    res = await hs.find_best_restore_candidate_async(
        ["h_a1", "h_a2", "h_x3"], [], 100, 0.6, "m1"
    )
    assert res == ("h_a2", 2 / 3)
    assert fresh_index.is_empty(), "flag off must leave the index untouched"


# ---- integration: cross-wpb (intentional permissiveness) --------------------

@pytest.mark.asyncio
async def test_cross_wpb_index_more_permissive(meta_dir, index_on):
    """The index drops the wpb filter (the .bin is wpb-agnostic), so it may
    return a cross-wpb candidate the on-disk tier-1 would skip. Documented,
    intentional divergence; in practice wpb is uniform so this never fires."""
    await hs.write_meta_async("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    req = ["h_a1", "h_a2", "h_x3"]
    idx = await hs.find_best_restore_candidate_async(req, [], 200, 0.6, "m1")
    disk = hs.find_best_restore_candidate(req, [], 200, 0.6, "m1")
    assert idx == ("h_a2", 2 / 3)
    assert disk is None


# ---- app reconcile loop -----------------------------------------------------

@pytest.mark.asyncio
async def test_meta_index_reconcile_loop_calls_reconcile(monkeypatch):
    calls = []

    async def fake_reconcile():
        calls.append(1)

    monkeypatch.setattr(hs, "reconcile_index_async", fake_reconcile)
    monkeypatch.setattr(app_module, "META_INDEX_RECONCILE_INTERVAL_S", 0.01)
    task = asyncio.create_task(app_module._meta_index_reconcile_loop())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert len(calls) >= 1


# ---- app lifespan wiring ----------------------------------------------------

def _mock_lifespan_deps(monkeypatch):
    """Stub the network-heavy startup work so the lifespan runs in-process."""

    def fake_client(url):
        c = MagicMock()
        c.close = AsyncMock()
        return c

    monkeypatch.setattr(app_module, "BACKENDS", [{"url": "http://be", "n_slots": 1}])
    monkeypatch.setattr(app_module, "LlamaClient", fake_client)
    monkeypatch.setattr(app_module, "SlotManager", lambda: MagicMock())
    monkeypatch.setattr(app_module, "_run_eviction", AsyncMock())
    monkeypatch.setattr(app_module, "_poll_slots", AsyncMock())


@pytest.mark.asyncio
async def test_lifespan_rebuilds_index_when_enabled(meta_dir, index_on, monkeypatch):
    hs.write_meta("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    _mock_lifespan_deps(monkeypatch)
    monkeypatch.setattr(app_module, "META_INDEX_ENABLED", True)
    monkeypatch.setattr(app_module, "META_INDEX_RECONCILE_INTERVAL_S", 0)

    async with app_module.lifespan(app_module.app):
        assert len(index_on) == 1, "startup must rebuild the index from disk"
        assert app_module._meta_index_reconcile_task is None  # interval 0 -> no task


@pytest.mark.asyncio
async def test_lifespan_starts_reconcile_task_when_enabled(meta_dir, index_on, monkeypatch):
    _mock_lifespan_deps(monkeypatch)
    monkeypatch.setattr(app_module, "META_INDEX_ENABLED", True)
    monkeypatch.setattr(app_module, "META_INDEX_RECONCILE_INTERVAL_S", 300)

    async with app_module.lifespan(app_module.app):
        assert app_module._meta_index_reconcile_task is not None


@pytest.mark.asyncio
async def test_lifespan_no_index_when_disabled(meta_dir, fresh_index, monkeypatch):
    hs.write_meta("h_a2", "p", [], 100, "m1", prefix_hashes=["h_a1", "h_a2"])
    _mock_lifespan_deps(monkeypatch)

    async with app_module.lifespan(app_module.app):
        assert fresh_index.is_empty(), "flag off must not build the index"
        assert app_module._meta_index_reconcile_task is None
