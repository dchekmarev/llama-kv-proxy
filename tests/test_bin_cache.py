# tests/test_bin_cache.py

"""Direct filesystem cleanup of backend .bin files (LRU by meta timestamp,
orphaned files first, size cap)."""

import json
import os
import time

import pytest

import bin_cache


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    d = tmp_path / "meta"
    d.mkdir()
    monkeypatch.setattr(bin_cache, "META_DIR", str(d))
    return d


@pytest.fixture()
def bin_dir(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    return d


def _make_bin(bin_dir, key: str, size_bytes: int, age_s: float = 0.0) -> str:
    path = os.path.join(str(bin_dir), key)
    with open(path, "wb") as f:
        f.write(b"x" * size_bytes)
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


def _make_meta(meta_dir, key: str, timestamp: float) -> str:
    path = os.path.join(str(meta_dir), f"{key}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"key": key, "timestamp": timestamp}, f)
    return path


def _make_ckpt(bin_dir, key: str, size_bytes: int, age_s: float = 0.0) -> str:
    path = os.path.join(str(bin_dir), f"{key}.ckpt")
    with open(path, "wb") as f:
        f.write(b"y" * size_bytes)
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


def test_clean_disabled_when_dir_empty():
    assert bin_cache.clean_bin_cache("", 10) == {"deleted": [], "remaining": 0}


def test_clean_disabled_when_max_mb_zero(bin_dir, meta_dir):
    _make_bin(bin_dir, "a", 100)
    res = bin_cache.clean_bin_cache(str(bin_dir), 0)
    assert res == {"deleted": [], "remaining": 0}
    assert os.path.exists(os.path.join(str(bin_dir), "a"))


def test_clean_noop_when_under_cap(bin_dir, meta_dir):
    _make_bin(bin_dir, "a", 100)
    _make_meta(meta_dir, "a", time.time())
    res = bin_cache.clean_bin_cache(str(bin_dir), 10)
    assert res["deleted"] == []
    assert res["remaining"] == 1


def test_clean_deletes_oldest_first_by_meta_timestamp(bin_dir, meta_dir):
    now = time.time()
    mb = 1024 * 1024
    _make_bin(bin_dir, "old", mb)
    _make_bin(bin_dir, "mid", mb)
    _make_bin(bin_dir, "new", mb)
    _make_meta(meta_dir, "old", now - 300)
    _make_meta(meta_dir, "mid", now - 200)
    _make_meta(meta_dir, "new", now - 100)

    res = bin_cache.clean_bin_cache(str(bin_dir), 2)

    assert res["deleted"] == ["old"]
    assert res["remaining"] == 2
    assert not os.path.exists(os.path.join(str(bin_dir), "old"))
    assert os.path.exists(os.path.join(str(bin_dir), "mid"))
    assert os.path.exists(os.path.join(str(bin_dir), "new"))


def test_clean_orphan_deleted_first(bin_dir, meta_dir):
    now = time.time()
    mb = 1024 * 1024
    # A true orphan has an old mtime (it has been sitting there); a fresh
    # meta-less .bin is an in-flight save and is protected by the grace window.
    _make_bin(bin_dir, "orphan", mb, age_s=3600)
    _make_bin(bin_dir, "tracked", mb)
    _make_meta(meta_dir, "tracked", now)

    res = bin_cache.clean_bin_cache(str(bin_dir), 1)

    assert res["deleted"] == ["orphan"]
    assert os.path.exists(os.path.join(str(bin_dir), "tracked"))


def test_clean_deletes_meta_alongside_bin(bin_dir, meta_dir):
    """M3: evicting a tracked .bin must also delete its meta (no orphan meta
    left behind for the next reconcile/restore to clean up)."""
    now = time.time()
    mb = 1024 * 1024
    _make_bin(bin_dir, "evict", mb, age_s=3600)
    _make_meta(meta_dir, "evict", now - 3600)
    _make_bin(bin_dir, "keep", mb)
    _make_meta(meta_dir, "keep", now)

    res = bin_cache.clean_bin_cache(str(bin_dir), 1)

    assert res["deleted"] == ["evict"]
    assert not os.path.exists(os.path.join(str(bin_dir), "evict"))
    assert not os.path.exists(
        os.path.join(str(meta_dir), "evict.meta.json")
    ), "meta must be deleted together with the .bin"
    assert os.path.exists(os.path.join(str(bin_dir), "keep"))
    assert os.path.exists(os.path.join(str(meta_dir), "keep.meta.json"))


def test_clean_deletes_ckpt_sidecar_alongside_bin(bin_dir, meta_dir):
    """Evicting a tracked .bin must also delete its .ckpt sidecar (and keep
    the pair of a surviving .bin intact)."""
    now = time.time()
    mb = 1024 * 1024
    _make_bin(bin_dir, "evict", mb, age_s=3600)
    _make_ckpt(bin_dir, "evict", 1024, age_s=3600)
    _make_meta(meta_dir, "evict", now - 3600)
    _make_bin(bin_dir, "keep", mb)
    _make_ckpt(bin_dir, "keep", 1024)
    _make_meta(meta_dir, "keep", now)

    # Total is ~2.002 MB; with a 2 MB cap only the older pair is evicted
    # (the survivor's .ckpt keeps it just under the cap).
    res = bin_cache.clean_bin_cache(str(bin_dir), 2)

    assert res["deleted"] == ["evict"]
    assert not os.path.exists(os.path.join(str(bin_dir), "evict"))
    assert not os.path.exists(
        os.path.join(str(bin_dir), "evict.ckpt")
    ), "sidecar must be deleted together with the .bin"
    assert os.path.exists(os.path.join(str(bin_dir), "keep"))
    assert os.path.exists(os.path.join(str(bin_dir), "keep.ckpt"))


def test_clean_counts_ckpt_size_in_cap(bin_dir, meta_dir):
    """The .ckpt sidecar counts toward the size cap: a .bin under the cap on
    its own is evicted once bin + sidecar exceed it."""
    now = time.time()
    mb = 1024 * 1024
    _make_bin(bin_dir, "big", int(0.9 * mb))
    _make_ckpt(bin_dir, "big", int(0.2 * mb))
    _make_meta(meta_dir, "big", now)

    # 0.9 + 0.2 = 1.1 MB > 1 MB cap.
    res = bin_cache.clean_bin_cache(str(bin_dir), 1)

    assert res["deleted"] == ["big"]
    assert not os.path.exists(os.path.join(str(bin_dir), "big"))
    assert not os.path.exists(os.path.join(str(bin_dir), "big.ckpt"))


def test_clean_deletes_standalone_orphan_ckpt(bin_dir, meta_dir):
    """A .ckpt whose .bin is gone is an orphan entry: evicted like any other
    orphan (old mtime, over cap)."""
    mb = 1024 * 1024
    _make_ckpt(bin_dir, "ghost", mb, age_s=3600)
    _make_bin(bin_dir, "keep", mb)
    _make_meta(meta_dir, "keep", time.time())

    res = bin_cache.clean_bin_cache(str(bin_dir), 1)

    assert res["deleted"] == ["ghost.ckpt"]
    assert not os.path.exists(os.path.join(str(bin_dir), "ghost.ckpt"))
    assert os.path.exists(os.path.join(str(bin_dir), "keep"))


def test_clean_keeps_paired_ckpt_with_bin(bin_dir, meta_dir):
    """Under the cap a .ckpt is not treated as an orphan: it stays with its
    .bin and does not inflate the remaining count."""
    now = time.time()
    _make_bin(bin_dir, "a", 100)
    _make_ckpt(bin_dir, "a", 50)
    _make_meta(meta_dir, "a", now)

    res = bin_cache.clean_bin_cache(str(bin_dir), 10)

    assert res["deleted"] == []
    assert res["remaining"] == 1
    assert os.path.exists(os.path.join(str(bin_dir), "a.ckpt"))


def test_delete_bin_file(bin_dir):
    _make_bin(bin_dir, "a", 100)
    assert bin_cache.delete_bin_file(str(bin_dir), "a") is True
    assert not os.path.exists(os.path.join(str(bin_dir), "a"))
    assert bin_cache.delete_bin_file(str(bin_dir), "a") is False


def test_delete_bin_file_removes_sidecar(bin_dir):
    _make_bin(bin_dir, "a", 100)
    ckpt = _make_ckpt(bin_dir, "a", 50)
    assert bin_cache.delete_bin_file(str(bin_dir), "a") is True
    assert not os.path.exists(os.path.join(str(bin_dir), "a"))
    assert not os.path.exists(ckpt)


def test_delete_bin_file_without_sidecar(bin_dir):
    _make_bin(bin_dir, "a", 100)
    assert bin_cache.delete_bin_file(str(bin_dir), "a") is True
    assert not os.path.exists(os.path.join(str(bin_dir), "a"))


def test_delete_bin_file_disabled_when_dir_empty():
    assert bin_cache.delete_bin_file("", "a") is False


def test_clear_bin_cache(bin_dir):
    _make_bin(bin_dir, "a", 100)
    _make_bin(bin_dir, "b", 200)
    n = bin_cache.clear_bin_cache(str(bin_dir))
    assert n == 2
    assert not os.path.exists(os.path.join(str(bin_dir), "a"))
    assert not os.path.exists(os.path.join(str(bin_dir), "b"))


def test_clear_bin_cache_disabled_when_dir_empty():
    assert bin_cache.clear_bin_cache("") == 0


def test_reconcile_disabled_when_dir_empty(meta_dir):
    assert bin_cache.reconcile_bin_cache("") == {
        "deleted_metas": [],
        "deleted_bins": [],
    }


def test_reconcile_noop_when_consistent(bin_dir, meta_dir):
    now = time.time()
    _make_bin(bin_dir, "a", 100)
    _make_meta(meta_dir, "a", now)
    res = bin_cache.reconcile_bin_cache(str(bin_dir))
    assert res == {"deleted_metas": [], "deleted_bins": []}
    assert os.path.exists(os.path.join(str(bin_dir), "a"))
    assert os.path.exists(os.path.join(str(meta_dir), "a.meta.json"))


def test_reconcile_noop_when_consistent_with_ckpt(bin_dir, meta_dir):
    """A .bin + meta + .ckpt triple is consistent: nothing is deleted."""
    now = time.time()
    _make_bin(bin_dir, "a", 100)
    _make_ckpt(bin_dir, "a", 50)
    _make_meta(meta_dir, "a", now)
    res = bin_cache.reconcile_bin_cache(str(bin_dir))
    assert res == {"deleted_metas": [], "deleted_bins": []}
    assert os.path.exists(os.path.join(str(bin_dir), "a"))
    assert os.path.exists(os.path.join(str(bin_dir), "a.ckpt"))


def test_reconcile_deletes_stale_meta(bin_dir, meta_dir):
    now = time.time()
    # Meta with no matching .bin -> stale, delete the meta.
    _make_meta(meta_dir, "stale", now)
    res = bin_cache.reconcile_bin_cache(str(bin_dir))
    assert res["deleted_metas"] == ["stale"]
    assert res["deleted_bins"] == []
    assert not os.path.exists(os.path.join(str(meta_dir), "stale.meta.json"))


def test_reconcile_deletes_orphan_bin(bin_dir, meta_dir):
    # .bin with no matching meta -> orphan, delete the .bin. An old mtime
    # marks it as a true orphan (a fresh meta-less .bin is an in-flight save).
    _make_bin(bin_dir, "orphan", 100, age_s=3600)
    res = bin_cache.reconcile_bin_cache(str(bin_dir))
    assert res["deleted_bins"] == ["orphan"]
    assert res["deleted_metas"] == []
    assert not os.path.exists(os.path.join(str(bin_dir), "orphan"))


def test_reconcile_both_directions(bin_dir, meta_dir):
    now = time.time()
    # Consistent pair (kept).
    _make_bin(bin_dir, "ok", 100)
    _make_meta(meta_dir, "ok", now)
    # Stale meta (no .bin).
    _make_meta(meta_dir, "stale", now)
    # Orphan .bin (no meta, old mtime = true orphan, not an in-flight save).
    _make_bin(bin_dir, "orphan", 100, age_s=3600)

    res = bin_cache.reconcile_bin_cache(str(bin_dir))

    assert res["deleted_metas"] == ["stale"]
    assert res["deleted_bins"] == ["orphan"]
    assert os.path.exists(os.path.join(str(bin_dir), "ok"))
    assert os.path.exists(os.path.join(str(meta_dir), "ok.meta.json"))
    assert not os.path.exists(os.path.join(str(meta_dir), "stale.meta.json"))
    assert not os.path.exists(os.path.join(str(bin_dir), "orphan"))


def test_reconcile_deletes_ckpt_with_orphan_bin(bin_dir, meta_dir):
    """An orphan .bin is deleted together with its .ckpt sidecar."""
    _make_bin(bin_dir, "orphan", 100, age_s=3600)
    _make_ckpt(bin_dir, "orphan", 50, age_s=3600)
    res = bin_cache.reconcile_bin_cache(str(bin_dir))
    assert res["deleted_bins"] == ["orphan"]
    assert not os.path.exists(os.path.join(str(bin_dir), "orphan"))
    assert not os.path.exists(os.path.join(str(bin_dir), "orphan.ckpt"))


def test_reconcile_deletes_orphan_ckpt(bin_dir, meta_dir):
    """A .ckpt whose .bin is gone (partial eviction) is an orphan: deleted."""
    _make_ckpt(bin_dir, "ghost", 50, age_s=3600)
    res = bin_cache.reconcile_bin_cache(str(bin_dir))
    assert res["deleted_bins"] == ["ghost.ckpt"]
    assert not os.path.exists(os.path.join(str(bin_dir), "ghost.ckpt"))
