# tests/test_bin_save_grace.py

"""M2: a .bin file written by an in-flight save (fresh mtime, meta not yet
written) must not be deleted by the LRU cleanup or orphan reconciliation. A
true orphan (old mtime, no meta) is still deleted; a tracked .bin (has a meta)
is always eligible regardless of mtime."""

import json
import os
import time

from cache import bin_cache

_MB = 1024 * 1024


def _make_bin(bin_dir, name: str, size_bytes: int, age_s: float = 0.0) -> str:
    path = os.path.join(str(bin_dir), name)
    with open(path, "wb") as f:
        f.write(b"x" * size_bytes)
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


def _make_meta(meta_dir, name: str, timestamp: float) -> None:
    with open(os.path.join(str(meta_dir), f"{name}.meta.json"), "w", encoding="utf-8") as f:
        json.dump({"key": name, "timestamp": timestamp}, f)


def _setup(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    monkeypatch.setattr(bin_cache, "META_DIR", str(meta_dir))
    monkeypatch.setattr(bin_cache, "BIN_SAVE_GRACE_S", 10.0)
    return bin_dir, meta_dir


def test_clean_keeps_recent_metaless_bin(tmp_path, monkeypatch):
    """A fresh .bin without a meta (in-flight save) is kept; an old meta-less
    .bin (true orphan) is deleted."""
    bin_dir, _ = _setup(tmp_path, monkeypatch)
    fresh = _make_bin(bin_dir, "fresh", 2 * _MB, age_s=0.0)
    old = _make_bin(bin_dir, "old", 2 * _MB, age_s=3600)

    bin_cache.clean_bin_cache(str(bin_dir), max_mb=1)

    assert os.path.exists(fresh), "in-flight .bin must be kept"
    assert not os.path.exists(old), "old orphan .bin must be deleted"


def test_reconcile_keeps_recent_metaless_bin(tmp_path, monkeypatch):
    """reconcile must not delete a fresh meta-less .bin (in-flight save)."""
    bin_dir, _ = _setup(tmp_path, monkeypatch)
    fresh = _make_bin(bin_dir, "fresh", 1024, age_s=0.0)
    old = _make_bin(bin_dir, "old", 1024, age_s=3600)

    res = bin_cache.reconcile_bin_cache(str(bin_dir))

    assert os.path.exists(fresh), "in-flight .bin must be kept"
    assert not os.path.exists(old), "old orphan .bin must be deleted"
    assert res["deleted_bins"] == ["old"]


def test_clean_deletes_old_metaless_bin(tmp_path, monkeypatch):
    """A meta-less .bin older than the grace window is a true orphan: deleted."""
    bin_dir, _ = _setup(tmp_path, monkeypatch)
    old = _make_bin(bin_dir, "old", 2 * _MB, age_s=3600)

    bin_cache.clean_bin_cache(str(bin_dir), max_mb=1)

    assert not os.path.exists(old)


def test_clean_deletes_tracked_bin_even_if_fresh(tmp_path, monkeypatch):
    """A .bin WITH a meta is always eligible, even with a fresh mtime."""
    bin_dir, meta_dir = _setup(tmp_path, monkeypatch)
    tracked = _make_bin(bin_dir, "tracked", 2 * _MB, age_s=0.0)
    _make_meta(meta_dir, "tracked", time.time())

    bin_cache.clean_bin_cache(str(bin_dir), max_mb=1)

    assert not os.path.exists(tracked), "tracked .bin must be deleted"


def _make_ckpt(bin_dir, key: str, size_bytes: int, age_s: float = 0.0) -> str:
    path = os.path.join(str(bin_dir), f"{key}.ckpt")
    with open(path, "wb") as f:
        f.write(b"x" * size_bytes)
    if age_s:
        t = time.time() - age_s
        os.utime(path, (t, t))
    return path


def test_clean_keeps_inflight_bin_with_ckpt(tmp_path, monkeypatch):
    """An in-flight save writes .bin then .ckpt; both are meta-less and
    fresh, so the whole pair must be kept (the .ckpt inherits the .bin's
    protection instead of looking like an orphan)."""
    bin_dir, _ = _setup(tmp_path, monkeypatch)
    fresh = _make_bin(bin_dir, "fresh", 2 * _MB, age_s=0.0)
    fresh_ckpt = _make_ckpt(bin_dir, "fresh", 1024, age_s=0.0)
    old = _make_bin(bin_dir, "old", 2 * _MB, age_s=3600)

    bin_cache.clean_bin_cache(str(bin_dir), max_mb=1)

    assert os.path.exists(fresh), "in-flight .bin must be kept"
    assert os.path.exists(fresh_ckpt), "in-flight .ckpt must be kept"
    assert not os.path.exists(old), "old orphan .bin must be deleted"


def test_reconcile_keeps_inflight_pair(tmp_path, monkeypatch):
    """An in-flight save (fresh .bin + fresh .ckpt, no meta yet) must survive
    reconcile: the .bin is grace-skipped and the .ckpt is not an orphan
    (its owner is present)."""
    bin_dir, _ = _setup(tmp_path, monkeypatch)
    fresh = _make_bin(bin_dir, "fresh", 1024, age_s=0.0)
    fresh_ckpt = _make_ckpt(bin_dir, "fresh", 1024, age_s=0.0)
    old = _make_bin(bin_dir, "old", 1024, age_s=3600)

    res = bin_cache.reconcile_bin_cache(str(bin_dir))

    assert os.path.exists(fresh), "in-flight .bin must be kept"
    assert os.path.exists(fresh_ckpt), "in-flight .ckpt must be kept"
    assert not os.path.exists(old), "old orphan .bin must be deleted"
    assert res["deleted_bins"] == ["old"]


def test_reconcile_keeps_recent_orphan_ckpt(tmp_path, monkeypatch):
    """A fresh .ckpt without its .bin is protected by the grace window; an
    old one is a true orphan and is deleted."""
    bin_dir, _ = _setup(tmp_path, monkeypatch)
    fresh = _make_ckpt(bin_dir, "fresh", 1024, age_s=0.0)
    old = _make_ckpt(bin_dir, "old", 1024, age_s=3600)

    res = bin_cache.reconcile_bin_cache(str(bin_dir))

    assert os.path.exists(fresh), "fresh orphan .ckpt must be kept"
    assert not os.path.exists(old), "old orphan .ckpt must be deleted"
    assert res["deleted_bins"] == ["old.ckpt"]
