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


def _make_bin(bin_dir, key: str, size_bytes: int) -> str:
    path = os.path.join(str(bin_dir), key)
    with open(path, "wb") as f:
        f.write(b"x" * size_bytes)
    return path


def _make_meta(meta_dir, key: str, timestamp: float) -> str:
    path = os.path.join(str(meta_dir), f"{key}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"key": key, "timestamp": timestamp}, f)
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
    _make_bin(bin_dir, "orphan", mb)
    _make_bin(bin_dir, "tracked", mb)
    _make_meta(meta_dir, "tracked", now)

    res = bin_cache.clean_bin_cache(str(bin_dir), 1)

    assert res["deleted"] == ["orphan"]
    assert os.path.exists(os.path.join(str(bin_dir), "tracked"))


def test_delete_bin_file(bin_dir):
    _make_bin(bin_dir, "a", 100)
    assert bin_cache.delete_bin_file(str(bin_dir), "a") is True
    assert not os.path.exists(os.path.join(str(bin_dir), "a"))
    assert bin_cache.delete_bin_file(str(bin_dir), "a") is False


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
