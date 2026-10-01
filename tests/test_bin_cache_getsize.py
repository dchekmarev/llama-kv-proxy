# tests/test_bin_cache_getsize.py

"""bin_cache.get_bin_size: size of <dir>/<key> in bytes, or None.

The backend .bin file is named exactly by the cache key (no suffix), matching
delete_bin_file / _make_bin conventions.
"""

from cache import bin_cache


def test_get_bin_size_existing(tmp_path):
    (tmp_path / "abc").write_bytes(b"0123456789")  # 10 bytes
    assert bin_cache.get_bin_size(str(tmp_path), "abc") == 10


def test_get_bin_size_missing(tmp_path):
    assert bin_cache.get_bin_size(str(tmp_path), "nope") is None


def test_get_bin_size_dir_missing(tmp_path):
    assert bin_cache.get_bin_size(str(tmp_path / "does-not-exist"), "abc") is None
