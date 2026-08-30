# tests/test_atomic_meta_write.py

"""M1: meta writes must be atomic. A crash mid-write must never leave a
truncated/corrupt target file or a stray temp file behind."""

import json

import pytest

import hashing as hs


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "META_DIR", str(tmp_path))
    return tmp_path


def _tmp_leftovers(meta_dir):
    return [p.name for p in meta_dir.iterdir() if p.name.endswith(".tmp")]


def test_write_meta_produces_valid_json_and_no_temp(meta_dir):
    key = "a" * 64
    hs.write_meta(key, "p", ["b"], 8, "m1")

    path = meta_dir / f"{key}.meta.json"
    assert json.loads(path.read_text())["key"] == key
    assert _tmp_leftovers(meta_dir) == []


def test_write_meta_crash_leaves_no_partial_file(meta_dir, monkeypatch):
    """A crash during json.dump must not truncate the existing valid meta."""
    key = "b" * 64
    hs.write_meta(key, "p", ["b"], 8, "m1")
    before = (meta_dir / f"{key}.meta.json").read_text()

    def _boom(f, *a, **k):
        raise OSError("simulated crash")

    monkeypatch.setattr(hs.json, "dump", _boom)

    with pytest.raises(OSError):
        hs.write_meta(key, "p2", ["c"], 8, "m1")

    path = meta_dir / f"{key}.meta.json"
    # The target still holds the previous valid document (never a partial one).
    assert path.read_text() == before
    assert _tmp_leftovers(meta_dir) == []


def test_touch_meta_crash_leaves_no_partial_file(meta_dir, monkeypatch):
    """touch_meta is best-effort: a crash during the rewrite must not corrupt
    the existing meta and must not leak a temp file."""
    key = "c" * 64
    hs.write_meta(key, "p", ["b"], 8, "m1")
    before = (meta_dir / f"{key}.meta.json").read_text()

    def _boom(f, *a, **k):
        raise OSError("simulated crash")

    monkeypatch.setattr(hs.json, "dump", _boom)

    # touch_meta swallows write errors; no exception propagates.
    hs.touch_meta(key)

    path = meta_dir / f"{key}.meta.json"
    assert path.read_text() == before
    assert _tmp_leftovers(meta_dir) == []


def test_touch_meta_updates_timestamp(meta_dir):
    key = "d" * 64
    hs.write_meta(key, "p", ["b"], 8, "m1")
    before = json.loads((meta_dir / f"{key}.meta.json").read_text())

    hs.touch_meta(key)

    after = json.loads((meta_dir / f"{key}.meta.json").read_text())
    assert after["timestamp"] >= before["timestamp"]
    assert after["key"] == key
