# tests/test_save_and_write_meta.py

"""_save_and_write_meta: subsumed metas are dropped ONLY after the new meta is
written. A failed meta write must not delete the still-valid shorter caches
(data-loss guard)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import bin_cache
import hashing as hs


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hs, "META_DIR", str(tmp_path))
    return tmp_path


def _sm():
    sm = MagicMock()
    sm.save_after = AsyncMock(return_value=True)
    return sm


def _write_prefix(key, prefix_hashes, model_id="m1"):
    # A meta's key is its full-conversation hash == the last prefix hash.
    assert key == prefix_hashes[-1]
    hs.write_meta(key, "p", [], 100, model_id, prefix_hashes=prefix_hashes)


def _patch(monkeypatch):
    monkeypatch.setattr(hs, "write_meta_async", AsyncMock())
    monkeypatch.setattr(app_module, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(app_module, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: None)


@pytest.mark.asyncio
async def test_drops_subsumed_metas_after_successful_meta(meta_dir, monkeypatch):
    """Successful save + meta: the strict-prefix meta is deleted, the new kept."""
    _write_prefix("h_ab", ["h_a", "h_ab"])
    _write_prefix("h_abc", ["h_a", "h_ab", "h_abc"])
    _patch(monkeypatch)

    ok = await app_module._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert ok is True
    assert not (meta_dir / "h_ab.meta.json").exists(), "subsumed meta must be deleted"
    assert (meta_dir / "h_abc.meta.json").exists(), "new meta must be kept"


@pytest.mark.asyncio
async def test_keeps_subsumed_metas_when_meta_write_fails(meta_dir, monkeypatch):
    """A failed meta write must NOT delete the still-valid shorter caches."""
    _write_prefix("h_ab", ["h_a", "h_ab"])

    async def _raise(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(hs, "write_meta_async", _raise)
    monkeypatch.setattr(app_module, "_schedule_lru_check", lambda: None)
    monkeypatch.setattr(app_module, "_purge_backend_files", AsyncMock())
    monkeypatch.setattr(bin_cache, "get_bin_size", lambda d, k: None)

    ok = await app_module._save_and_write_meta(
        [], _sm(), ("g",), "h_abc", "p", [], ["h_a", "h_ab", "h_abc"], "m1"
    )

    assert ok is True, "the slot save itself succeeded"
    assert (meta_dir / "h_ab.meta.json").exists(), "subsumed meta must be kept"
    assert not (meta_dir / "h_abc.meta.json").exists(), "failed meta must not exist"
