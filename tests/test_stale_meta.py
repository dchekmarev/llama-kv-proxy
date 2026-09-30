# tests/test_stale_meta.py

"""P1-2: a meta file pointing at a cache that cannot be restored is stale and
must be dropped, otherwise every big request retries a doomed restore."""

from unittest.mock import AsyncMock

import pytest

import app as app_module
import chat_flow
import hashing as hs
from llama_client import RESTORE_MISSING


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


def _big_content():
    return " ".join(["w"] * 600)


def _write_meta_for(content, meta_dir):
    prefix = hs.raw_prefix([{"role": "user", "content": content}])
    key = hs.prefix_key_sha256("m1\n" + prefix)
    blocks = hs.block_hashes_from_text(prefix, hs.WORDS_PER_BLOCK)
    hs.write_meta(key, prefix, blocks, hs.WORDS_PER_BLOCK, "m1")
    return key


async def _chat(sm, content):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    data = {
        "messages": [{"role": "user", "content": content}],
        "stream": False,
    }
    return await app_module.chat(FakeRequest(data))


@pytest.mark.asyncio
async def test_delete_meta_removes_file(meta_dir):
    key = "a" * 64
    hs.write_meta(key, "p", ["b"], hs.WORDS_PER_BLOCK, "m1")
    assert (meta_dir / f"{key}.meta.json").exists()

    assert hs.delete_meta(key) is True
    assert not (meta_dir / f"{key}.meta.json").exists()


@pytest.mark.asyncio
async def test_delete_meta_missing_returns_false(meta_dir):
    assert hs.delete_meta("b" * 64) is False


@pytest.mark.asyncio
async def test_delete_meta_async(meta_dir):
    key = "c" * 64
    hs.write_meta(key, "p", ["b"], hs.WORDS_PER_BLOCK, "m1")

    assert await hs.delete_meta_async(key) is True
    assert not (meta_dir / f"{key}.meta.json").exists()


@pytest.mark.asyncio
async def test_stale_meta_dropped_when_file_missing(sm, meta_dir, monkeypatch):
    """restore -> RESTORE_MISSING (404): the meta must be deleted."""
    content = _big_content()
    key = _write_meta_for(content, meta_dir)
    sm.backends[0]["client"].restore_slot = AsyncMock(
        return_value=RESTORE_MISSING
    )

    delete_meta_async = AsyncMock()
    monkeypatch.setattr(hs, "delete_meta_async", delete_meta_async)

    await _chat(sm, content)

    delete_meta_async.assert_awaited_once_with(key)


@pytest.mark.asyncio
async def test_meta_kept_when_restore_fails_other(sm, meta_dir, monkeypatch):
    """restore -> False (a non-missing failure): the meta must be kept, since
    the cache may still be valid and a retry can succeed."""
    content = _big_content()
    _write_meta_for(content, meta_dir)
    sm.backends[0]["client"].restore_slot = AsyncMock(return_value=False)

    delete_meta_async = AsyncMock()
    monkeypatch.setattr(hs, "delete_meta_async", delete_meta_async)

    await _chat(sm, content)

    delete_meta_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_meta_kept_when_restore_succeeds(sm, meta_dir, monkeypatch):
    """restore -> True: no meta deletion."""
    content = _big_content()
    _write_meta_for(content, meta_dir)
    sm.backends[0]["client"].restore_slot = AsyncMock(return_value=True)

    delete_meta_async = AsyncMock()
    monkeypatch.setattr(hs, "delete_meta_async", delete_meta_async)

    await _chat(sm, content)

    delete_meta_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_meta_dropped_for_resolved_key_on_substitution(
    sm, meta_dir, monkeypatch
):
    """Substitution K1->K2 + RESTORE_MISSING: the RESOLVED key's (K2) stale
    meta must be dropped, not the original candidate's (K1) — the original's
    meta was already deleted by the subsumption that created the alias, so
    cleaning it is a no-op and leaves K2's stale meta behind for repeated
    hopeless 404 restores."""
    content = _big_content()
    k1 = _write_meta_for(content, meta_dir)
    k2 = "d" * 64  # replacement cache: meta on disk, .bin gone
    hs.write_meta(k2, "p", ["b"], hs.WORDS_PER_BLOCK, "m1")

    chat_flow._PENDING_RESTORES.clear()
    chat_flow._RESTORE_ALIAS.clear()
    chat_flow._register_pending_restore(k1)
    chat_flow._RESTORE_ALIAS[k1] = k2

    sm.backends[0]["client"].restore_slot = AsyncMock(return_value=RESTORE_MISSING)

    delete_meta_async = AsyncMock()
    monkeypatch.setattr(hs, "delete_meta_async", delete_meta_async)

    await _chat(sm, content)

    delete_meta_async.assert_awaited_once_with(k2)
