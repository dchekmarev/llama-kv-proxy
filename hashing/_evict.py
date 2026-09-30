# hashing/_evict.py

"""TTL / size eviction, the full wipe, and the _MetaFile view of the store."""

import asyncio
import os
import time
from typing import NamedTuple

import hashing as hs
import promstats

from ._state import log


class _MetaFile(NamedTuple):
    """One meta file on disk, as the eviction pass needs it."""

    path: str
    key: str
    mtime: float
    size: int


def evict_meta(ttl_hours: float, max_files: int, max_mb: float) -> dict:
    """Delete meta files until every cap holds; 0 disables that cap.

    The TTL uses the file mtime (refreshed by write_meta and touch_meta, so
    "oldest" is least recently used); the file-count and byte caps delete oldest
    first. Returns the deleted keys and how many metas remain. Callers pass the
    three settings by keyword.
    """
    alive: list[_MetaFile] = []
    for path in hs._meta_files():
        try:
            st = os.stat(path)
        except OSError:
            continue
        alive.append(_MetaFile(path, hs._key_of(path), st.st_mtime, st.st_size))
    deleted: list[str] = []

    def _delete(entry: _MetaFile, reason: str) -> None:
        if hs._remove_meta_file(entry.path):
            deleted.append(entry.key)
            promstats.evictions_total.labels(reason=reason).inc()

    if ttl_hours > 0:
        cutoff = time.time() - ttl_hours * 3600.0
        for entry in [e for e in alive if e.mtime < cutoff]:
            _delete(entry, "ttl")
        alive = [e for e in alive if e.mtime >= cutoff]
    if max_files > 0:
        while len(alive) > max_files:
            oldest = min(alive, key=lambda e: e.mtime)
            _delete(oldest, "max_files")
            alive.remove(oldest)
    if max_mb > 0:
        limit = max_mb * 1024 * 1024
        total = sum(e.size for e in alive)
        while alive and total > limit:
            oldest = min(alive, key=lambda e: e.mtime)
            total -= oldest.size
            _delete(oldest, "max_mb")
            alive.remove(oldest)
    if deleted:
        log.info("meta_evicted n=%d remaining=%d", len(deleted), len(alive))
    return {"deleted": deleted, "remaining": len(alive)}


async def evict_meta_async(ttl_hours: float, max_files: int, max_mb: float) -> dict:
    """evict_meta in a worker thread (the scan is disk-bound)."""
    return await asyncio.to_thread(hs.evict_meta, ttl_hours, max_files, max_mb)


def _clear_all_meta() -> list[str]:
    """Delete every meta file; the index is emptied by the async wrapper."""
    deleted = []
    for path in hs._meta_files():
        if hs._remove_meta_file(path):
            deleted.append(hs._key_of(path))
    if deleted:
        promstats.evictions_total.labels(reason="clear").inc(len(deleted))
    log.info("meta_cache_cleared n=%d", len(deleted))
    return deleted


async def clear_all_meta_async() -> list[str]:
    """Delete every meta file in a thread and empty the index on the loop."""
    deleted = await asyncio.to_thread(_clear_all_meta)
    if hs.META_INDEX_ENABLED:
        hs._index.clear()
    return deleted
