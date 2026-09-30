# hashing/_index_ops.py

"""Restore-index maintenance: the startup rebuild and the disk reconcile."""

import asyncio

import hashing as hs


def _index_entries() -> list[dict]:
    """Index entries ({key, model_id, size, hashes}) for every meta on disk."""
    entries = []
    for meta in hs.scan_all_meta():
        key = meta.get("key")
        if not key:
            continue
        entries.append(
            {
                "key": key,
                "model_id": meta.get("model_id") or "",
                "size": hs._candidate_size(meta),
                "hashes": hs._prefix_hashes_of(meta),
            }
        )
    return entries


async def rebuild_index_async() -> int:
    """Rebuild the index from disk (startup). Returns the number of metas."""
    entries = await asyncio.to_thread(_index_entries)
    hs._index.rebuild_from(entries)
    return len(entries)


async def reconcile_index_async() -> list[str]:
    """Drop index entries whose meta file is gone; returns the dropped keys.

    External mutators (bin_cache reconcile/clean) delete meta files without
    notifying the index, so the periodic reconcile turns those ghosts into
    clean misses.
    """
    if not hs.META_INDEX_ENABLED:
        return []
    live = await asyncio.to_thread(lambda: {hs._key_of(p) for p in hs._meta_files()})
    return hs._index.reconcile_keys(live)
