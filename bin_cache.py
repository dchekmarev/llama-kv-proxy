# bin_cache.py

"""
Direct filesystem cleanup of backend .bin cache files.

llama.cpp has no endpoint to delete files from --slot-save-path: the `erase`
slot action only clears in-memory state, and save/restore only write/read.
So when the save directory is mounted into the proxy we remove .bin files
ourselves.

LRU order: a .bin file's "last use" is its meta file's timestamp, which is
refreshed on both save and restore (see hashing.write_meta / touch_meta).
Orphaned .bin files (no matching meta) are deleted first.
"""

import glob
import json
import logging
import os

from config import META_DIR

log = logging.getLogger(__name__)


def _meta_timestamp(basename: str) -> float | None:
    """Last-use time from the meta file, or None if the meta is missing."""
    path = os.path.join(META_DIR, f"{basename}.meta.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("timestamp") or 0)
    except (OSError, ValueError, TypeError):
        return None


def _entries(dir: str) -> list[tuple[float, str, int]]:
    """(last_use, path, size) for every file in dir.

    last_use is the meta timestamp when a meta exists, else 0 (orphaned
    files sort first and are deleted before tracked ones).
    """
    entries: list[tuple[float, str, int]] = []
    for path in glob.glob(os.path.join(dir, "*")):
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        ts = _meta_timestamp(os.path.basename(path))
        entries.append((ts if ts is not None else 0.0, path, size))
    return entries


def clean_bin_cache(dir: str, max_mb: int) -> dict:
    """Delete oldest .bin files until total size <= max_mb.

    LRU order: meta timestamp (last save/restore); orphaned files (no meta)
    are deleted first. Returns {"deleted": [...], "remaining": n}.
    """
    if not dir or max_mb <= 0 or not os.path.isdir(dir):
        return {"deleted": [], "remaining": 0}

    max_bytes = max_mb * 1024 * 1024
    entries = _entries(dir)
    total = sum(size for _, _, size in entries)
    if total <= max_bytes:
        return {"deleted": [], "remaining": len(entries)}

    deleted: list[str] = []
    # Oldest (lowest last-use) first; stop once under the cap.
    for _, path, size in sorted(entries, key=lambda e: e[0]):
        if total <= max_bytes:
            break
        try:
            os.remove(path)
            deleted.append(os.path.basename(path))
            total -= size
            log.info(
                "bin_cache_deleted file=%s size_mb=%.1f",
                os.path.basename(path),
                size / 1024 / 1024,
            )
        except OSError as e:
            log.warning("bin_cache_remove_fail %s: %s", path, e)

    remaining = len(entries) - len(deleted)
    log.info("bin_cache_clean deleted=%d remaining=%d", len(deleted), remaining)
    return {"deleted": deleted, "remaining": remaining}


def delete_bin_file(dir: str, basename: str) -> bool:
    """Delete a single .bin file. True if it existed."""
    if not dir:
        return False
    path = os.path.join(dir, basename)
    try:
        size = os.path.getsize(path)
        os.remove(path)
        log.info(
            "bin_cache_deleted file=%s size_mb=%.1f", basename, size / 1024 / 1024
        )
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        log.warning("bin_cache_delete_fail %s: %s", path, e)
        return False


def clear_bin_cache(dir: str) -> int:
    """Delete every .bin file in dir (tracked and orphaned). Returns count."""
    if not dir or not os.path.isdir(dir):
        return 0
    deleted = 0
    for path in glob.glob(os.path.join(dir, "*")):
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
            os.remove(path)
            deleted += 1
            log.info(
                "bin_cache_deleted file=%s size_mb=%.1f",
                os.path.basename(path),
                size / 1024 / 1024,
            )
        except OSError as e:
            log.warning("bin_cache_clear_fail %s: %s", path, e)
    log.info("bin_cache_clear deleted=%d", deleted)
    return deleted


def reconcile_bin_cache(dir: str) -> dict:
    """Reconcile meta files and .bin files in both directions.

    - A meta without a matching .bin is stale (the backend cache is gone):
      delete the meta.
    - A .bin without a matching meta is orphaned (no proxy record): delete
      the .bin.

    Returns {"deleted_metas": [...], "deleted_bins": [...]}.
    """
    if not dir or not os.path.isdir(dir):
        return {"deleted_metas": [], "deleted_bins": []}

    bin_basenames = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(dir, "*"))
        if os.path.isfile(p)
    }
    meta_basenames = {
        os.path.basename(p)[: -len(".meta.json")]
        for p in glob.glob(os.path.join(META_DIR, "*.meta.json"))
        if os.path.isfile(p)
    }

    deleted_metas: list[str] = []
    for basename in sorted(meta_basenames - bin_basenames):
        meta_path = os.path.join(META_DIR, f"{basename}.meta.json")
        try:
            os.remove(meta_path)
            deleted_metas.append(basename)
            log.info("bin_reconcile_deleted_meta %s.meta.json (no .bin)", basename)
        except OSError as e:
            log.warning("bin_reconcile_meta_fail %s: %s", meta_path, e)

    deleted_bins: list[str] = []
    for basename in sorted(bin_basenames - meta_basenames):
        bin_path = os.path.join(dir, basename)
        try:
            size = os.path.getsize(bin_path)
            os.remove(bin_path)
            deleted_bins.append(basename)
            log.info(
                "bin_reconcile_deleted_bin file=%s size_mb=%.1f (no meta)",
                basename,
                size / 1024 / 1024,
            )
        except OSError as e:
            log.warning("bin_reconcile_bin_fail %s: %s", bin_path, e)

    log.info(
        "bin_reconcile deleted_metas=%d deleted_bins=%d",
        len(deleted_metas),
        len(deleted_bins),
    )
    return {"deleted_metas": deleted_metas, "deleted_bins": deleted_bins}
