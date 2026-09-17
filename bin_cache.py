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
import time

import promstats
from config import BIN_SAVE_GRACE_S, META_DIR

log = logging.getLogger(__name__)


def _meta_timestamp(basename: str) -> float | None:
    """Last-use time from the meta file, or None if the meta is missing."""
    path = os.path.join(META_DIR, f"{basename}.meta.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return float(json.load(f).get("timestamp") or 0)
    except (OSError, ValueError, TypeError):
        return None


def _is_recent(path: str, now: float, grace_s: float) -> bool:
    """True if the file was modified within the last grace_s seconds.

    A .bin written by an in-flight save has a fresh mtime but no meta yet;
    skipping it avoids deleting a just-saved cache before its meta lands.
    A non-positive grace window disables the guard.
    """
    if grace_s <= 0:
        return False
    try:
        return now - os.path.getmtime(path) <= grace_s
    except OSError:
        return False


def _delete_meta(basename: str) -> None:
    """Best-effort delete of the meta file for a .bin basename."""
    meta_path = os.path.join(META_DIR, f"{basename}.meta.json")
    try:
        os.remove(meta_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("bin_cache_meta_remove_fail %s: %s", meta_path, e)


def _entries(dir: str) -> list[tuple[float, str, int, bool]]:
    """(last_use, path, size, has_meta) for every file in dir.

    last_use is the meta timestamp when a meta exists, else 0 (orphaned
    files sort first and are deleted before tracked ones). has_meta reports
    whether a matching meta file exists (used to protect in-flight saves).
    """
    entries: list[tuple[float, str, int, bool]] = []
    for path in glob.glob(os.path.join(dir, "*")):
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        ts = _meta_timestamp(os.path.basename(path))
        if ts is None:
            entries.append((0.0, path, size, False))
        else:
            entries.append((ts, path, size, True))
    return entries


def clean_bin_cache(dir: str, max_mb: int) -> dict:
    """Delete oldest .bin files until total size <= max_mb.

    LRU order: meta timestamp (last save/restore); orphaned files (no meta)
    are deleted first. Returns {"deleted": [...], "remaining": n}.
    """
    if not dir or max_mb <= 0 or not os.path.isdir(dir):
        return {"deleted": [], "remaining": 0}

    now = time.time()
    max_bytes = max_mb * 1024 * 1024
    entries = _entries(dir)
    total = sum(size for _, _, size, _ in entries)
    if total <= max_bytes:
        return {"deleted": [], "remaining": len(entries)}

    deleted: list[str] = []
    # Oldest (lowest last-use) first; stop once under the cap.
    for _, path, size, has_meta in sorted(entries, key=lambda e: e[0]):
        if total <= max_bytes:
            break
        # A .bin without a meta that was just written is an in-flight save
        # (its meta may not have landed yet); skip it this round.
        if not has_meta and _is_recent(path, now, BIN_SAVE_GRACE_S):
            continue
        try:
            os.remove(path)
            basename = os.path.basename(path)
            deleted.append(basename)
            total -= size
            # Evict the meta together with the .bin: a meta without its .bin
            # is stale and would otherwise linger until the next reconcile.
            if has_meta:
                _delete_meta(basename)
            log.info(
                "bin_cache_deleted file=%s size_mb=%.1f",
                basename,
                size / 1024 / 1024,
            )
        except OSError as e:
            log.warning("bin_cache_remove_fail %s: %s", path, e)

    remaining = len(entries) - len(deleted)
    if deleted:
        promstats.evictions_total.labels(reason="lru").inc(len(deleted))
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


def get_bin_size(dir: str, key: str) -> int | None:
    """Size in bytes of the backend .bin file <dir>/<key>, or None.

    Returns None when the file or directory is missing (e.g. the .bin cache
    dir is not mounted), so callers can fall back to another size estimate.
    """
    if not dir:
        return None
    try:
        return os.path.getsize(os.path.join(dir, key))
    except OSError:
        return None


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
    if deleted:
        promstats.evictions_total.labels(reason="bin_clear").inc(deleted)
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
    now = time.time()
    for basename in sorted(bin_basenames - meta_basenames):
        bin_path = os.path.join(dir, basename)
        # A fresh .bin without a meta is an in-flight save (its meta may land
        # shortly); skip it so a just-saved cache is not deleted as an orphan.
        if _is_recent(bin_path, now, BIN_SAVE_GRACE_S):
            continue
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

    n_deleted = len(deleted_metas) + len(deleted_bins)
    if n_deleted:
        promstats.evictions_total.labels(reason="reconcile").inc(n_deleted)
    log.info(
        "bin_reconcile deleted_metas=%d deleted_bins=%d",
        len(deleted_metas),
        len(deleted_bins),
    )
    return {"deleted_metas": deleted_metas, "deleted_bins": deleted_bins}
