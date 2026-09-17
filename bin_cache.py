# bin_cache.py

"""
Direct filesystem cleanup of backend .bin cache files.

llama.cpp has no endpoint to delete files from --slot-save-path: the `erase`
slot action only clears in-memory state, and save/restore only write/read.
So when the save directory is mounted into the proxy we remove .bin files
ourselves.

Each .bin may have a .ckpt checkpoint sidecar (written by llama.cpp next to
the slot blob); it is always evicted together with its .bin and counted in
the size cap.

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

# llama.cpp writes a prompt-checkpoint sidecar next to each saved slot blob
# (server-context.cpp save_slot_checkpoints, magic LSCKPT3). It must be
# evicted together with its .bin and counted in the size cap.
CKPT_SUFFIX = ".ckpt"


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


def _remove_ckpt(bin_path: str) -> None:
    """Best-effort delete of the .ckpt sidecar paired with a .bin path."""
    ckpt_path = bin_path + CKPT_SUFFIX
    try:
        os.remove(ckpt_path)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("bin_cache_ckpt_remove_fail %s: %s", ckpt_path, e)


def _entries(dir: str) -> list[tuple[float, str, int, bool]]:
    """(last_use, path, size, has_meta) for every .bin file in dir.

    A .ckpt sidecar's size is folded into its .bin entry so LRU evicts the
    pair together; a .ckpt whose .bin is missing is listed on its own as an
    orphan (last_use 0, has_meta False).

    last_use is the meta timestamp when a meta exists, else 0 (orphaned
    files sort first and are deleted before tracked ones). has_meta reports
    whether a matching meta file exists (used to protect in-flight saves).
    """
    files: dict[str, int] = {}
    for path in glob.glob(os.path.join(dir, "*")):
        if not os.path.isfile(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        files[os.path.basename(path)] = size

    # Accumulate (not assign): a sidecar may be iterated before its .bin and
    # must not be clobbered when the .bin arrives.
    sizes: dict[str, int] = {}
    for name, size in files.items():
        owner = name.removesuffix(CKPT_SUFFIX)
        if owner != name and owner in files:
            sizes[owner] = sizes.get(owner, 0) + size
        else:
            sizes[name] = sizes.get(name, 0) + size

    entries: list[tuple[float, str, int, bool]] = []
    for name, size in sizes.items():
        ts = _meta_timestamp(name)
        if ts is None:
            entries.append((0.0, os.path.join(dir, name), size, False))
        else:
            entries.append((ts, os.path.join(dir, name), size, True))
    return entries


def clean_bin_cache(dir: str, max_mb: int) -> dict:
    """Delete oldest .bin files until total size <= max_mb.

    LRU order: meta timestamp (last save/restore); orphaned files (no meta)
    are deleted first. A .ckpt sidecar is evicted together with its .bin and
    counts toward the cap. Returns {"deleted": [...], "remaining": n}.
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
            # Evict the checkpoint sidecar together with the .bin: a .ckpt
            # without its .bin is useless and would linger until reconcile.
            _remove_ckpt(path)
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
    """Delete a single .bin file and its .ckpt sidecar. True if it existed."""
    if not dir:
        return False
    path = os.path.join(dir, basename)
    try:
        size = os.path.getsize(path)
        os.remove(path)
        _remove_ckpt(path)
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
      the .bin and its .ckpt sidecar.
    - A .ckpt without its .bin is orphaned: delete the .ckpt.

    Returns {"deleted_metas": [...], "deleted_bins": [...]}.
    """
    if not dir or not os.path.isdir(dir):
        return {"deleted_metas": [], "deleted_bins": []}

    bin_basenames: set[str] = set()
    ckpt_basenames: set[str] = set()
    for p in glob.glob(os.path.join(dir, "*")):
        if not os.path.isfile(p):
            continue
        name = os.path.basename(p)
        if name.endswith(CKPT_SUFFIX):
            ckpt_basenames.add(name)
        else:
            bin_basenames.add(name)
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
            _remove_ckpt(bin_path)
            deleted_bins.append(basename)
            log.info(
                "bin_reconcile_deleted_bin file=%s size_mb=%.1f (no meta)",
                basename,
                size / 1024 / 1024,
            )
        except OSError as e:
            log.warning("bin_reconcile_bin_fail %s: %s", bin_path, e)

    # A .ckpt whose .bin is gone is an orphan (the pair was evicted partially
    # or the backend removed the blob); drop it.
    for name in sorted(
        c for c in ckpt_basenames if c.removesuffix(CKPT_SUFFIX) not in bin_basenames
    ):
        ckpt_path = os.path.join(dir, name)
        if _is_recent(ckpt_path, now, BIN_SAVE_GRACE_S):
            continue
        try:
            size = os.path.getsize(ckpt_path)
            os.remove(ckpt_path)
            deleted_bins.append(name)
            log.info(
                "bin_reconcile_deleted_bin file=%s size_mb=%.1f (no .bin)",
                name,
                size / 1024 / 1024,
            )
        except OSError as e:
            log.warning("bin_reconcile_bin_fail %s: %s", ckpt_path, e)

    n_deleted = len(deleted_metas) + len(deleted_bins)
    if n_deleted:
        promstats.evictions_total.labels(reason="reconcile").inc(n_deleted)
    log.info(
        "bin_reconcile deleted_metas=%d deleted_bins=%d",
        len(deleted_metas),
        len(deleted_bins),
    )
    return {"deleted_metas": deleted_metas, "deleted_bins": deleted_bins}
