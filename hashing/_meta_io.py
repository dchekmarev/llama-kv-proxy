# hashing/_meta_io.py

"""The on-disk <key>.meta.json store: the document format, atomic writes, the
subsume-on-continuation pass and the async wrappers that keep the restore index
in step with the disk.
"""

import asyncio
import contextlib
import glob
import json
import os
import tempfile
import time

import hashing as hs

from ._state import log

META_SUFFIX = ".meta.json"


def _meta_path(key: str) -> str:
    return os.path.join(hs.META_DIR, key + META_SUFFIX)


def _key_of(path: str) -> str:
    """The cache key a meta file belongs to."""
    return os.path.basename(path)[: -len(META_SUFFIX)]


def _meta_files() -> list[str]:
    """Every meta file path in META_DIR (META_DIR is read per call)."""
    return glob.glob(os.path.join(hs.META_DIR, "*" + META_SUFFIX))


def _atomic_write(path: str, doc: dict) -> None:
    """Write doc as JSON through a temp file in the same dir + os.replace.

    A crash (or a full disk) mid-write leaves the previous document intact and
    removes the temp file, so a meta is never half-written and a reader never
    sees a truncated file.
    """
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _meta_doc(
    key: str,
    prefix: str,
    blocks: list[str],
    words_per_block: int,
    model_id: str,
    prefix_hashes: list[str] | None,
    bin_size: int | None,
    saved_prefix_hashes: list[str] | None,
) -> dict:
    """The meta document for one cached conversation.

    The prompt text itself is not stored, only its size (chars and UTF-8
    bytes), which is all the restore ranking needs; the blocks and the
    per-message prefix hashes are what the search matches on.
    """
    text = prefix or ""
    return {
        "key": key,
        "model_id": model_id,
        "words_per_block": words_per_block,
        "blocks": list(blocks or []),
        "prefix_len": len(text),
        "prefix_bytes": len(text.encode("utf-8")),
        "prefix_hashes": list(prefix_hashes or []),
        "saved_prefix_hashes": list(saved_prefix_hashes or []),
        "bin_size": bin_size,
        "timestamp": time.time(),
    }


def _candidate_size(meta: dict) -> int:
    """Cache size used to break ties between equally long candidates.

    The real .bin size when it is known, else the prompt's UTF-8 byte length,
    else its char length (metas written before prefix_bytes existed), else 0.
    """
    for field in ("bin_size", "prefix_bytes", "prefix_len"):
        value = meta.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value > 0:
            return int(value)
    return 0


def _prefix_hashes_of(meta: dict) -> list[str]:
    """The hash list tier 1 (and the index) matches on.

    The response-extended list wins over the prompt-only one, so a continuation
    matches a meta saved for prompt + response.
    """
    return meta.get("saved_prefix_hashes") or meta.get("prefix_hashes") or []


def _blocks_of(meta: dict) -> list[str]:
    return meta.get("blocks") or []


def write_meta(
    key: str,
    prefix: str,
    blocks: list[str],
    words_per_block: int,
    model_id: str,
    prefix_hashes: list[str] | None = None,
    bin_size: int | None = None,
    saved_prefix_hashes: list[str] | None = None,
) -> None:
    """(Re)write <key>.meta.json for a just-saved slot cache."""
    _atomic_write(
        _meta_path(key),
        _meta_doc(
            key, prefix, blocks, words_per_block, model_id,
            prefix_hashes, bin_size, saved_prefix_hashes,
        ),
    )


def touch_meta(key: str) -> None:
    """Refresh a meta's timestamp so an actively restored entry survives TTL.

    Best-effort: any failure is swallowed (a stale timestamp only risks an
    early eviction, never a wrong cache hit).
    """
    try:
        path = _meta_path(key)
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            return
        meta["timestamp"] = time.time()
        _atomic_write(path, meta)
    except (OSError, ValueError, TypeError):
        log.debug("touch_meta_failed key=%s", key[:16])


def _remove_meta_file(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def delete_meta(key: str) -> bool:
    """Delete <key>.meta.json. True only when a file was actually removed."""
    return _remove_meta_file(_meta_path(key))


def delete_subsumed_metas(
    new_key: str, new_hashes: list[str], model_id: str
) -> list[str]:
    """Delete the metas the new conversation supersedes; returns their keys.

    A meta is subsumed when its key -- always its full-conversation hash -- is
    one of the new conversation's prefix hashes: that conversation is a strict
    prefix of the new one, and its .bin is superseded by the new save. The new
    key itself and other models' metas are never touched.
    """
    subsumed = set(new_hashes or [])
    if not subsumed:
        return []
    deleted = []
    for meta in hs.scan_all_meta():
        key = meta.get("key")
        if not key or key == new_key or key not in subsumed:
            continue
        if meta.get("model_id") != model_id:
            continue
        if _remove_meta_file(_meta_path(key)):
            deleted.append(key)
    if deleted:
        log.info("subsumed_metas_deleted n=%d", len(deleted))
    return deleted


def scan_all_meta() -> list[dict]:
    """Every readable meta document on disk.

    A file that disappears (or is being written) between the glob and the read
    is skipped, not fatal: the remaining files are still worth matching.
    """
    metas = []
    for path in _meta_files():
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            metas.append(meta)
    return metas


async def scan_all_meta_async() -> list[dict]:
    """scan_all_meta off the event loop."""
    return await asyncio.to_thread(hs.scan_all_meta)


async def write_meta_async(
    key: str,
    prefix: str,
    blocks: list[str],
    words_per_block: int,
    model_id: str,
    prefix_hashes: list[str] | None = None,
    bin_size: int | None = None,
    saved_prefix_hashes: list[str] | None = None,
) -> None:
    """write_meta in a worker thread; the index is updated back on the loop."""
    await asyncio.to_thread(
        hs.write_meta,
        key,
        prefix,
        blocks,
        words_per_block,
        model_id,
        prefix_hashes,
        bin_size,
        saved_prefix_hashes,
    )
    if hs.META_INDEX_ENABLED:
        doc = _meta_doc(
            key, prefix, blocks, words_per_block, model_id,
            prefix_hashes, bin_size, saved_prefix_hashes,
        )
        hs._index.add(doc["key"], doc["model_id"], _candidate_size(doc), _prefix_hashes_of(doc))


async def delete_meta_async(key: str) -> bool:
    """delete_meta in a worker thread; the index is updated back on the loop."""
    removed = await asyncio.to_thread(hs.delete_meta, key)
    if hs.META_INDEX_ENABLED:
        hs._index.remove(key)
    return removed


async def delete_subsumed_metas_async(
    new_key: str, new_hashes: list[str], model_id: str
) -> list[str]:
    """delete_subsumed_metas in a worker thread, index updates on the loop."""
    deleted = await asyncio.to_thread(hs.delete_subsumed_metas, new_key, new_hashes, model_id)
    if hs.META_INDEX_ENABLED:
        for key in deleted:
            hs._index.remove(key)
    return deleted
