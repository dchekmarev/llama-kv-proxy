# meta_index.py

"""In-memory restore-candidate index (the "B+A" design).

The on-disk cache is one ``.meta.json`` per conversation; the legacy restore
search scanned and JSON-parsed every file on each big request. This module keeps
an in-RAM index of the metas' per-message prefix hashes so the search is an
O(n) set lookup (walk the request's prefix hashes from the longest and take the
first that is a known meta) instead of a full disk scan.

Why "first hit from the end" equals the on-disk LCP: prefix hashes are
cumulative (``H(messages[:k])``), so a request hash ``R[j]`` equals a meta hash
only when the two conversations share the identical prefix of length ``j+1``.
The longest request hash present in the index is therefore exactly the longest
common prefix, matching the disk tier-1 ranking (longest prefix, smallest size
on ties).

Design constraints:
- The index is a CACHE, not the source of truth. External mutators (bin_cache
  reconcile/clean) delete meta files without notifying it, so a search validates
  a hit against the on-disk meta file (validate-on-hit) and a periodic reconcile
  drops ghost entries.
- The index is touched only from the event loop. File I/O happens in worker
  threads; the parsed result is applied here on the loop. No worker thread ever
  mutates the dicts, so no lock is needed.
- Only the per-message prefix hashes are kept (small, one per message). The
  large per-100-word ``blocks`` lists are NOT indexed; they stay on disk for the
  block-based (tier-2) fallback.
- Each hash records every meta that owns it (not just the current winner) so
  that removing a meta re-resolves the hash to the next-smallest survivor
  instead of dropping it.
"""

import logging
import os

log = logging.getLogger(__name__)


def _hash_key(h: str) -> bytes:
    """Canonical bytes for a prefix hash.

    Real sha256 hex digests become their 32 raw bytes (half the memory of the
    64-char hex string). Non-hex values (synthetic hashes in tests) fall back to
    their UTF-8 bytes. The same mapping is applied on both the index side and
    the request side, so the two always compare equal.
    """
    try:
        return bytes.fromhex(h)
    except ValueError:
        return h.encode("utf-8")


class MetaIndex:
    """In-RAM index of meta prefix hashes for O(n) restore search."""

    def __init__(self) -> None:
        # key -> (model_id, size). size is the cache size used for tie-breaks.
        self._meta: dict[str, tuple[str, int]] = {}
        # key -> its hash bytes (so remove() knows which hashes to update).
        self._key_hashes: dict[str, list[bytes]] = {}
        # hash bytes -> every key that owns it (for collision re-resolution).
        self._owners: dict[bytes, set[str]] = {}
        # hash bytes -> winning key (smallest size among owners), the lookup.
        self._winner: dict[bytes, str] = {}

    def __len__(self) -> int:
        return len(self._meta)

    def is_empty(self) -> bool:
        return not self._meta

    def add(self, key: str, model_id: str, size: int, hashes: list[str]) -> None:
        """Index a meta's effective prefix hashes (saved or prompt-only)."""
        if not key:
            return
        if key in self._meta:
            self.remove(key)
        hkeys = list(dict.fromkeys(_hash_key(h) for h in hashes))
        self._meta[key] = (model_id, size)
        self._key_hashes[key] = hkeys
        for h in hkeys:
            owners = self._owners.setdefault(h, set())
            owners.add(key)
            cur = self._winner.get(h)
            if cur is None or size < self._meta[cur][1]:
                self._winner[h] = key

    def remove(self, key: str) -> None:
        """Drop a meta and re-resolve each of its hashes to the next-smallest
        surviving owner (or delete the hash if no owner remains)."""
        if key not in self._meta:
            return
        for h in self._key_hashes[key]:
            owners = self._owners.get(h)
            if owners is None:
                continue
            owners.discard(key)
            if self._winner.get(h) != key:
                continue
            if not owners:
                del self._owners[h]
                del self._winner[h]
            else:
                self._winner[h] = min(owners, key=lambda k: self._meta[k][1])
        del self._meta[key]
        del self._key_hashes[key]

    def search(
        self,
        req_hashes: list[str],
        th: float,
        model_id: str,
        meta_dir: str,
    ) -> tuple[str, float] | None:
        """Longest-prefix restore candidate, or None.

        Walks the request's prefix hashes from the longest; the first one that is
        a known meta is the longest shared prefix. The hit is validated against
        the on-disk meta file so a ghost (deleted externally) is dropped, and the
        model must match. Returns (key, ratio) where ratio is the fraction of the
        request covered.
        """
        n = len(req_hashes)
        if n == 0 or self.is_empty():
            return None
        for j in range(n - 1, -1, -1):
            key = self._winner.get(_hash_key(req_hashes[j]))
            if key is None:
                continue
            info = self._meta.get(key)
            if info is None or info[0] != model_id:
                continue
            # Validate-on-hit: the meta may have been deleted out-of-band
            # (bin_cache reconcile/clean); drop the ghost and keep searching.
            if not os.path.exists(os.path.join(meta_dir, f"{key}.meta.json")):
                self.remove(key)
                continue
            ratio = (j + 1) / n
            if ratio >= th:
                return key, ratio
        return None

    def clear(self) -> None:
        self._meta.clear()
        self._key_hashes.clear()
        self._owners.clear()
        self._winner.clear()

    def rebuild_from(self, entries: list[dict]) -> None:
        """Replace the index contents with entries ({key, model_id, size,
        hashes}). Runs on the event loop; the disk scan that produced entries
        runs in a worker thread."""
        self.clear()
        for e in entries:
            self.add(e["key"], e["model_id"], e["size"], e["hashes"])
        log.info(
            "meta_index_rebuilt n_metas=%d n_hashes=%d",
            len(self._meta),
            len(self._winner),
        )

    def reconcile_keys(self, live_keys: set[str]) -> list[str]:
        """Drop index entries whose key is no longer on disk. Returns dropped."""
        dropped = [k for k in self._meta if k not in live_keys]
        for k in dropped:
            self.remove(k)
        if dropped:
            log.info("meta_index_reconciled dropped=%d", len(dropped))
        return dropped

    def stats(self) -> dict:
        return {"index_metas": len(self._meta), "index_hashes": len(self._winner)}
