# hashing/_search.py

"""The restore search: the in-RAM index first, the two-tier on-disk scan second.
"""

import asyncio
from collections.abc import Callable

import hashing as hs
import promstats


def _lcp_len(request: list[str], candidate: list[str]) -> int:
    """How many leading elements the request and the candidate share."""
    shared = 0
    for want, have in zip(request, candidate):
        if want != have:
            break
        shared += 1
    return shared


def _is_candidate(meta: dict, model_id: str, words_per_block: int) -> bool:
    """Both tiers only consider metas of the same model and block size."""
    if meta.get("model_id") != model_id:
        return False
    cached_wpb = meta.get("words_per_block")
    return cached_wpb is None or cached_wpb == words_per_block


def _best_match(
    metas: list[dict], request: list[str], hashes_of: Callable[[dict], list[str]]
) -> tuple[str, int] | None:
    """Longest common prefix over the metas, smallest cache on ties."""
    best_key: str | None = None
    best_shared = 0
    best_size = 0
    for meta in metas:
        key = meta.get("key")
        if not key:
            continue
        shared = _lcp_len(request, hashes_of(meta))
        if not shared:
            continue
        size = hs._candidate_size(meta)
        if shared > best_shared or (shared == best_shared and size < best_size):
            best_key, best_shared, best_size = key, shared, size
    return (best_key, best_shared) if best_key else None


def find_best_restore_candidate(
    req_hashes: list[str] | None,
    req_blocks: list[str] | None,
    words_per_block: int,
    th: float,
    model_id: str,
) -> tuple[str, float] | None:
    """Best restore candidate for a big request, or None. On-disk tiers only.

    Both tiers filter by model_id and words_per_block, and both score a
    candidate by the fraction of the REQUEST it covers (lcp / len(request)), so
    a long partial match beats a short full one. Tier 1 wins outright when it
    matches at all -- including when its best match falls below the threshold,
    in which case the request has no usable cache:

    1. tier 1: per-message prefix hashes, longest common prefix, smallest cache
       on ties. The response-extended hash list wins over the prompt-only one,
       so a continuation matches a meta saved for prompt + response.
    2. tier 2: block hashes, the fallback for metas written before per-message
       hashes existed.
    """
    hashes = list(req_hashes or [])
    blocks = list(req_blocks or [])
    if not hashes and not blocks:
        return None
    metas = [m for m in hs.scan_all_meta() if _is_candidate(m, model_id, words_per_block)]
    if hashes:
        best = _best_match(metas, hashes, hs._prefix_hashes_of)
        if best is not None:
            ratio = best[1] / len(hashes)
            if ratio >= th:
                promstats.restore_tier_total.labels(tier="t1_disk").inc()
                return best[0], ratio
            return None
    if blocks:
        best = _best_match(metas, blocks, hs._blocks_of)
        if best is not None and best[1] / len(blocks) >= th:
            ratio = best[1] / len(blocks)
            promstats.restore_tier_total.labels(tier="t2_blocks").inc()
            return best[0], ratio
    return None


async def find_best_restore_candidate_async(
    req_hashes: list[str] | None,
    req_blocks: list[str] | None,
    words_per_block: int,
    th: float,
    model_id: str,
) -> tuple[str, float] | None:
    """find_best_restore_candidate off the event loop, index first.

    With META_INDEX_ENABLED the in-RAM index answers tier 1 in O(n) instead of
    scanning and parsing every meta file. It validates a hit against the
    on-disk meta (dropping ghosts) and ignores words_per_block, so it can be
    more permissive; a miss falls through to the full two-tier disk scan, which
    also covers metas the index does not know (blocks-only legacy entries).
    """
    hashes = list(req_hashes or [])
    if hs.META_INDEX_ENABLED and hashes:
        hit = hs._index.search(hashes, th, model_id, hs.META_DIR)
        if hit is not None:
            promstats.restore_tier_total.labels(tier="index").inc()
            return hit
    return await asyncio.to_thread(
        hs.find_best_restore_candidate, req_hashes, req_blocks, words_per_block, th, model_id
    )
