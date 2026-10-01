# hashing/_stats.py

"""The /cache/stats view of the meta store and the lifetime restore counters."""

import asyncio
import os

import hashing as hs
from core import promstats

from ._state import _OUTCOME_HIT, _OUTCOME_MISS, log


def cache_stats() -> dict:
    """Meta store state plus the lifetime restore hit/miss counters."""
    files = hs._meta_files()
    total = 0
    for path in files:
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return {
        "files": len(files),
        "total_bytes": total,
        "hits": int(promstats.counter_sum(promstats.restore_total, outcome=_OUTCOME_HIT)),
        "misses": int(promstats.counter_sum(promstats.restore_total, outcome=_OUTCOME_MISS)),
    }


async def cache_stats_async() -> dict:
    """cache_stats off the event loop."""
    return await asyncio.to_thread(hs.cache_stats)


def record_hit(model: str) -> None:
    """Count a request that restored from the cache."""
    promstats.restore_total.labels(model=model, outcome=_OUTCOME_HIT).inc()
    log.debug("restore_hit model=%s", model)


def record_miss(model: str) -> None:
    """Count a big request with no usable restore candidate."""
    promstats.restore_total.labels(model=model, outcome=_OUTCOME_MISS).inc()
    log.debug("restore_miss model=%s", model)
