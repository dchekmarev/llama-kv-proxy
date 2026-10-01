# app/background.py

"""The background jobs and the cache-administration helpers they share with
/cache/clear: slot discovery, the stuck-slot watchdog, the eviction pass, the
.bin reconcile and the meta-index reconcile."""

import asyncio
import logging
from typing import Any, cast

import app as app_pkg
import hashing
from cache import bin_cache
from core import promstats

log = logging.getLogger(__name__)

# First poll at which a slot was seen busy: (backend index, model, slot id) ->
# time.time(). The watchdog erases a slot that stays busy too long.
_stuck_slot_first_busy: dict[tuple[int, str, int], float] = {}


async def _key_model_pairs() -> dict[str, str | None]:
    """{meta key: model id} from a full meta scan, run off the event loop."""
    metas = await asyncio.to_thread(hashing.scan_all_meta)
    pairs: dict[str, str | None] = {}
    for meta in metas or []:
        if not isinstance(meta, dict):
            continue
        key = meta.get("key")
        if isinstance(key, str) and key:
            pairs[key] = meta.get("model_id")
    return pairs


async def _delete_backend_caches(
    clients: list[Any], key_models: list[tuple[str, str | None]]
) -> None:
    """Best-effort .bin removal on every backend for the given cache keys.

    llama.cpp has no endpoint to delete a file from --slot-save-path, so this is
    only a fallback for backends whose cache dir is not mounted into the proxy;
    failures are expected and ignored. The model is passed on because a router
    backend needs it to route the delete to the right child.
    """
    for key, model_id in key_models:
        for client in clients:
            try:
                await client.delete_cache_file(key, model=model_id)
            except Exception as e:  # noqa: BLE001
                log.warning("delete_cache_file_fail key=%s: %s", key, e)


# --- background jobs ----------------------------------------------------------


async def _check_stuck_slots(
    backend_index: int, model: str, client: Any, slots: list[dict]
) -> None:
    """Erase slots that report is_processing for longer than the threshold.

    A wedged slot (llama.cpp stuck in PROCESSING_PROMPT) never finishes and the
    backend busy-loops on it; erasing its KV recovers the slot without a
    restart. The /slots payload has no per-slot timestamp, so the proxy tracks
    how long it has seen each slot busy. A threshold of 0 disables the
    watchdog, and the timer is re-armed after an erase so one wedged slot is not
    erased on every poll.
    """
    threshold = app_pkg.STUCK_SLOT_THRESHOLD_S
    if threshold <= 0:
        return
    now = app_pkg.time.time()
    for slot in slots:
        slot_id = cast(int, slot.get("id"))
        key = (backend_index, model, slot_id)
        if not slot.get("is_processing"):
            app_pkg._stuck_slot_first_busy.pop(key, None)
            continue
        first_busy = app_pkg._stuck_slot_first_busy.get(key)
        if first_busy is None:
            app_pkg._stuck_slot_first_busy[key] = now
        elif now - first_busy >= threshold:
            app_pkg._stuck_slot_first_busy[key] = now
            try:
                await client.erase_slot(slot_id, model=model)
            except Exception as e:  # noqa: BLE001
                log.warning("stuck_slot_erase_fail slot=%s: %s", key, e)
            else:
                promstats.stuck_slot_erases_total.labels(
                    backend=str(backend_index), model=model
                ).inc()
                log.warning(
                    "stuck_slot_erased slot=%s busy_for=%.0fs",
                    key,
                    now - first_busy,
                )


def _publish_slots_gauge(backend_index: int, model: str, slots: list[dict]) -> None:
    """Publish slots_total{backend,model,state} for one polled model.

    States that disappeared fall back to 0: a gauge is only as current as its
    last update, and a slot that was cut must not keep reporting forever.
    """
    counts: dict[str, int] = {}
    for slot in slots:
        state = str(slot.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    for state in {*counts, "free", "busy"}:
        promstats.slots_total.labels(
            backend=str(backend_index), model=model, state=state
        ).set(counts.get(state, 0))


async def _poll_backend(backend_index: int, client: Any) -> None:
    """Refresh one backend's slot pools and run the stuck-slot watchdog.

    A failed or unresolvable discovery keeps the previous pool: dropping it
    would send every in-flight request onto the bootstrap slot and, with a
    literal "unknown" model key, merge all models into one cache namespace.
    """
    backend = str(backend_index)
    try:
        is_router = await client.is_router()
        if is_router:
            models = await client.get_active_models()
        else:
            model = await client.get_model_id()
            if model == "unknown":
                log.warning("poll_model_unknown backend=%d", backend_index)
                models = []
            else:
                models = [model]
        for model in models:
            if not model:
                continue
            slots = (
                await client.get_slots(model=model)
                if is_router
                else await client.get_slots()
            )
            if slots is None:
                continue
            app_pkg.app.state.sm.set_backend_slots(backend_index, model, slots)
            _publish_slots_gauge(backend_index, model, slots)
            await _check_stuck_slots(backend_index, model, client, slots)
        promstats.backend_up.labels(backend=backend).set(1)
    except Exception as e:  # noqa: BLE001
        promstats.backend_up.labels(backend=backend).set(0)
        log.warning("poll_slots_failed backend=%s: %s", backend, e)


async def _poll_slots() -> None:
    """One discovery pass over every backend (GET /slots)."""
    for backend_index, client in enumerate(app_pkg.app.state.clients):
        await _poll_backend(backend_index, client)


async def _poll_slots_loop() -> None:
    """Refresh the slot pools every SLOT_POLL_INTERVAL_S (0 disables)."""
    if app_pkg.SLOT_POLL_INTERVAL_S <= 0:
        return
    while True:
        try:
            await app_pkg._poll_slots()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("poll_slots_loop_failed")
        await asyncio.sleep(app_pkg.SLOT_POLL_INTERVAL_S)


async def _run_eviction() -> None:
    """One cache-maintenance pass: meta caps, backend purge, LRU, gauges.

    The key -> model map is read before the eviction, because a deleted meta can
    no longer tell which model its .bin belongs to (a router needs it to route
    the delete to the right child).
    """
    key_models = await app_pkg._key_model_pairs()
    result = await hashing.evict_meta_async(
        ttl_hours=app_pkg.META_TTL_H,
        max_files=app_pkg.META_MAX_FILES,
        max_mb=app_pkg.META_MAX_MB,
    )
    deleted = list(result.get("deleted") or [])
    if deleted:
        await app_pkg._delete_backend_caches(
            app_pkg.app.state.clients, [(key, key_models.get(key)) for key in deleted]
        )
    if app_pkg.BIN_CACHE_DIR:
        await asyncio.to_thread(
            bin_cache.clean_bin_cache, app_pkg.BIN_CACHE_DIR, app_pkg.BIN_CACHE_MAX_MB
        )
    await asyncio.to_thread(promstats.refresh_storage_gauges)
    log.info(
        "eviction_run deleted=%d remaining=%s", len(deleted), result.get("remaining")
    )


async def _eviction_loop() -> None:
    """Run the eviction pass every EVICT_INTERVAL_S (0 disables)."""
    if app_pkg.EVICT_INTERVAL_S <= 0:
        return
    while True:
        try:
            await app_pkg._run_eviction()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("eviction_failed")
        await asyncio.sleep(app_pkg.EVICT_INTERVAL_S)


async def _bin_reconcile_loop() -> None:
    """Reconcile metas and .bin files in both directions, every interval.

    Drops a meta whose .bin is gone (the restore can never succeed) and a .bin
    whose meta is gone (nothing references it). Runs only when the .bin cache
    directory is mounted.
    """
    if app_pkg.BIN_RECONCILE_INTERVAL_S <= 0:
        return
    while True:
        await asyncio.sleep(app_pkg.BIN_RECONCILE_INTERVAL_S)
        if not app_pkg.BIN_CACHE_DIR:
            continue
        try:
            result = await asyncio.to_thread(
                bin_cache.reconcile_bin_cache, app_pkg.BIN_CACHE_DIR
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("bin_reconcile_failed")
            continue
        if result["deleted_metas"] or result["deleted_bins"]:
            log.info(
                "bin_reconcile deleted_metas=%d deleted_bins=%d",
                len(result["deleted_metas"]),
                len(result["deleted_bins"]),
            )


async def _meta_index_reconcile_loop() -> None:
    """Drop index entries whose meta file vanished, every interval.

    The index is kept live on proxy-driven writes and deletes, but external
    deletions (bin_cache reconcile/cleanup, an operator) leave ghost entries
    behind. The lookup resolves late: hashing.reconcile_index_async is read
    from the module on every round.
    """
    while True:
        await asyncio.sleep(app_pkg.META_INDEX_RECONCILE_INTERVAL_S)
        try:
            dropped = await hashing.reconcile_index_async()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("meta_index_reconcile_failed")
            continue
        if dropped:
            log.info("meta_index_reconciled dropped=%d", len(dropped))
