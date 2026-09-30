# chat_flow/_restore.py

"""Restore-candidate selection, in-flight save waiting and restore settling."""

import asyncio

import chat_flow
import hashing as hs
import promstats
from llama_client import LlamaClient

from . import _state

log = _state.log


def _register_pending_restore(key: str) -> None:
    chat_flow._PENDING_RESTORES[key] = chat_flow._PENDING_RESTORES.get(key, 0) + 1


def _unregister_pending_restore(key: str) -> None:
    n = chat_flow._PENDING_RESTORES.get(key, 0) - 1
    if n <= 0:
        chat_flow._PENDING_RESTORES.pop(key, None)
        # No waiter depends on the key anymore: its substitution (if any) is
        # stale, drop it so the alias table cannot grow unbounded.
        chat_flow._RESTORE_ALIAS.pop(key, None)
    else:
        chat_flow._PENDING_RESTORES[key] = n


def _resolve_restore_key(key: str) -> str:
    """Follow substitution aliases to the live cache a pending waiter must
    restore. The chain is acyclic: each alias points to a longer conversation
    that superseded the previous one."""
    while key in chat_flow._RESTORE_ALIAS:
        key = chat_flow._RESTORE_ALIAS[key]
    return key


def _register_inflight_save(key: str) -> None:
    # Idempotent: _background_save registers the key before the response-hash
    # phase and _save_and_write_meta re-registers it; the second call must not
    # replace the live event that waiters are already awaiting.
    if key not in chat_flow._INFLIGHT_SAVES:
        chat_flow._INFLIGHT_SAVES[key] = asyncio.Event()


def _finish_inflight_save(key: str) -> None:
    ev = chat_flow._INFLIGHT_SAVES.pop(key, None)
    if ev is not None:
        ev.set()


async def _wait_for_inflight_save(prefix_hashes: list[str]) -> bool:
    """Wait for an in-flight save of a prefix of this conversation.

    Returns True when a matching save was found and completed (the caller
    must re-run the restore search), False when there is nothing to wait for
    or the wait timed out (proceed without a restore).
    """
    if chat_flow.SAVE_WAIT_TIMEOUT <= 0:
        return False
    req_hashes = set(prefix_hashes)
    ev = next((e for k, e in chat_flow._INFLIGHT_SAVES.items() if k in req_hashes), None)
    if ev is None:
        return False
    try:
        await asyncio.wait_for(ev.wait(), timeout=chat_flow.SAVE_WAIT_TIMEOUT)
    except TimeoutError:
        log.warning("inflight_save_wait_timeout timeout_s=%.1f", chat_flow.SAVE_WAIT_TIMEOUT)
        return False
    log.info("inflight_save_waited")
    return True


async def _find_restore_candidate(
    prefix_hashes: list[str], blocks: list[str], model_id: str
) -> tuple[str, float] | None:
    return await hs.find_best_restore_candidate_async(
        prefix_hashes, blocks, chat_flow.WORDS_PER_BLOCK, chat_flow.LCP_TH, model_id
    )


async def _select_restore_candidate(
    prefix_hashes: list[str],
    blocks: list[str],
    model_id: str,
    do_cache: bool,
    no_cache: bool,
    n_words: int,
    decision: dict,
) -> str | None:
    # Per-message prefix hashes (last == key): the meta is findable by any of
    # its prefixes, and a continuation supersedes its strict prefixes. Only
    # big requests restore or save, so the candidate search is lazy.
    restore_key: str | None = None
    restore_ratio: float | None = None
    if do_cache:
        cand = await chat_flow._find_restore_candidate(prefix_hashes, blocks, model_id)
        if cand is None and await chat_flow._wait_for_inflight_save(prefix_hashes):
            decision["wait_inflight_save"] = True
            # The previous message's save finished while we waited: its meta
            # is on disk now, search again before declaring a miss.
            cand = await chat_flow._find_restore_candidate(prefix_hashes, blocks, model_id)
            promstats.inflight_save_waits_total.labels(
                model=model_id, result="hit" if cand else "miss"
            ).inc()
        if cand:
            restore_key, restore_ratio = cand
            decision["restore"]["candidate_key"] = restore_key
            decision["restore"]["candidate_ratio"] = restore_ratio
            hs.record_hit(model_id)
            promstats.restore_ratio.labels(model=model_id).observe(restore_ratio)
            # Refresh the meta timestamp so an actively used entry survives
            # TTL eviction.
            await asyncio.to_thread(hs.touch_meta, restore_key)
            log.info(
                "restore_candidate basename=%s ratio=%.3f",
                restore_key[:16],
                restore_ratio,
            )
        else:
            hs.record_miss(model_id)
            log.info("restore_candidate none")
    elif no_cache:
        log.info("no_cache_request n_words=%d (proxied without cache)", n_words)
    else:
        log.info(
            "small_request n_words=%d threshold=%d",
            n_words,
            chat_flow.BIG_THRESHOLD_WORDS,
        )
    return restore_key


async def _settle_restore(
    client: LlamaClient,
    slot_id: int,
    model_id: str,
    restored: object,
    used_key: str | None,
    no_cache: bool,
    decision: dict,
) -> None:
    # A restore is only attempted when a key was selected, so both branches
    # below imply used_key is a non-None string: the key actually restored
    # (the original candidate, or its substitution alias target).
    if restored == chat_flow.RESTORE_MISSING and used_key:
        # The backend explicitly reported the cache file is gone (404): the
        # meta is stale, delete it so every big request does not repeat a
        # hopeless restore. With substitution, used_key is the replacement
        # whose .bin is gone, not the original candidate (whose meta the
        # subsumption that created the alias already deleted).
        try:
            await hs.delete_meta_async(used_key)
            decision["restore"]["stale_meta_dropped"] = True
            promstats.stale_meta_drops_total.labels(model=model_id).inc()
            promstats.evictions_total.labels(reason="stale").inc()
            log.info("stale_meta_dropped key=%s", used_key[:16])
        except Exception as e:  # noqa: BLE001
            log.warning("delete_meta_failed key=%s: %s", used_key[:16], e)
    elif restored is not True and restored != chat_flow.RESTORE_MISSING and used_key:
        # A non-missing restore failure: a plain False is a definitive miss,
        # RESTORE_ERROR is "the restore request itself failed" (backend
        # down/transient). In both cases the cache may still be valid and a
        # retry can succeed, so keep the meta.
        log.warning("restore_failed_kept_meta key=%s", used_key[:16])

    # A successful restore already set the slot's prompt to the correct prefix,
    # so it is left untouched. In every other case (small request, big request
    # with no restore hit, or a failed/missing restore) the slot may still hold
    # a stale or oversized prompt from a previous conversation; starting a chat
    # on top of it can wedge llama.cpp in PROCESSING_PROMPT (the slot never
    # finishes and the server busy-loops). Clear the slot first. A no-cache
    # request is proxied onto the slot untouched, so it never erases.
    erase_done = (not no_cache) and restored is not True and chat_flow.ERASE_BEFORE_CHAT
    decision["erase_done"] = erase_done
    if erase_done:
        await client.erase_slot(slot_id, model=model_id)
