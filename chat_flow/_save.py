# chat_flow/_save.py

"""Slot save, meta write, subsumed-meta cleanup and the background save task."""

import bin_cache
import chat_flow
import hashing as hs
import promstats
import reqlog
from llama_client import LlamaClient
from slot_manager import GSlot, SlotManager

from . import _state

log = _state.log


async def _save_and_write_meta(
    clients: list[LlamaClient] | None,
    sm: SlotManager,
    g: GSlot,
    key: str,
    prefix: str,
    blocks: list[str],
    prefix_hashes: list[str],
    model_id: str,
    saved_prefix_hashes: list[str] | None = None,
    decision: dict | None = None,
) -> bool:
    """Save the slot, write its meta, then drop the metas it supersedes.

    Shared by the stream and non-stream save paths (DRY). Returns True only
    when the slot save succeeded (the meta is then written and the LRU check
    scheduled). The meta records saved_prefix_hashes for the stored prompt +
    response, while subsumed deletion still uses the incoming request prompt
    hashes. The new conversation supersedes every strict prefix of itself:
    those metas and their backend .bin files are removed so a continuation
    does not leave stale, shorter caches behind.

    The save is registered in _INFLIGHT_SAVES for its whole duration so a
    continuation request can wait for the meta instead of missing the restore
    (see _wait_for_inflight_save); the entry is cleared on every exit path.
    """
    chat_flow._register_inflight_save(key)
    try:
        try:
            ok = await sm.save_after(g, key)
        except Exception as e:  # noqa: BLE001
            log.warning("save_after_exception g=%s key=%s: %s", g, key[:16], e)
            promstats.saves_total.labels(model=model_id, outcome="save_error").inc()
            chat_flow._set_save_outcome(decision, success=False, phase="save_error")
            return False
        if not ok:
            promstats.saves_total.labels(model=model_id, outcome="save_failed").inc()
            chat_flow._set_save_outcome(decision, success=False, phase="save_failed")
            return False
        bin_size = bin_cache.get_bin_size(chat_flow.BIN_CACHE_DIR, key) if chat_flow.BIN_CACHE_DIR else None
        if (
            bin_size is not None
            and chat_flow.MIN_BIN_SIZE_VALID > 0
            and bin_size < chat_flow.MIN_BIN_SIZE_VALID * 1024 * 1024
        ):
            log.warning(
                "save_empty_capture g=%s key=%s bin_size=%d discard",
                g,
                key[:16],
                bin_size,
            )
            try:
                bin_cache.delete_bin_file(chat_flow.BIN_CACHE_DIR, key)
            except Exception:
                log.warning(
                    "save_empty_capture g=%s key=%s bin_delete_failed", g, key[:16], exc_info=True
                )
            promstats.saves_total.labels(model=model_id, outcome="empty_capture").inc()
            # save_after already recorded this key; the capture is gone now.
            sm.forget_saved(g, key)
            chat_flow._set_save_outcome(decision, success=False, phase="empty_capture", bin_size_bytes=bin_size)
            return False
        meta_written = False
        try:
            await hs.write_meta_async(
                key,
                prefix,
                blocks,
                chat_flow.WORDS_PER_BLOCK,
                model_id,
                prefix_hashes,
                bin_size,
                saved_prefix_hashes or prefix_hashes,
            )
            meta_written = True
        except Exception as e:  # noqa: BLE001
            log.warning("write_meta_exception key=%s: %s", key[:16], e)
        # Only drop the superseded metas once the new meta is on disk:
        # otherwise a failed meta write would delete the still-valid shorter
        # caches.
        if meta_written:
            try:
                deleted = await hs.delete_subsumed_metas_async(
                    key, prefix_hashes, model_id
                )
                if deleted:
                    # A deleted key that another big request is waiting to
                    # restore maps to this longer cache (its meta and .bin are
                    # already on disk): the waiter must restore it instead, or
                    # the now-deleted .bin makes its restore fail. Alias BEFORE
                    # the file is purged so a concurrent restore resolves it.
                    for k in deleted:
                        if chat_flow._PENDING_RESTORES.get(k, 0) > 0:
                            chat_flow._RESTORE_ALIAS[k] = key
                            log.info(
                                "restore_substituted pending=%s replacement=%s",
                                k[:16],
                                key[:16],
                            )
                    # An existing alias whose target was just deleted would
                    # otherwise terminate at a purged cache; re-point it to
                    # this live key so _resolve_restore_key stays valid.
                    deleted_set = set(deleted)
                    for x, target in list(chat_flow._RESTORE_ALIAS.items()):
                        if target in deleted_set:
                            chat_flow._RESTORE_ALIAS[x] = key
                            log.info(
                                "restore_alias_repointed alias=%s replacement=%s",
                                x[:16],
                                key[:16],
                            )
                    await chat_flow._purge_backend_files(
                        clients or [], [(k, model_id) for k in deleted]
                    )
                    log.info(
                        "subsumed_metas_deleted key=%s count=%d",
                        key[:16],
                        len(deleted),
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("delete_subsumed_exception key=%s: %s", key[:16], e)
        promstats.saves_total.labels(model=model_id, outcome="ok").inc()
        chat_flow._set_save_outcome(decision, success=True, meta_written=meta_written, bin_size_bytes=bin_size)
        chat_flow._schedule_lru_check()
        return True
    finally:
        chat_flow._finish_inflight_save(key)


async def _background_save(
    clients: list[LlamaClient],
    sm: SlotManager,
    g: GSlot,
    key: str,
    prefix: str,
    blocks: list[str],
    prefix_hashes: list[str],
    model_id: str,
    messages: list[dict],
    response_text: str,
    response_reasoning: str = "",
    response_reasoning_field: str = "reasoning_content",
    rid: str = "",
    ts: str = "",
    decision: dict | None = None,
    render_ctx: dict | None = None,
) -> None:
    """Non-stream big-request save+meta, run after the response is returned.

    The hundreds-of-MB .bin write must not add latency to the JSON response,
    so it runs in a background task (mirroring the stream reader's finally).
    The task owns the slot: it releases it in the finally, even if the save
    or the meta write raises (the client already has its response, so the
    error is only logged).

    The save is registered in _INFLIGHT_SAVES at the very start, before the
    response-hash phase, so a continuation arriving right after the response
    can wait for it (see _wait_for_inflight_save); the entry is cleared in
    the finally on every exit path.
    """
    try:
        chat_flow._register_inflight_save(key)
        try:
            saved_prefix, saved_blocks, saved_hashes = (
                await chat_flow._saved_conversation_values(
                    messages,
                    response_text,
                    model_id,
                    prefix,
                    blocks,
                    prefix_hashes,
                    response_reasoning,
                    response_reasoning_field,
                    render_ctx,
                )
            )
        except Exception as e:  # noqa: BLE001
            log.warning("saved_conversation_values_fail key=%s: %s", key[:16], e)
            saved_prefix, saved_blocks, saved_hashes = (
                prefix,
                blocks,
                prefix_hashes,
            )
        # The request group's prefix.json reuses the values computed above
        # (no second tokenization): the conversation the next message's
        # request will be matched against.
        if rid and ts:
            reqlog.log_file(
                "prefix",
                rid,
                ts,
                {
                    "prefix": saved_prefix,
                    "key": saved_hashes[-1] if saved_hashes else None,
                },
            )
        ok = await chat_flow._save_and_write_meta(
            clients,
            sm,
            g,
            key,
            saved_prefix,
            saved_blocks,
            prefix_hashes,
            model_id,
            saved_hashes,
            decision=decision,
        )
        log.info("bg_save_done g=%s key=%s saved=%s", g, key[:16], ok)
    except Exception as e:  # noqa: BLE001
        log.warning("background_save_error g=%s key=%s: %s", g, key[:16], e)
    finally:
        # Clear the in-flight entry on every exit path (a no-op when
        # _save_and_write_meta already finished it).
        chat_flow._finish_inflight_save(key)
        if decision is not None:
            decision.setdefault("save", {"attempted": True, "success": False, "phase": "save_error"})
            chat_flow._emit_decision(decision, rid, ts)
        # Release is guaranteed: a synchronous call that cannot be
        # interrupted; it runs even if a re-cancellation interrupts the save.
        log.info("slot_release g=%s key=%s via=bg_save", g, key[:16])
        sm.release(g)
