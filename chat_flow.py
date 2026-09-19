# chat_flow.py

# -*- coding: utf-8 -*-

"""Chat request orchestration for llama-kv-proxy.

Holds the /v1/chat/completions request pipeline: effective-model resolution,
cache-key computation, restore-candidate selection, slot acquisition, dispatch
to the backend, and the save/meta/LRU follow-up. The FastAPI endpoint in
app.py only parses the JSON body and delegates here.

Additionally:

- acquire_for_request is wrapped in a timeout so it cannot hang forever if a
  slot is never released.
- For non-streaming and streaming big requests the save+meta runs in a
  background task (_BG_SAVE_TASKS), so the .bin write does not add latency
  and survives a client disconnect (the task owns and releases the slot).
- For streaming:
    * reading from llama.cpp happens in a separate background task (the
      reader);
    * the reader pushes chunks into an asyncio.Queue (put with bounded wait,
      so it cannot block forever when the consumer is gone);
    * on a big-completed stream the reader hands off the slot to a detached
      _background_save task (which releases it in its own finally); on any
      other path the reader releases the slot itself;
    * a sentinel None is pushed into the queue (with bounded wait) after the
      slot is released or handed off;
    * on a mid-stream backend error the reader pushes an SSE error event
      (data: {"error": "stream interrupted: ..."}) before the sentinel, so
      the client can distinguish a truncated stream from a normal one;
    * when the generator is closed (client disconnect) the reader is
      cancelled;
    * reader tasks are kept by strong references (_READER_TASKS) so they are
      not GC-collected before their finally runs.
"""

import asyncio
import codecs
import json
import logging
import time
from collections.abc import AsyncGenerator

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

import bin_cache
import hashing as hs
import promstats
import reqlog
import ui as ui_obs
from config import (
    ACQUIRE_TIMEOUT,
    BIG_THRESHOLD_WORDS,
    BIN_CACHE_DIR,
    BIN_CACHE_MAX_MB,
    ERASE_BEFORE_CHAT,
    LCP_TH,
    MIN_BIN_SIZE_VALID,
    MODEL_ID,
    REASONING_IN_KEY,
    RENDER_CTX_FIELDS,
    SAVE_WAIT_TIMEOUT,
    WORDS_PER_BLOCK,
)
from llama_client import RESTORE_MISSING, LlamaClient
from request_id import request_id_var
from slot_manager import GSlot, SlotManager

log = logging.getLogger(__name__)

STREAM_QUEUE_SIZE = 16
# Bounded wait on queue push: if the consumer disappears (client
# disconnected and the generator is closed), the reader must not block on
# put forever — otherwise its finally (and the slot release) would never run.
STREAM_PUT_TIMEOUT = 60.0

# Strong references to running reader tasks: the event loop keeps only
# weak references to tasks; without this a reader could be GC-collected
# mid-execution and skip its finally (slot release, response aclose).
_READER_TASKS: "set[asyncio.Task]" = set()

# Strong references to in-flight LRU check tasks (same GC reason as above)
# and a guard against concurrent checks racing on the same files.
_LRU_TASKS: "set[asyncio.Task]" = set()
_lru_check_in_flight = False

# Strong references to in-flight non-stream background save tasks (same GC
# reason as _READER_TASKS): the .bin write must not delay the JSON response.
_BG_SAVE_TASKS: "set[asyncio.Task]" = set()

# In-flight saves: request key -> completion event. The client treats [DONE]
# as the end of the response and sends the continuation immediately, but the
# previous message's meta only lands after its .bin write finishes. The
# previous request is a strict prefix of the continuation's request, so its
# key is among the continuation's prefix hashes: a continuation that misses
# the restore search can detect the relevant in-flight save and wait for it
# instead of reprocessing the whole prompt.
_INFLIGHT_SAVES: dict[str, asyncio.Event] = {}

# Keys that big requests have selected for restore and are now waiting to
# acquire a slot for (refcount of waiters per key). A concurrent save that
# subsumes such a key deletes its .bin; the waiter must restore the longer
# cache that superseded it instead (substitution), or the restore fails with
# "file not found" and the whole prompt is reprocessed.
_PENDING_RESTORES: dict[str, int] = {}
# Deleted-key -> replacement-key for pending restores. Newer saves only alias
# keys that are still awaited; once no waiter depends on a key its alias is
# dropped (see _unregister_pending_restore).
_RESTORE_ALIAS: dict[str, str] = {}


def _register_pending_restore(key: str) -> None:
    _PENDING_RESTORES[key] = _PENDING_RESTORES.get(key, 0) + 1


def _unregister_pending_restore(key: str) -> None:
    n = _PENDING_RESTORES.get(key, 0) - 1
    if n <= 0:
        _PENDING_RESTORES.pop(key, None)
        # No waiter depends on the key anymore: its substitution (if any) is
        # stale, drop it so the alias table cannot grow unbounded.
        _RESTORE_ALIAS.pop(key, None)
    else:
        _PENDING_RESTORES[key] = n


def _resolve_restore_key(key: str) -> str:
    """Follow substitution aliases to the live cache a pending waiter must
    restore. The chain is acyclic: each alias points to a longer conversation
    that superseded the previous one."""
    while key in _RESTORE_ALIAS:
        key = _RESTORE_ALIAS[key]
    return key


def _register_inflight_save(key: str) -> None:
    # Idempotent: _background_save registers the key before the response-hash
    # phase and _save_and_write_meta re-registers it; the second call must not
    # replace the live event that waiters are already awaiting.
    if key not in _INFLIGHT_SAVES:
        _INFLIGHT_SAVES[key] = asyncio.Event()


def _finish_inflight_save(key: str) -> None:
    ev = _INFLIGHT_SAVES.pop(key, None)
    if ev is not None:
        ev.set()


async def _wait_for_inflight_save(prefix_hashes: list[str]) -> bool:
    """Wait for an in-flight save of a prefix of this conversation.

    Returns True when a matching save was found and completed (the caller
    must re-run the restore search), False when there is nothing to wait for
    or the wait timed out (proceed without a restore).
    """
    if SAVE_WAIT_TIMEOUT <= 0:
        return False
    req_hashes = set(prefix_hashes)
    ev = next((e for k, e in _INFLIGHT_SAVES.items() if k in req_hashes), None)
    if ev is None:
        return False
    try:
        await asyncio.wait_for(ev.wait(), timeout=SAVE_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("inflight_save_wait_timeout timeout_s=%.1f", SAVE_WAIT_TIMEOUT)
        return False
    log.info("inflight_save_waited")
    return True


async def _find_restore_candidate(
    prefix_hashes: list[str], blocks: list[str], model_id: str
) -> tuple[str, float] | None:
    return await hs.find_best_restore_candidate_async(
        prefix_hashes, blocks, WORDS_PER_BLOCK, LCP_TH, model_id
    )


def _emit_decision(decision: dict, rid: str, ts: str) -> None:
    """Persist the per-request cache decision (fire-and-forget, idempotent).

    The same decision dict is threaded through the request pipeline (restore
    phase in chat_flow, save outcome in the background save), so exactly one
    writer emits it — the task that finishes last, or the small non-stream
    path in chat_flow. The "_emitted" marker is never serialized.
    """
    if not rid or not ts or decision.pop("_emitted", False):
        return
    decision["_emitted"] = True
    reqlog.log_file(
        "decision",
        rid,
        ts,
        {k: v for k, v in decision.items() if k != "_emitted"},
    )


def _set_save_outcome(decision: dict | None, **kw: object) -> None:
    """Record the save result in the decision dict (no-op without one)."""
    if decision is not None:
        decision["save"] = {"attempted": True, **kw}


def _record_tokens(model_id: str, usage: dict | None) -> None:
    """Count backend-reported tokens (prompt/completion/cached) into metrics."""
    if not isinstance(usage, dict):
        return
    for key, kind in (
        ("prompt_tokens", "prompt"),
        ("completion_tokens", "completion"),
        ("prompt_cached_tokens", "cached"),
    ):
        v = usage.get(key)
        if isinstance(v, (int, float)) and v > 0:
            promstats.tokens_total.labels(model=model_id, kind=kind).inc(v)


async def _snapshot_slot(
    client: LlamaClient, slot_id: int, model_id: str
) -> dict | None:
    """KV state of the just-restored slot (GET /slots), or None if unavailable.

    The recorded fields are the llama.cpp /slots subset that distinguishes a
    restore that populated the KV cache (n_past == restored prefix tokens)
    from one the backend reset (n_past == 0): the key datapoint for "restore
    reported ok but the whole prompt is reprocessed" investigations. Best
    effort — a missing /slots endpoint just yields None.
    """
    try:
        slots = await client.get_slots(model=model_id)
    except Exception:  # noqa: BLE001
        log.debug("slot_snapshot_fail slot=%d model=%s", slot_id, model_id)
        return None
    if isinstance(slots, list):
        for s in slots:
            if isinstance(s, dict) and s.get("id") == slot_id:
                return {
                    k: s.get(k)
                    for k in ("state", "n_ctx", "n_past", "n_tokens", "total_tokens")
                    if k in s
                }
    return None


def _provider_error_status(status: object) -> int:
    """Map a provider failure to the HTTP status for the client.

    A backend 4xx is a client fault and passes through unchanged (so client
    5xx retry/alerting logic does not fire); everything else (backend 5xx,
    connect errors, non-JSON bodies, missing/malformed status) is a genuine
    upstream failure and maps to 502. Shared by the stream and non-stream
    dispatch paths so both stay consistent (M-8).
    """
    if isinstance(status, int) and 400 <= status < 500:
        return status
    return 502


def _render_ctx_of(data: dict) -> dict:
    """The render-affecting request params (tools, reasoning_effort, ...).

    These fields change the PROMPT the backend template renders (tools and
    reasoning instructions land inside the system text), so they must be part
    of the cache key: two requests with identical messages but different
    render params must never share a KV cache. Explicit nulls and absent
    fields are omitted, so their canonical form is empty.
    """
    return {
        k: data[k] for k in RENDER_CTX_FIELDS if k in data and data[k] is not None
    }


def _reasoning_of(msg: dict) -> tuple[object, str]:
    """(reasoning value, field name) from a message/delta dict.

    reasoning_content takes precedence over the reasoning field variant; the
    field name is "reasoning_content" when neither is present. Shared by the
    non-stream and stream extraction paths so both stay in parity.
    """
    value = msg.get("reasoning_content")
    if value is not None:
        return value, "reasoning_content"
    value = msg.get("reasoning")
    if value is not None:
        return value, "reasoning"
    return None, "reasoning_content"


def _assistant_content(out: dict) -> tuple[str, str, str]:
    """(Assistant text, reasoning text, reasoning field name) from a non-stream
    chat completion body.

    The reasoning text (reasoning_content, else reasoning) and the field name
    the backend used are only used for the saved-conversation values when
    REASONING_IN_KEY is on; they are always extracted so the caller decides.
    The field name is "reasoning_content" when no reasoning is present.
    """
    choices = out.get("choices") if isinstance(out, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return "", "", "reasoning_content"
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return "", "", "reasoning_content"
    content = message.get("content")
    if content is None:
        content = ""
    content = content if isinstance(content, str) else str(content)
    reasoning, field = _reasoning_of(message)
    if reasoning is None:
        reasoning = ""
    return content, (reasoning if isinstance(reasoning, str) else str(reasoning)), field


def _append_stream_content(
    line: str,
    parts: list[str],
    reasoning_parts: list[str] | None = None,
    ui_reason_parts: list[str] | None = None,
) -> str | None:
    """Append assistant text from one SSE line (delta first, message fallback).

    When reasoning_parts is given (REASONING_IN_KEY on), the line's reasoning
    (delta first, message fallback; reasoning_content, else reasoning — same
    priority as the non-stream path) is appended there too, so the save site
    has the response's reasoning trace. ui_reason_parts is the dashboard's
    always-on reasoning sink (independent of REASONING_IN_KEY). Returns the
    reasoning field name the line used ("reasoning_content" or "reasoning"),
    or None when the line carried no reasoning.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return None
    first = choices[0]
    content = None
    reasoning = None
    field = "reasoning_content"
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        reasoning, field = _reasoning_of(delta)
    if content is None or reasoning is None:
        message = first.get("message")
        if isinstance(message, dict):
            if content is None:
                content = message.get("content")
            if reasoning is None:
                reasoning, field = _reasoning_of(message)
    if isinstance(content, str) and content:
        parts.append(content)
    if isinstance(reasoning, str) and reasoning:
        if reasoning_parts is not None:
            reasoning_parts.append(reasoning)
        if ui_reason_parts is not None:
            ui_reason_parts.append(reasoning)
        return field
    return None


def _stream_usage_of(line: str) -> tuple[dict, dict] | None:
    """(usage, timings) from an SSE line that carries the final usage chunk.

    llama.cpp emits one final SSE chunk with usage/timings (no choices) when
    prompt caching is on. Returns None for content chunks, [DONE] and
    malformed lines. The chunk's timings are kept when present; comma-joined
    keys are not polluted into the assembled content because the usage chunk
    has no delta.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict) or not usage:
        return None
    timings = data.get("timings")
    return usage, timings if isinstance(timings, dict) else {}


async def _saved_conversation_values(
    messages: list[dict] | None,
    response_text: str,
    model_id: str,
    fallback_prefix: str,
    fallback_blocks: list[str],
    fallback_hashes: list[str],
    response_reasoning: str = "",
    response_reasoning_field: str = "reasoning_content",
    render_ctx: dict | None = None,
) -> tuple[str, list[str], list[str]]:
    """Prefix values for the stored conversation, or prompt-only fallback.

    response_reasoning_field is the field name the backend used for the
    reasoning trace; the saved assistant message carries it under that name
    so an echoed continuation request matches. render_ctx is the SAME dict
    the request side hashed with, so the saved meta's last prefix hash equals
    the continuation request's key.
    """
    if not response_text and not (REASONING_IN_KEY and response_reasoning):
        return fallback_prefix, fallback_blocks, fallback_hashes
    return await asyncio.to_thread(
        hs.saved_conversation_values,
        messages,
        response_text,
        model_id,
        WORDS_PER_BLOCK,
        REASONING_IN_KEY,
        response_reasoning or None,
        response_reasoning_field,
        render_ctx,
    )


async def _log_prefix_group(
    rid: str,
    ts: str,
    messages: list[dict],
    model_id: str,
    response_text: str,
    response_reasoning: str,
    response_reasoning_field: str,
    prefix: str,
    blocks: list[str],
    prefix_hashes: list[str],
    render_ctx: dict | None = None,
) -> None:
    """Write the group's prefix.json (fire-and-forget, never raises).

    The prefix is the conversation actually stored (prompt + assistant
    response): the text the next message's request will be matched against.
    Used where the saved values are not computed elsewhere (small requests,
    interrupted streams); big completed requests reuse the values the
    background save already computed.
    """
    try:
        saved_prefix, _saved_blocks, saved_hashes = await _saved_conversation_values(
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
    except Exception as e:  # noqa: BLE001
        log.warning("reqlog_prefix_fail rid=%s: %s", rid, e)
        return
    reqlog.log_file(
        "prefix",
        rid,
        ts,
        {"prefix": saved_prefix, "key": saved_hashes[-1] if saved_hashes else None},
    )


async def _purge_backend_files(
    clients: list[LlamaClient], key_models: list[tuple[str, str | None]]
) -> None:
    # Backend .bin files are purged best-effort. llama.cpp intentionally has
    # no endpoint to delete files from --slot-save-path (the `erase` slot
    # action only clears in-memory state), so when the save directory is
    # mounted we remove the .bin files directly; the DELETE call stays as a
    # fallback for plain backends without a mount. A router backend needs the
    # model to route the delete to the right child.
    for key, model_id in key_models:
        for client in clients:
            await client.delete_cache_file(key, model=model_id)
        if BIN_CACHE_DIR:
            await asyncio.to_thread(bin_cache.delete_bin_file, BIN_CACHE_DIR, key)


def _schedule_lru_check() -> None:
    """Fire-and-forget LRU check after a slot write (save).

    Must not delay the response: the cleanup runs in a background task.
    Skipped when the .bin cache is disabled or a check is already in
    flight (concurrent runs would race on the same files). Call it only
    after the meta is written: a .bin without a meta looks orphaned and
    would be deleted by the very check it triggered.
    """
    global _lru_check_in_flight
    if not BIN_CACHE_DIR or BIN_CACHE_MAX_MB <= 0 or _lru_check_in_flight:
        return
    _lru_check_in_flight = True

    async def _run() -> None:
        # `global` is required: without it the assignment in the finally
        # would create a local of this closure, not reset the module flag.
        global _lru_check_in_flight
        try:
            await asyncio.to_thread(
                bin_cache.clean_bin_cache, BIN_CACHE_DIR, BIN_CACHE_MAX_MB
            )
        except Exception as e:  # noqa: BLE001
            log.warning("lru_check_error: %s", e)
        finally:
            _lru_check_in_flight = False

    task = asyncio.create_task(_run())
    _LRU_TASKS.add(task)
    task.add_done_callback(_LRU_TASKS.discard)


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
    _register_inflight_save(key)
    try:
        try:
            ok = await sm.save_after(g, key)
        except Exception as e:  # noqa: BLE001
            log.warning("save_after_exception g=%s key=%s: %s", g, key[:16], e)
            promstats.saves_total.labels(model=model_id, outcome="save_error").inc()
            _set_save_outcome(decision, success=False, phase="save_error")
            return False
        if not ok:
            promstats.saves_total.labels(model=model_id, outcome="save_failed").inc()
            _set_save_outcome(decision, success=False, phase="save_failed")
            return False
        bin_size = bin_cache.get_bin_size(BIN_CACHE_DIR, key) if BIN_CACHE_DIR else None
        if (
            bin_size is not None
            and MIN_BIN_SIZE_VALID > 0
            and bin_size < MIN_BIN_SIZE_VALID * 1024 * 1024
        ):
            log.warning(
                "save_empty_capture g=%s key=%s bin_size=%d discard",
                g,
                key[:16],
                bin_size,
            )
            try:
                bin_cache.delete_bin_file(BIN_CACHE_DIR, key)
            except Exception:  # noqa: BLE001
                pass
            promstats.saves_total.labels(model=model_id, outcome="empty_capture").inc()
            _set_save_outcome(decision, success=False, phase="empty_capture", bin_size_bytes=bin_size)
            return False
        meta_written = False
        try:
            await hs.write_meta_async(
                key,
                prefix,
                blocks,
                WORDS_PER_BLOCK,
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
                        if _PENDING_RESTORES.get(k, 0) > 0:
                            _RESTORE_ALIAS[k] = key
                            log.info(
                                "restore_substituted pending=%s replacement=%s",
                                k[:16],
                                key[:16],
                            )
                    # An existing alias whose target was just deleted would
                    # otherwise terminate at a purged cache; re-point it to
                    # this live key so _resolve_restore_key stays valid.
                    deleted_set = set(deleted)
                    for x, target in list(_RESTORE_ALIAS.items()):
                        if target in deleted_set:
                            _RESTORE_ALIAS[x] = key
                            log.info(
                                "restore_alias_repointed alias=%s replacement=%s",
                                x[:16],
                                key[:16],
                            )
                    await _purge_backend_files(
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
        _set_save_outcome(decision, success=True, meta_written=meta_written, bin_size_bytes=bin_size)
        _schedule_lru_check()
        return True
    finally:
        _finish_inflight_save(key)


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
        _register_inflight_save(key)
        try:
            saved_prefix, saved_blocks, saved_hashes = (
                await _saved_conversation_values(
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
        ok = await _save_and_write_meta(
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
        _finish_inflight_save(key)
        if decision is not None:
            decision.setdefault("save", {"attempted": True, "success": False, "phase": "save_error"})
            _emit_decision(decision, rid, ts)
        # Release is guaranteed: a synchronous call that cannot be
        # interrupted; it runs even if a re-cancellation interrupts the save.
        log.info("slot_release g=%s key=%s via=bg_save", g, key[:16])
        sm.release(g)


async def start_stream_task(
    resp: httpx.Response,
    g: GSlot,
    key: str,
    prefix: str,
    blocks: list[str],
    model_id: str,
    sm: SlotManager,
    is_big: bool,
    prefix_hashes: list[str] | None = None,
    clients: list[LlamaClient] | None = None,
    messages: list[dict] | None = None,
    rid: str = "",
    ts: str = "",
    decision: dict | None = None,
    render_ctx: dict | None = None,
    t0: float | None = None,
) -> AsyncGenerator[bytes, None]:
    """t0: the request start (time.monotonic); when given, the reader records
    the request duration, TTFT, outcome and token metrics in its finally."""
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)

    async def reader():
        log.info("stream_reader_start g=%s key=%s", g, key[:16])
        # True only if the stream was read to the end without exceptions or
        # cancellation: a partial KV cache is useless for restore and just
        # wastes disk.
        completed = False
        push_failed = False
        # Set when the stream was interrupted by a backend failure (any
        # exception, including a CancelledError raised from inside
        # resp.aiter_raw() by a transport/backend cancel): the finally below
        # then pushes the SSE error event. A client-disconnect cancellation
        # (gen's finally cancels the reader task) leaves this unset: the
        # client is already gone, so no error event is emitted.
        error_reason: str | None = None
        # Always decode: besides the assembled content (big requests) the raw
        # SSE lines are captured for the request group's raw.json (all sizes).
        decoder = codecs.getincrementaldecoder("utf-8")()
        sse_buffer = ""
        response_parts: list[str] = []
        raw_parts: list[str] = []
        # Response reasoning trace (REASONING_IN_KEY only): captured from the
        # delta chunks so the save site can fold it into the saved
        # conversation; None keeps the flag-off path byte-identical.
        reasoning_parts: list[str] | None = [] if REASONING_IN_KEY else None
        # The reasoning field name the backend streamed (reasoning_content or
        # reasoning): the saved assistant message must carry the trace under
        # the same name the client received and echoes back.
        reasoning_field = "reasoning_content"
        # Always-on reasoning sink for the live dashboard (independent of
        # REASONING_IN_KEY, which gates the save path).
        ui_reason_parts: list[str] = []
        # usage/timings from the final SSE chunk (llama.cpp, prompt caching on):
        # the ground-truth for "was the whole prompt reprocessed" — cached_tokens
        # must match the restored prefix. None until/unless the backend sent it.
        stream_usage: dict | None = None
        stream_timings: dict | None = None
        # Time to first token (request start to first backend chunk), or None
        # when no chunk arrived (or t0 was not provided).
        ttft: float | None = None
        # True when the reader was cancelled from outside (client disconnect):
        # set in the CancelledError handler, read in the finally to pick the
        # request outcome.
        externally_cancelled = False
        # Dashboard feed: parse the line once and forward the new content /
        # reasoning deltas to the live UI (no-op when the UI is off).
        ui_content_n = 0
        ui_reason_n = 0

        def _feed(line: str) -> str | None:
            nonlocal ui_content_n, ui_reason_n
            field = _append_stream_content(
                line, response_parts, reasoning_parts, ui_reason_parts
            )
            content = (
                response_parts[-1] if len(response_parts) > ui_content_n else ""
            )
            reason = (
                ui_reason_parts[-1] if len(ui_reason_parts) > ui_reason_n else ""
            )
            if content or reason:
                ui_obs.req_tokens(rid, content, reason)
            ui_content_n = len(response_parts)
            ui_reason_n = len(ui_reason_parts)
            return field

        try:
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
                if ttft is None and t0 is not None:
                    ttft = time.monotonic() - t0
                    ui_obs.req_ttft(rid, ttft)
                try:
                    await asyncio.wait_for(queue.put(chunk), timeout=STREAM_PUT_TIMEOUT)
                except asyncio.TimeoutError:
                    # Consumer gone (the queue is not drained): stop pushing
                    # and move on to cleanup. The stream was not read to the
                    # end, so the KV cache must not be saved. Signal the
                    # truncation via the single error-event push site in the
                    # finally below (bounded wait; dropped if the consumer is
                    # still stalled).
                    push_failed = True
                    error_reason = "consumer stalled: stream aborted after waiting for client"
                    log.warning("stream_reader_put_timeout g=%s key=%s", g, key[:16])
                    break
                sse_buffer += decoder.decode(chunk)
                while "\n" in sse_buffer:
                    line, sse_buffer = sse_buffer.split("\n", 1)
                    raw_parts.append(line + "\n")
                    field = _feed(line)
                    if field:
                        reasoning_field = field
                    u = _stream_usage_of(line)
                    if u is not None:
                        stream_usage, stream_timings = u
            sse_buffer += decoder.decode(b"", True)
            if sse_buffer:
                raw_parts.append(sse_buffer)
                field = _feed(sse_buffer)
                if field:
                    reasoning_field = field
                u = _stream_usage_of(sse_buffer)
                if u is not None:
                    stream_usage, stream_timings = u
            completed = not push_failed
            if stream_usage is not None:
                ui_obs.req_usage(rid, stream_usage)
        except asyncio.CancelledError:
            # Distinguish the two cancellation sources: a client disconnect
            # cancels this task from outside (gen's finally), so the task's
            # cancel counter is > 0; a backend/transport cancel raises
            # CancelledError from inside resp.aiter_raw() with no external
            # cancel(), so the counter is 0. Only the latter is a mid-stream
            # failure to signal (on a disconnect the client is already gone
            # and a push would just hit the bounded-wait timeout).
            task = asyncio.current_task()
            externally_cancelled = task is not None and task.cancelling() > 0
            if not externally_cancelled:
                error_reason = "stream cancelled by backend"
            log.warning(
                "stream_reader_cancelled g=%s key=%s external=%s",
                g,
                key[:16],
                externally_cancelled,
            )
            raise
        except Exception as e:
            log.exception("stream_reader_error g=%s key=%s", g, key[:16])
            error_reason = str(e)
        finally:
            # Request metrics, recorded first (synchronously, before any await
            # below) so a cancellation delivered during cleanup cannot skip
            # them. Outcome: a client-disconnect cancellation is reported as
            # client_disconnect; any other interruption (backend error, push
            # timeout) is an error; a clean read-to-the-end is ok.
            if t0 is not None:
                if externally_cancelled:
                    outcome = "client_disconnect"
                elif error_reason is not None:
                    outcome = "error"
                else:
                    outcome = "ok"
                promstats.requests_total.labels(
                    model=model_id, stream="true", outcome=outcome
                ).inc()
                promstats.request_duration_seconds.labels(
                    model=model_id, stream="true"
                ).observe(time.monotonic() - t0)
                if ttft is not None:
                    promstats.ttft_seconds.labels(
                        model=model_id, stream="true"
                    ).observe(ttft)
                _record_tokens(model_id, stream_usage)
            # Signal the mid-stream failure to the client exactly once (single
            # push site, guarded by error_reason): without this the stream
            # would end silently (no [DONE], no error) and the client could
            # not distinguish a truncated stream from a normal one.
            if error_reason is not None:
                try:
                    payload = json.dumps(
                        {"error": f"stream interrupted: {error_reason}"}
                    )
                    await asyncio.wait_for(
                        queue.put(f"data: {payload}\n\n".encode()),
                        timeout=STREAM_PUT_TIMEOUT,
                    )
                except Exception as push_err:  # noqa: BLE001
                    # No consumer left (put timed out) or encoding failed:
                    # nothing to signal; proceed to cleanup.
                    log.warning(
                        "stream_error_event_failed g=%s key=%s: %s",
                        g,
                        key[:16],
                        push_err,
                    )
            # --- request group logging (response + raw SSE, always) ---
            # Fire-and-forget: logging must not delay the slot release or the
            # sentinel. Big completed streams get their prefix.json from the
            # background save below (it reuses the values it computes); every
            # other path (small, interrupted) computes them in its own task.
            if rid and ts:
                resp_payload: dict = {
                    "content": "".join(response_parts),
                    "reasoning": "".join(reasoning_parts or []),
                    "completed": completed,
                    "error": error_reason,
                }
                if stream_usage is not None:
                    resp_payload["usage"] = stream_usage
                    if stream_timings:
                        resp_payload["timings"] = stream_timings
                reqlog.log_file("response", rid, ts, resp_payload)
                reqlog.log_file("raw", rid, ts, {"sse": "".join(raw_parts)})
                if not (completed and is_big):
                    ptask = asyncio.create_task(
                        _log_prefix_group(
                            rid,
                            ts,
                            messages or [],
                            model_id,
                            "".join(response_parts),
                            "".join(reasoning_parts or []),
                            reasoning_field,
                            prefix,
                            blocks,
                            prefix_hashes or [],
                            render_ctx,
                        )
                    )
                    _BG_SAVE_TASKS.add(ptask)
                    ptask.add_done_callback(_BG_SAVE_TASKS.discard)
            # --- slot ownership transfer ---
            # A big completed stream must save the KV cache, but the save
            # cannot run inline here: a client disconnect (hermes closes the
            # SSE connection after receiving content) cancels the reader
            # task, and the CancelledError raised at a save await point
            # (_saved_conversation_values / _save_and_write_meta) silently
            # kills the save — no .bin or meta is written and every next
            # request reprocesses from scratch.  Mirror the non-stream path
            # (see _background_save): hand off the slot to a detached task
            # that owns the lock and releases it in its own finally.
            handed_off = False
            # Create the detached save BEFORE the only await below (aclose):
            # create_task is synchronous, so by the time aclose suspends the
            # reader the save already exists and handed_off is set. A
            # client-disconnect CancelledError delivered at aclose then skips
            # nothing — the save survives the reader's cancellation and owns
            # the slot. (Creating it after aclose lost the save to exactly that
            # cancellation: handed_off stayed False and the slot was released
            # with no .bin/meta written.)
            if completed and is_big:
                save_task = asyncio.create_task(
                    _background_save(
                        clients or [],
                        sm,
                        g,
                        key,
                        prefix,
                        blocks,
                        prefix_hashes or [],
                        model_id,
                        messages or [],
                        "".join(response_parts),
                        "".join(reasoning_parts or []),
                        reasoning_field,
                        rid,
                        ts,
                        decision=decision,
                        render_ctx=render_ctx,
                    )
                )
                _BG_SAVE_TASKS.add(save_task)
                save_task.add_done_callback(_BG_SAVE_TASKS.discard)
                handed_off = True
                log.info(
                    "stream_reader_save_handoff g=%s key=%s",
                    g,
                    key[:16],
                )
            else:
                log.info(
                    "stream_reader_done g=%s key=%s saved=%s completed=%s",
                    g,
                    key[:16],
                    False,
                    completed,
                )
                if decision is not None:
                    decision["save"] = {"attempted": False}
                    _emit_decision(decision, rid, ts)
            try:
                try:
                    await resp.aclose()
                except Exception:  # noqa: BLE001, S110
                    pass
            finally:
                if not handed_off:
                    log.info("slot_release g=%s key=%s via=stream", g, key[:16])
                    sm.release(g)
            # The client has received everything (or the stream failed): the
            # request is over from the dashboard's point of view, even when a
            # big-completed stream hands the slot to a background save.
            ui_obs.req_end(
                rid,
                status=(
                    ui_obs.STATUS_ERROR
                    if error_reason
                    else ui_obs.STATUS_CANCELLED
                    if externally_cancelled
                    else ui_obs.STATUS_DONE
                ),
                error=error_reason,
            )
            # Sentinel with bounded wait: if there is no consumer, do not
            # block (the slot is already released).
            try:
                await asyncio.wait_for(queue.put(None), timeout=STREAM_PUT_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning("stream_reader_sentinel_timeout g=%s key=%s", g, key[:16])

    reader_task = asyncio.create_task(reader())
    _READER_TASKS.add(reader_task)
    reader_task.add_done_callback(_READER_TASKS.discard)

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            # Consumer gone (client disconnect / generator closed): stop the
            # reader; its finally closes the backend response and frees the slot.
            reader_task.cancel()

    return gen()


async def chat_flow(
    sm: SlotManager, clients: list[LlamaClient], data: dict
) -> Response:
    """Run the chat request pipeline for an already-parsed JSON body.

    `sm` and `clients` are the app's slot manager and backend clients (read
    from app.state by the endpoint); `data` is the validated request body.
    Returns the FastAPI response (JSON or streaming).
    """
    t0 = time.time()
    t0_mono = time.monotonic()

    messages: list[dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))
    # A client opts a request out of the KV cache by sending cache_prompt:false
    # in the body. Such a request is proxied onto a free/oldest slot untouched:
    # no restore search, no pre-chat erase, and no save of bin/meta. Absent or
    # true keeps the normal big/small cache behavior.
    no_cache = data.get("cache_prompt") is False

    # Request group id: the correlation id (X-Request-ID or generated) plus a
    # millisecond timestamp; every file of this request shares the prefix.
    # The request body is logged up front so even failed requests are kept.
    rid = request_id_var.get()
    ts = reqlog.new_group_ts()
    reqlog.log_file("request", rid, ts, data)

    # Effective model: the client's model, else the loaded model, else MODEL_ID.
    # This single value drives the cache key, the slot pool, and the request.
    client_model = data.get("model") or None
    if client_model:
        effective_model = client_model
        # Model aliases: llama.cpp resolves aliases (e.g. "default") to the
        # real loaded model, but the proxy discovers slot pools only under the
        # real id (app._poll_slots). An alias would never find a pool and every
        # such request would collapse onto the bootstrap slot (0, alias, 0) and
        # cache into a separate namespace. When the requested model has no pool
        # and exactly one model is detected, treat the request as that model so
        # pooling, restore and cache keys are shared with real-name requests.
        if not sm.has_pool(effective_model):
            resolved = await clients[0].get_model_id_cached()
            if resolved != "unknown" and sm.discovered_models() == {resolved}:
                effective_model = resolved
                log.info(
                    "model_alias_resolved alias=%s resolved=%s",
                    client_model,
                    resolved,
                )
    else:
        # TTL-cached model id: no per-request HTTP round-trip. "unknown"
        # (never resolved) falls back to MODEL_ID.
        mid = await clients[0].get_model_id_cached()
        effective_model = mid if mid != "unknown" else MODEL_ID

    # All request-side prefix values in one pass, off the event loop: the
    # words are tokenized once (blocks + word count share them) and the
    # per-message prefix hashes are computed incrementally (O(n), not O(n^2)).
    # The render-context leader (tools/reasoning params) is part of the prefix,
    # so differently-rendered conversations never share a key.
    render_ctx = _render_ctx_of(data)
    prefix, key, blocks, prefix_hashes, n_words = await hs.request_prefix_values_async(
        messages, effective_model, WORDS_PER_BLOCK, REASONING_IN_KEY, render_ctx
    )
    is_big = n_words > BIG_THRESHOLD_WORDS
    # A no-cache request skips all cache treatment (restore + save) and the
    # pre-chat erase: it is proxied onto a free/oldest slot untouched.
    do_cache = is_big and not no_cache

    # Live dashboard: register the request (no-op when the UI is off).
    ui_obs.req_start(
        rid,
        model=effective_model,
        stream=stream,
        n_words=n_words,
        is_big=is_big,
        key=key,
        messages=messages,
    )

    def _req_outcome(outcome: str) -> None:
        promstats.requests_total.labels(
            model=effective_model, stream="true" if stream else "false", outcome=outcome
        ).inc()

    # Per-request cache decision, threaded through the request pipeline and
    # finally persisted as decision.json by whichever task finishes last
    # (restore phase here, save outcome in the background save or the stream
    # reader). Skipped (no emit) when neither a handler path completes.
    decision: dict = {
        "is_big": is_big,
        "no_cache": no_cache,
        "n_words": n_words,
        "words_threshold": BIG_THRESHOLD_WORDS,
        "model": effective_model,
        "render_ctx_sha256": hs.render_ctx_digest(render_ctx) or None,
        "restore": {
            "candidate_key": None,
            "candidate_ratio": None,
            "used_key": None,
            "outcome": None,
            "stale_meta_dropped": False,
        },
        "wait_inflight_save": False,
        "erase_done": False,
        "slot": None,
        "slot_before_chat": None,
    }

    # Per-message prefix hashes (last == key): the meta is findable by any of
    # its prefixes, and a continuation supersedes its strict prefixes. Only
    # big requests restore or save, so the candidate search is lazy.
    restore_key: str | None = None
    restore_ratio: float | None = None
    if do_cache:
        cand = await _find_restore_candidate(prefix_hashes, blocks, effective_model)
        if cand is None and await _wait_for_inflight_save(prefix_hashes):
            decision["wait_inflight_save"] = True
            # The previous message's save finished while we waited: its meta
            # is on disk now, search again before declaring a miss.
            cand = await _find_restore_candidate(prefix_hashes, blocks, effective_model)
            promstats.inflight_save_waits_total.labels(
                model=effective_model, result="hit" if cand else "miss"
            ).inc()
        if cand:
            restore_key, restore_ratio = cand
            decision["restore"]["candidate_key"] = restore_key
            decision["restore"]["candidate_ratio"] = restore_ratio
            hs.record_hit(effective_model)
            promstats.restore_ratio.labels(model=effective_model).observe(restore_ratio)
            # Refresh the meta timestamp so an actively used entry survives
            # TTL eviction.
            await asyncio.to_thread(hs.touch_meta, restore_key)
            log.info(
                "restore_candidate basename=%s ratio=%.3f",
                restore_key[:16],
                restore_ratio,
            )
        else:
            hs.record_miss(effective_model)
            log.info("restore_candidate none")
    elif no_cache:
        log.info("no_cache_request n_words=%d (proxied without cache)", n_words)
    else:
        log.info(
            "small_request n_words=%d threshold=%d",
            n_words,
            BIG_THRESHOLD_WORDS,
        )

    log.info(
        "before_acquire is_big=%s restore_key=%s",
        is_big,
        restore_key[:16] if restore_key else None,
    )

    # Only a big request with a selected candidate can restore, so only it
    # needs a pending entry: a concurrent subsumed deletion of its key must be
    # able to substitute the replacement cache before the slot is acquired.
    if do_cache and restore_key is not None:
        _register_pending_restore(restore_key)
    # Refresh the model's slot pools before picking a slot: a slot cut/reload on
    # the backend is otherwise only seen at the next periodic poll, and a stale
    # slot_id wraps onto a live physical slot (id % n_slots), risking a save/
    # restore collision. Rate-limited and non-fatal inside freshen_model.
    await sm.freshen_model(effective_model)
    t_acq = time.monotonic()
    try:
        try:
            g, _lock, restored, used_key = await asyncio.wait_for(
                sm.acquire_for_request(
                    effective_model,
                    restore_key if do_cache else None,
                    resolve_restore_key=_resolve_restore_key if do_cache else None,
                ),
                timeout=ACQUIRE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.error(
                "acquire_timeout is_big=%s restore_key=%s",
                is_big,
                restore_key[:16] if restore_key else None,
            )
            promstats.requests_total.labels(
                model=effective_model, stream="true" if stream else "false",
                outcome="acquire_timeout",
            ).inc()
            ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error="acquire_timeout")
            return JSONResponse(
                {"error": "all slots busy, please retry later"},
                status_code=503,
            )
    finally:
        if do_cache and restore_key:
            _unregister_pending_restore(restore_key)
    promstats.slot_wait_seconds.labels(model=effective_model).observe(
        time.monotonic() - t_acq
    )

    log.info("after_acquire g=%s key=%s restored=%s", g, key[:16], restored)

    be_id, _mid, slot_id = g
    client = clients[be_id]
    decision["restore"]["used_key"] = used_key
    decision["restore"]["outcome"] = restored
    decision["slot"] = {"backend": be_id, "model": effective_model, "id": slot_id}
    ui_obs.req_slot(rid, be_id, effective_model, slot_id)
    decision["slot_before_chat"] = await _snapshot_slot(client, slot_id, effective_model)

    # A restore is only attempted when a key was selected, so both branches
    # below imply used_key is a non-None string: the key actually restored
    # (the original candidate, or its substitution alias target).
    if restored == RESTORE_MISSING and used_key:
        # The backend explicitly reported the cache file is gone (404): the
        # meta is stale, delete it so every big request does not repeat a
        # hopeless restore. With substitution, used_key is the replacement
        # whose .bin is gone, not the original candidate (whose meta the
        # subsumption that created the alias already deleted).
        try:
            await hs.delete_meta_async(used_key)
            decision["restore"]["stale_meta_dropped"] = True
            promstats.stale_meta_drops_total.labels(model=effective_model).inc()
            promstats.evictions_total.labels(reason="stale").inc()
            log.info("stale_meta_dropped key=%s", used_key[:16])
        except Exception as e:  # noqa: BLE001
            log.warning("delete_meta_failed key=%s: %s", used_key[:16], e)
    elif restored is not True and restored != RESTORE_MISSING and used_key:
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
    erase_done = (not no_cache) and restored is not True and ERASE_BEFORE_CHAT
    decision["erase_done"] = erase_done
    if erase_done:
        await client.erase_slot(slot_id, model=effective_model)

    body = dict(data)
    body["model"] = effective_model
    body["cache_prompt"] = bool(do_cache)
    body["n_keep"] = -1

    opts = dict(body.get("options") or {})
    opts["slot_id"] = slot_id
    opts["id_slot"] = slot_id
    opts["n_keep"] = -1
    opts["cache_prompt"] = bool(do_cache)
    body["options"] = opts

    log.info(
        "dispatch be=%d slot=%d is_big=%s (restore_target=%s restored=%s model_id=%s)",
        be_id,
        slot_id,
        is_big,
        restore_key[:16] if restore_key else None,
        restored,
        effective_model,
    )

    # The slot is released exactly once per request:
    # - successful stream: the reader task releases it (the generator owns
    #   the slot from this point on);
    # - successful big non-stream: the background save task releases it;
    # - every other path (errors, exceptions, cancellation): the finally below.
    task_owns_slot = False
    try:
        if stream:
            resp = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=True,
            )
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                promstats.requests_total.labels(
                    model=effective_model, stream="true", outcome="error"
                ).inc()
                ui_obs.req_end(
                    rid,
                    status=ui_obs.STATUS_ERROR,
                    error=f"backend {resp.status_code}",
                )
                return JSONResponse(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status_code=_provider_error_status(resp.status_code),
                )

            gen = await start_stream_task(
                resp,
                g,
                key,
                prefix,
                blocks,
                effective_model,
                sm,
                do_cache,
                prefix_hashes,
                clients,
                messages,
                rid,
                ts,
                decision,
                render_ctx=render_ctx,
                t0=t0_mono,
            )
            task_owns_slot = True

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers=headers,
            )

        else:
            out = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=False,
            )
            if not isinstance(out, dict):
                _req_outcome("error")
                ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR,
                              error="provider non-JSON body")
                return JSONResponse(
                    {"error": "provider non-JSON body"},
                    status_code=502,
                )

            # The client maps provider failures to {"object": "error", ...};
            # such a body must not be returned to the caller as HTTP 200.
            if out.get("object") == "error":
                log.error(
                    "provider_error key=%s status=%s message=%s",
                    key[:16],
                    out.get("status"),
                    out.get("message"),
                )
                body = {"error": out.get("message") or "provider error"}
                if out.get("raw"):
                    body["raw"] = out["raw"]
                _req_outcome("error")
                ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR,
                              error=str(out.get("message") or "provider error"))
                return JSONResponse(
                    body,
                    status_code=_provider_error_status(out.get("status")),
                )

            # The full backend body is the group's response.json (both sizes);
            # the assistant content is extracted for the prefix computation.
            response_text, response_reasoning, response_reasoning_field = (
                _assistant_content(out)
            )
            reqlog.log_file("response", rid, ts, out)
            # Non-stream: the whole answer arrived at once — feed it to the
            # dashboard in a single batch (no incremental visibility exists).
            ui_obs.req_tokens(rid, response_text, response_reasoning)
            if isinstance(out.get("usage"), dict):
                ui_obs.req_usage(rid, out["usage"])
            ui_obs.req_ttft(rid, time.monotonic() - t0_mono)
            if do_cache:
                # Save + meta + subsumed-meta cleanup (see the stream reader)
                # runs in the background: the .bin write must not delay the
                # JSON response. The task owns the slot and releases it, and
                # writes the group's prefix.json (reusing its saved values).
                save_task = asyncio.create_task(
                    _background_save(
                        clients,
                        sm,
                        g,
                        key,
                        prefix,
                        blocks,
                        prefix_hashes,
                        effective_model,
                        messages,
                        response_text,
                        response_reasoning,
                        response_reasoning_field,
                        rid,
                        ts,
                        decision=decision,
                        render_ctx=render_ctx,
                    )
                )
                _BG_SAVE_TASKS.add(save_task)
                save_task.add_done_callback(_BG_SAVE_TASKS.discard)
                task_owns_slot = True
            else:
                # No background save for small requests: the saved
                # conversation (the next message's match target) is computed
                # in its own task so the response is not delayed.
                decision["save"] = {"attempted": False}
                _emit_decision(decision, rid, ts)
                ptask = asyncio.create_task(
                    _log_prefix_group(
                        rid,
                        ts,
                        messages,
                        effective_model,
                        response_text,
                        response_reasoning,
                        response_reasoning_field,
                        prefix,
                        blocks,
                        prefix_hashes,
                        render_ctx,
                    )
                )
                _BG_SAVE_TASKS.add(ptask)
                ptask.add_done_callback(_BG_SAVE_TASKS.discard)

            # Non-stream: the whole answer arrived at once, so the time to
            # first token equals the total duration.
            _record_tokens(effective_model, out.get("usage"))
            dur = time.monotonic() - t0_mono
            promstats.request_duration_seconds.labels(
                model=effective_model, stream="false"
            ).observe(dur)
            promstats.ttft_seconds.labels(
                model=effective_model, stream="false"
            ).observe(dur)
            _req_outcome("ok")
            log.info(
                "json_done g=%s key=%s is_big=%s dur_ms=%d",
                g,
                key[:16],
                is_big,
                int((time.time() - t0) * 1000),
            )
            ui_obs.req_end(rid)
            return JSONResponse(content=out, status_code=200)

    except httpx.HTTPError as e:
        # A connect/timeout failure: no backend response at all, so this is a
        # genuine upstream failure (502), not a proxy bug (500).
        log.exception("chat_upstream_error g=%s key=%s", g, key[:16])
        _req_outcome("error")
        ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error=str(e))
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        log.exception("chat_error g=%s key=%s", g, key[:16])
        _req_outcome("error")
        ui_obs.req_end(rid, status=ui_obs.STATUS_ERROR, error=str(e))
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if not task_owns_slot:
            log.info("slot_release g=%s key=%s via=finally", g, key[:16])
            sm.release(g)
            # Every path that did not hand the slot to a save/reader task ends
            # here: error returns (provider failure, upstream error, exception)
            # now persist the restore-phase diagnostics too — the very
            # decision.json needed to debug failures. Save was never attempted,
            # so mark it explicitly; _emit_decision is idempotent and no-ops
            # when this request already emitted one.
            decision.setdefault("save", {"attempted": False})
            _emit_decision(decision, rid, ts)
