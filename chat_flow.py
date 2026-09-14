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
- For non-streaming big requests the save+meta runs in a background task
  (_BG_SAVE_TASKS) after the JSON response is returned, so the .bin write
  does not add latency to the response; the task owns and releases the slot.
- For streaming:
    * reading from llama.cpp happens in a separate background task (the
      reader);
    * the reader pushes chunks into an asyncio.Queue (put with bounded wait,
      so it cannot block forever when the consumer is gone);
    * in its finally the reader always does release(g) (release is guaranteed
      even on re-cancellation) and puts a sentinel None into the queue (with
      bounded wait); save_after + write_meta — only if the stream was read to
      the end and the save succeeded;
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
from config import (
    ACQUIRE_TIMEOUT,
    BIG_THRESHOLD_WORDS,
    BIN_CACHE_DIR,
    BIN_CACHE_MAX_MB,
    MIN_BIN_SIZE_VALID,
    ERASE_BEFORE_CHAT,
    LCP_TH,
    MODEL_ID,
    REASONING_IN_KEY,
    SAVE_WAIT_TIMEOUT,
    WORDS_PER_BLOCK,
)
from llama_client import RESTORE_MISSING, LlamaClient
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


def _register_inflight_save(key: str) -> None:
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
    line: str, parts: list[str], reasoning_parts: list[str] | None = None
) -> str | None:
    """Append assistant text from one SSE line (delta first, message fallback).

    When reasoning_parts is given (REASONING_IN_KEY on), the line's reasoning
    (delta first, message fallback; reasoning_content, else reasoning — same
    priority as the non-stream path) is appended there too, so the save site
    has the response's reasoning trace. Returns the reasoning field name the
    line used ("reasoning_content" or "reasoning"), or None when the line
    carried no reasoning.
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
    if reasoning_parts is not None and isinstance(reasoning, str) and reasoning:
        reasoning_parts.append(reasoning)
        return field
    return None


async def _saved_conversation_values(
    messages: list[dict] | None,
    response_text: str,
    model_id: str,
    fallback_prefix: str,
    fallback_blocks: list[str],
    fallback_hashes: list[str],
    response_reasoning: str = "",
    response_reasoning_field: str = "reasoning_content",
) -> tuple[str, list[str], list[str]]:
    """Prefix values for the stored conversation, or prompt-only fallback.

    response_reasoning_field is the field name the backend used for the
    reasoning trace; the saved assistant message carries it under that name
    so an echoed continuation request matches.
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
            return False
        if not ok:
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
) -> None:
    """Non-stream big-request save+meta, run after the response is returned.

    The hundreds-of-MB .bin write must not add latency to the JSON response,
    so it runs in a background task (mirroring the stream reader's finally).
    The task owns the slot: it releases it in the finally, even if the save
    or the meta write raises (the client already has its response, so the
    error is only logged).
    """
    try:
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
                )
            )
        except Exception as e:  # noqa: BLE001
            log.warning("saved_conversation_values_fail key=%s: %s", key[:16], e)
            saved_prefix, saved_blocks, saved_hashes = (
                prefix,
                blocks,
                prefix_hashes,
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
        )
        log.info("bg_save_done g=%s key=%s saved=%s", g, key[:16], ok)
    except Exception as e:  # noqa: BLE001
        log.warning("background_save_error g=%s key=%s: %s", g, key[:16], e)
    finally:
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
) -> AsyncGenerator[bytes, None]:
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
        decoder = codecs.getincrementaldecoder("utf-8")() if is_big else None
        sse_buffer = ""
        response_parts: list[str] = []
        # Response reasoning trace (REASONING_IN_KEY only): captured from the
        # delta chunks so the save site can fold it into the saved
        # conversation; None keeps the flag-off path byte-identical.
        reasoning_parts: list[str] | None = [] if REASONING_IN_KEY else None
        # The reasoning field name the backend streamed (reasoning_content or
        # reasoning): the saved assistant message must carry the trace under
        # the same name the client received and echoes back.
        reasoning_field = "reasoning_content"
        try:
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
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
                if decoder is not None:
                    sse_buffer += decoder.decode(chunk)
                    while "\n" in sse_buffer:
                        line, sse_buffer = sse_buffer.split("\n", 1)
                        field = _append_stream_content(
                            line, response_parts, reasoning_parts
                        )
                        if field:
                            reasoning_field = field
            if decoder is not None:
                sse_buffer += decoder.decode(b"", True)
                if sse_buffer:
                    field = _append_stream_content(
                        sse_buffer, response_parts, reasoning_parts
                    )
                    if field:
                        reasoning_field = field
            completed = not push_failed
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
            try:
                try:
                    await resp.aclose()
                except Exception:  # noqa: BLE001, S110
                    pass
                ok = False
                # Save and meta only for big requests: small ones must not
                # pollute the disk cache.
                if completed and is_big:
                    response_text = "".join(response_parts)
                    response_reasoning = "".join(reasoning_parts or [])
                    try:
                        saved_prefix, saved_blocks, saved_hashes = (
                            await _saved_conversation_values(
                                messages,
                                response_text,
                                model_id,
                                prefix,
                                blocks,
                                prefix_hashes or [],
                                response_reasoning,
                                reasoning_field,
                            )
                        )
                    except Exception as e:  # noqa: BLE001
                        log.warning(
                            "saved_conversation_values_fail key=%s: %s", key[:16], e
                        )
                        saved_prefix, saved_blocks, saved_hashes = (
                            prefix,
                            blocks,
                            prefix_hashes or [],
                        )
                    ok = await _save_and_write_meta(
                        clients,
                        sm,
                        g,
                        key,
                        saved_prefix,
                        saved_blocks,
                        prefix_hashes or [],
                        model_id,
                        saved_hashes,
                    )
                log.info(
                    "stream_reader_done g=%s key=%s saved=%s completed=%s",
                    g,
                    key[:16],
                    ok,
                    completed,
                )
            finally:
                # Release is guaranteed: a synchronous call that cannot be
                # interrupted; it runs even if a re-cancellation interrupts the save.
                log.info("slot_release g=%s key=%s via=stream", g, key[:16])
                sm.release(g)
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

    messages: list[dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))

    # Effective model: the client's model, else the loaded model, else MODEL_ID.
    # This single value drives the cache key, the slot pool, and the request.
    client_model = data.get("model") or None
    if client_model:
        effective_model = client_model
    else:
        # TTL-cached model id: no per-request HTTP round-trip. "unknown"
        # (never resolved) falls back to MODEL_ID.
        mid = await clients[0].get_model_id_cached()
        effective_model = mid if mid != "unknown" else MODEL_ID

    # All request-side prefix values in one pass, off the event loop: the
    # words are tokenized once (blocks + word count share them) and the
    # per-message prefix hashes are computed incrementally (O(n), not O(n^2)).
    prefix, key, blocks, prefix_hashes, n_words = await hs.request_prefix_values_async(
        messages, effective_model, WORDS_PER_BLOCK, REASONING_IN_KEY
    )
    is_big = n_words > BIG_THRESHOLD_WORDS

    # Per-message prefix hashes (last == key): the meta is findable by any of
    # its prefixes, and a continuation supersedes its strict prefixes. Only
    # big requests restore or save, so the candidate search is lazy.
    restore_key: str | None = None
    if is_big:
        cand = await _find_restore_candidate(prefix_hashes, blocks, effective_model)
        if cand is None and await _wait_for_inflight_save(prefix_hashes):
            # The previous message's save finished while we waited: its meta
            # is on disk now, search again before declaring a miss.
            cand = await _find_restore_candidate(prefix_hashes, blocks, effective_model)
        if cand:
            restore_key, ratio = cand
            hs.record_hit()
            # Refresh the meta timestamp so an actively used entry survives
            # TTL eviction.
            await asyncio.to_thread(hs.touch_meta, restore_key)
            log.info(
                "restore_candidate basename=%s ratio=%.3f",
                restore_key[:16],
                ratio,
            )
        else:
            hs.record_miss()
            log.info("restore_candidate none")
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

    try:
        g, _lock, restored = await asyncio.wait_for(
            sm.acquire_for_request(effective_model, restore_key if is_big else None),
            timeout=ACQUIRE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error(
            "acquire_timeout is_big=%s restore_key=%s",
            is_big,
            restore_key[:16] if restore_key else None,
        )
        return JSONResponse(
            {"error": "all slots busy, please retry later"},
            status_code=503,
        )

    log.info("after_acquire g=%s key=%s restored=%s", g, key[:16], restored)

    # A restore is only attempted when restore_key is set, so both branches
    # below imply restore_key is a non-None string.
    if restored == RESTORE_MISSING and restore_key:
        # The backend explicitly reported the cache file is gone (404): the
        # meta is stale, delete it so every big request does not repeat a
        # hopeless restore.
        try:
            await hs.delete_meta_async(restore_key)
            log.info("stale_meta_dropped key=%s", restore_key[:16])
        except Exception as e:  # noqa: BLE001
            log.warning("delete_meta_failed key=%s: %s", restore_key[:16], e)
    elif restored is False and restore_key:
        # A non-missing restore failure (transient error, backend down): the
        # cache may still be valid and a retry can succeed, so keep the meta.
        log.warning("restore_failed_kept_meta key=%s", restore_key[:16])

    be_id, _mid, slot_id = g
    client = clients[be_id]

    # A successful restore already set the slot's prompt to the correct prefix,
    # so it is left untouched. In every other case (small request, big request
    # with no restore hit, or a failed/missing restore) the slot may still hold
    # a stale or oversized prompt from a previous conversation; starting a chat
    # on top of it can wedge llama.cpp in PROCESSING_PROMPT (the slot never
    # finishes and the server busy-loops). Clear the slot first.
    if restored is not True and ERASE_BEFORE_CHAT:
        await client.erase_slot(slot_id, model=effective_model)

    body = dict(data)
    body["model"] = effective_model
    body["cache_prompt"] = bool(is_big)
    body["n_keep"] = -1

    opts = dict(body.get("options") or {})
    opts["slot_id"] = slot_id
    opts["id_slot"] = slot_id
    opts["n_keep"] = -1
    opts["cache_prompt"] = bool(is_big)
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
                is_big,
                prefix_hashes,
                clients,
                messages,
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
                return JSONResponse(
                    body,
                    status_code=_provider_error_status(out.get("status")),
                )

            if is_big:
                # Save + meta + subsumed-meta cleanup (see the stream reader)
                # runs in the background: the .bin write must not delay the
                # JSON response. The task owns the slot and releases it.
                response_text, response_reasoning, response_reasoning_field = (
                    _assistant_content(out)
                )
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
                    )
                )
                _BG_SAVE_TASKS.add(save_task)
                save_task.add_done_callback(_BG_SAVE_TASKS.discard)
                task_owns_slot = True

            log.info(
                "json_done g=%s key=%s is_big=%s dur_ms=%d",
                g,
                key[:16],
                is_big,
                int((time.time() - t0) * 1000),
            )
            return JSONResponse(content=out, status_code=200)

    except httpx.HTTPError as e:
        # A connect/timeout failure: no backend response at all, so this is a
        # genuine upstream failure (502), not a proxy bug (500).
        log.exception("chat_upstream_error g=%s key=%s", g, key[:16])
        return JSONResponse({"error": str(e)}, status_code=502)
    except Exception as e:
        log.exception("chat_error g=%s key=%s", g, key[:16])
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if not task_owns_slot:
            log.info("slot_release g=%s key=%s via=finally", g, key[:16])
            sm.release(g)
