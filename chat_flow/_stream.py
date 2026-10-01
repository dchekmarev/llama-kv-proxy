# chat_flow/_stream.py

"""The streaming reader task and its bounded-queue client generator."""

import asyncio
import codecs
import json
import time
from collections.abc import AsyncGenerator

import httpx

import chat_flow
import promstats
import reqlog
import ui as ui_obs
from llama_client import LlamaClient
from slot_manager import GSlot, SlotManager

from . import _state

log = _state.log


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
    metric_model: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """t0: the request start (time.monotonic); when given, the reader records
    the request duration, TTFT, outcome and token metrics in its finally.

    metric_model: the bounded label for the model metrics. Defaults to model_id,
    which is only safe when the caller knows the name resolved to a real model
    id; an unresolved client alias must pass the fixed bucket instead."""
    label = promstats.model_label(metric_model or model_id)
    queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=chat_flow.STREAM_QUEUE_SIZE)

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
        reasoning_parts: list[str] | None = [] if chat_flow.REASONING_IN_KEY else None
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
            field = chat_flow._append_stream_content(
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
                    await asyncio.wait_for(queue.put(chunk), timeout=chat_flow.STREAM_PUT_TIMEOUT)
                except TimeoutError:
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
                    u = chat_flow._stream_usage_of(line)
                    if u is not None:
                        stream_usage, stream_timings = u
            sse_buffer += decoder.decode(b"", True)
            if sse_buffer:
                raw_parts.append(sse_buffer)
                field = _feed(sse_buffer)
                if field:
                    reasoning_field = field
                u = chat_flow._stream_usage_of(sse_buffer)
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
                    model=label, stream="true", outcome=outcome
                ).inc()
                promstats.request_duration_seconds.labels(
                    model=label, stream="true"
                ).observe(time.monotonic() - t0)
                if ttft is not None:
                    promstats.ttft_seconds.labels(
                        model=label, stream="true"
                    ).observe(ttft)
                chat_flow._record_tokens(label, stream_usage)
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
                        timeout=chat_flow.STREAM_PUT_TIMEOUT,
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
                        chat_flow._log_prefix_group(
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
                    chat_flow._BG_SAVE_TASKS.add(ptask)
                    ptask.add_done_callback(chat_flow._BG_SAVE_TASKS.discard)
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
                    chat_flow._background_save(
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
                chat_flow._BG_SAVE_TASKS.add(save_task)
                save_task.add_done_callback(chat_flow._BG_SAVE_TASKS.discard)
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
                    chat_flow._emit_decision(decision, rid, ts)
            # The client has received everything (or the stream failed): the
            # request is over from the dashboard's point of view, even when a
            # big-completed stream hands the slot to a background save.
            # Recorded before aclose: a disconnect cancels the reader inside
            # that await, and req_end below it would be skipped, leaving the
            # request active forever (a stuck "generating" row, a busy_rid on a
            # free slot). It is synchronous and pops by rid, so it is safe here
            # and exactly-once.
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
            try:
                try:
                    await resp.aclose()
                except Exception:  # noqa: BLE001, S110
                    pass
            finally:
                if not handed_off:
                    log.info("slot_release g=%s key=%s via=stream", g, key[:16])
                    sm.release(g)
            # Sentinel with bounded wait: if there is no consumer, do not
            # block (the slot is already released).
            try:
                await asyncio.wait_for(queue.put(None), timeout=chat_flow.STREAM_PUT_TIMEOUT)
            except TimeoutError:
                log.warning("stream_reader_sentinel_timeout g=%s key=%s", g, key[:16])

    reader_task = asyncio.create_task(reader())
    chat_flow._READER_TASKS.add(reader_task)
    reader_task.add_done_callback(chat_flow._READER_TASKS.discard)

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
