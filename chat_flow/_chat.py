# chat_flow/_chat.py

"""The /v1/chat/completions request pipeline."""

import asyncio
import time

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

import chat_flow as chat_flow_pkg
import hashing as hs
import promstats
import reqlog
import ui as ui_obs
from llama_client import LlamaClient
from request_id import request_id_var
from slot_manager import SlotManager

from . import _state

log = _state.log


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

    client_model = data.get("model") or None
    effective_model, no_cache = await chat_flow_pkg._resolve_effective_model(
        sm, clients, client_model, no_cache
    )

    # All request-side prefix values in one pass, off the event loop: the
    # words are tokenized once (blocks + word count share them) and the
    # per-message prefix hashes are computed incrementally (O(n), not O(n^2)).
    # The render-context leader (tools/reasoning params) is part of the prefix,
    # so differently-rendered conversations never share a key.
    render_ctx = chat_flow_pkg._render_ctx_of(data)
    prefix, key, blocks, prefix_hashes, n_words = await hs.request_prefix_values_async(
        messages, effective_model, chat_flow_pkg.WORDS_PER_BLOCK, chat_flow_pkg.REASONING_IN_KEY, render_ctx
    )
    is_big = n_words > chat_flow_pkg.BIG_THRESHOLD_WORDS
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

    decision: dict = chat_flow_pkg._new_decision(
        is_big, no_cache, n_words, effective_model, render_ctx
    )

    restore_key = await chat_flow_pkg._select_restore_candidate(
        prefix_hashes, blocks, effective_model, do_cache, no_cache, n_words, decision
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
        chat_flow_pkg._register_pending_restore(restore_key)
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
                    resolve_restore_key=chat_flow_pkg._resolve_restore_key if do_cache else None,
                ),
                timeout=chat_flow_pkg.ACQUIRE_TIMEOUT,
            )
        except TimeoutError:
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
            chat_flow_pkg._unregister_pending_restore(restore_key)
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
    decision["slot_before_chat"] = await chat_flow_pkg._snapshot_slot(client, slot_id, effective_model)

    await chat_flow_pkg._settle_restore(
        client, slot_id, effective_model, restored, used_key, no_cache, decision
    )

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
                    status_code=chat_flow_pkg._provider_error_status(resp.status_code),
                )

            gen = await chat_flow_pkg.start_stream_task(
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
                    status_code=chat_flow_pkg._provider_error_status(out.get("status")),
                )

            # The full backend body is the group's response.json (both sizes);
            # the assistant content is extracted for the prefix computation.
            response_text, response_reasoning, response_reasoning_field = (
                chat_flow_pkg._assistant_content(out)
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
                    chat_flow_pkg._background_save(
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
                chat_flow_pkg._BG_SAVE_TASKS.add(save_task)
                save_task.add_done_callback(chat_flow_pkg._BG_SAVE_TASKS.discard)
                task_owns_slot = True
            else:
                # No background save for small requests: the saved
                # conversation (the next message's match target) is computed
                # in its own task so the response is not delayed.
                decision["save"] = {"attempted": False}
                chat_flow_pkg._emit_decision(decision, rid, ts)
                ptask = asyncio.create_task(
                    chat_flow_pkg._log_prefix_group(
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
                chat_flow_pkg._BG_SAVE_TASKS.add(ptask)
                ptask.add_done_callback(chat_flow_pkg._BG_SAVE_TASKS.discard)

            # Non-stream: the whole answer arrived at once, so the time to
            # first token equals the total duration.
            chat_flow_pkg._record_tokens(effective_model, out.get("usage"))
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
            chat_flow_pkg._emit_decision(decision, rid, ts)
